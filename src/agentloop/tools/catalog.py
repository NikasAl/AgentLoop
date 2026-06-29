"""
ToolCatalog — реестр всех инструментов системы.

Объединяет Layer 1 (core), Layer 2 (discovered), Layer 3 (custom).
Поддерживает поиск по имени, описанию, ключевым словам.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .base import (
    CustomToolResult,
    CustomToolSpec,
    InstallSpec,
    SafetyReport,
    SearchResult,
    ToolCategory,
    ToolDescriptor,
    ToolLayer,
)
from .core import CORE_TOOLS, get_core_tool
from .scanner import SystemScanner


class ToolCatalog:
    """
    Реестр инструментов всех трёх слоёв.

    Использование:
        catalog = ToolCatalog()
        catalog.scan_system()  # заполнить Layer 2
        results = catalog.search("pdf text extract")
    """

    def __init__(self, scanner: SystemScanner | None = None):
        self.scanner = scanner or SystemScanner()
        self._layer1: list[ToolDescriptor] = CORE_TOOLS.copy()
        self._layer2: list[ToolDescriptor] = []
        self._layer3: list[ToolDescriptor] = []
        self._custom_dir = Path.home() / ".agentloop" / "custom_tools"
        self._custom_dir.mkdir(parents=True, exist_ok=True)

    # ─── Заполнение каталога ────────────────────────────────

    def scan_system(self, use_cache: bool = True) -> None:
        """Запускает SystemScanner для заполнения Layer 2."""
        self._layer2 = self.scanner.scan_all(use_cache=use_cache)

    def refresh(self) -> None:
        """Принудительное пересканирование без кеша."""
        self._layer2 = self.scanner.refresh()

    def load_custom(self) -> None:
        """Загружает custom-инструменты из ~/.agentloop/custom_tools/."""
        self._layer3 = []
        for path in self._custom_dir.glob("*.py"):
            meta_path = path.with_suffix(".meta.yaml")
            if not meta_path.exists():
                continue
            import yaml

            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
            self._layer3.append(
                ToolDescriptor(
                    name=meta["name"],
                    layer=ToolLayer.CUSTOM,
                    category=ToolCategory(meta.get("category", "custom")),
                    description=meta.get("description", ""),
                    available=True,
                    input_schema=meta.get("input_schema", {}),
                    output_schema=meta.get("output_schema", {}),
                    example_usage=meta.get("example_usage", ""),
                    script_path=str(path),
                    dependencies=meta.get("dependencies", []),
                    version=meta.get("version", "1.0"),
                    keywords=meta.get("keywords", []),
                    safety_report=SafetyReport(
                        verdict=meta.get("safety_verdict", "needs_review"),
                        network_access_used=meta.get("network_access", False),
                        fs_access=meta.get("fs_access", "unknown"),
                    ),
                )
            )

    # ─── Доступ к слоям ─────────────────────────────────────

    def layer1(self) -> list[ToolDescriptor]:
        """Layer 1: 7 базовых примитивов."""
        return self._layer1.copy()

    def layer2(self) -> list[ToolDescriptor]:
        """Layer 2: обнаруженные системные утилиты."""
        return self._layer2.copy()

    def layer3(self) -> list[ToolDescriptor]:
        """Layer 3: custom Python-инструменты."""
        return self._layer3.copy()

    def all_tools(self) -> list[ToolDescriptor]:
        """Все инструменты всех слоёв."""
        return self._layer1 + self._layer2 + self._layer3

    def available_tools(self) -> list[ToolDescriptor]:
        """Только доступные инструменты (available=True)."""
        return [t for t in self.all_tools() if t.available]

    # ─── Поиск ──────────────────────────────────────────────

    def get(self, name: str) -> ToolDescriptor | None:
        """Точное совпадение по имени или alias."""
        for t in self.all_tools():
            if t.name == name or name in t.aliases:
                return t
        return None

    def get_core(self, name: str) -> ToolDescriptor | None:
        """Только в Layer 1."""
        return get_core_tool(name)

    def search(
        self,
        query: str,
        *,
        layer: ToolLayer | None = None,
        category: ToolCategory | None = None,
        only_available: bool = False,
        limit: int = 20,
    ) -> SearchResult:
        """
        Семантический поиск по описанию и ключевым словам.

        Args:
            query: текст запроса (например, "извлечение текста из PDF")
            layer: фильтр по слою
            category: фильтр по категории
            only_available: только доступные
            limit: максимум результатов
        """
        query_lower = query.lower()
        query_words = set(re.findall(r"\w+", query_lower))

        scored: list[tuple[int, ToolDescriptor]] = []
        for t in self.all_tools():
            if layer and t.layer != layer:
                continue
            if category and t.category != category:
                continue
            if only_available and not t.available:
                continue

            score = self._score_tool(t, query_lower, query_words)
            if score > 0:
                scored.append((score, t))

        scored.sort(key=lambda x: -x[0])
        found = [t for _, t in scored[:limit]]

        return SearchResult(
            query=query,
            found=found,
            not_found=len(found) == 0,
        )

    def _score_tool(self, t: ToolDescriptor, query_lower: str, query_words: set[str]) -> int:
        """Оценка релевантности инструмента запросу. Больше = лучше."""
        score = 0
        name_lower = t.name.lower()
        desc_lower = t.description.lower()
        keywords_lower = [k.lower() for k in t.keywords]

        # Точное совпадение имени
        if query_lower == name_lower:
            score += 100
        elif query_lower in name_lower or name_lower in query_lower:
            score += 30

        # Совпадение по alias
        for alias in t.aliases:
            alias_lower = alias.lower()
            if query_lower == alias_lower:
                score += 80
            elif alias_lower in query_lower:
                score += 20

        # Совпадение ключевых слов
        for kw in keywords_lower:
            if kw in query_words:
                score += 15
            elif kw in query_lower:
                score += 8

        # Совпадение слов из описания
        desc_words = set(re.findall(r"\w+", desc_lower))
        overlap = query_words & desc_words
        score += len(overlap) * 3

        # Бонус за доступность
        if t.available:
            score += 5

        return score

    # ─── Добавление custom ──────────────────────────────────

    def add_custom(self, descriptor: ToolDescriptor) -> None:
        """Добавляет custom-инструмент в каталог."""
        if descriptor.layer != ToolLayer.CUSTOM:
            raise ValueError(f"Tool must be CUSTOM layer, got {descriptor.layer}")
        self._layer3.append(descriptor)

    def custom_dir(self) -> Path:
        """Директория для custom-инструментов."""
        return self._custom_dir

    # ─── Экспорт для промпта ────────────────────────────────

    def summary_for_builder(self) -> str:
        """
        Компактная сводка для системного промпта Builder'а.
        Показывает Layer 1 полностью + количество Layer 2/3.
        """
        lines = ["# Tool Catalog Summary\n"]

        lines.append("## Layer 1 — Core tools (всегда доступны)")
        for t in self._layer1:
            lines.append(f"- **{t.name}**: {t.description}")
        lines.append("")

        avail_l2 = [t for t in self._layer2 if t.available]
        unavail_l2 = [t for t in self._layer2 if not t.available]
        lines.append(
            f"## Layer 2 — Discovered tools ({len(avail_l2)} доступно, {len(unavail_l2)} можно установить)"
        )
        if avail_l2:
            lines.append("Доступные:")
            for t in avail_l2[:30]:  # первые 30, чтобы не раздувать
                lines.append(f"- **{t.name}**: {t.description}")
            if len(avail_l2) > 30:
                lines.append(f"... и ещё {len(avail_l2) - 30}")
        if unavail_l2:
            lines.append("\nМожно установить через Steward:")
            for t in unavail_l2[:10]:
                lines.append(f"- **{t.name}**: {t.description}")
        lines.append("")

        if self._layer3:
            lines.append(f"## Layer 3 — Custom tools ({len(self._layer3)})")
            for t in self._layer3:
                safety = "✓" if t.safety_report and t.safety_report.is_safe else "?"
                lines.append(f"- {safety} **{t.name}** v{t.version}: {t.description}")
        else:
            lines.append("## Layer 3 — Custom tools (пока нет)")

        lines.append(
            "\n---\n"
            "Если нужного инструмента нет — обратись к Steward:\n"
            "- `steward.search(query)` — поиск по описанию в Layer 2/3\n"
            "- `steward.create_custom(spec)` — создать Python-инструмент (Layer 3)"
        )
        return "\n".join(lines)

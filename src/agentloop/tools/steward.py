"""
Steward — гибрид function + agent для управления инструментами.

4 метода:
- list_core() — Layer 1
- search(query) — поиск в Layer 2/3, при необходимости включает agent
- create_custom(spec) — создание custom Python-инструмента с safety-check
- install(spec) — установка системного пакета через HITL

Steward-agent (LLM) включается ТОЛЬКО когда function не нашла подходящий инструмент.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

import yaml

from ..providers import Message, Provider, get_provider
from ..providers.base import Response
from .base import (
    CustomToolResult,
    CustomToolSpec,
    InstallResult,
    InstallSpec,
    SafetyReport,
    SearchResult,
    ToolCategory,
    ToolDescriptor,
    ToolLayer,
)
from .catalog import ToolCatalog
from .safety import SafetyAgent


class Steward:
    """
    Менеджер инструментов. Builder обращается к нему, когда не находит нужного в Layer 1.

    Args:
        catalog: экземпляр ToolCatalog
        llm_provider: провайдер для Steward-agent (обычно local:gemma-4-26b)
        human_approval: требует ли установка пакетов HITL-апрув (default True)
    """

    def __init__(
        self,
        catalog: ToolCatalog,
        llm_provider: Provider | None = None,
        human_approval: bool = True,
    ):
        self.catalog = catalog
        self.llm = llm_provider
        self.human_approval = human_approval
        self.safety = SafetyAgent()

    # ─── 1. list_core ───────────────────────────────────────

    def list_core(self) -> list[ToolDescriptor]:
        """Возвращает Layer 1 (7 базовых примитивов)."""
        return self.catalog.layer1()

    # ─── 2. search ──────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        only_available: bool = False,
    ) -> SearchResult:
        """
        Поиск инструмента по описанию.

        Сначала ищет в Layer 2/3 через function (без LLM).
        Если function не нашла — включает Steward-agent (LLM).
        """
        # Сначала function search
        result = self.catalog.search(
            query,
            layer=ToolLayer.DISCOVERED,
            only_available=only_available,
        )

        if result.found:
            return result

        # Function не нашла — пробуем custom
        custom_result = self.catalog.search(
            query,
            layer=ToolLayer.CUSTOM,
            only_available=only_available,
        )
        if custom_result.found:
            return custom_result

        # Ничего не нашли — включаем agent
        if self.llm is None:
            return SearchResult(
                query=query,
                found=[],
                not_found=True,
                escalated_to_agent=False,
                agent_suggestion="LLM provider not configured for Steward-agent",
            )

        return self._agent_search(query)

    def _agent_search(self, query: str) -> SearchResult:
        """Steward-agent через LLM для сложных запросов."""
        prompt = self._build_agent_prompt(query)
        try:
            response = self.llm.chat(
                messages=[
                    Message(
                        role="system",
                        content=(
                            "Ты — Steward-agent. Анализируешь запрос Builder'а "
                            "и предлагаешь инструменты. Возвращай JSON по схеме."
                        ),
                    ),
                    Message(role="user", content=prompt),
                ],
                model=self._agent_model(),
                temperature=0.3,
                json_mode=True,
                max_tokens=2048,
            )
        except Exception as e:
            return SearchResult(
                query=query,
                found=[],
                not_found=True,
                escalated_to_agent=True,
                agent_suggestion=f"Agent failed: {e}",
            )

        # Парсим ответ агента
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
            return SearchResult(
                query=query,
                found=[],
                not_found=True,
                escalated_to_agent=True,
                agent_suggestion=response.content[:500],
            )

        suggestion = data.get("suggestion", "")
        custom_possible = data.get("custom_tool_possible", False)

        # Если агент предложил custom-инструмент — добавим spec
        custom_spec = None
        if custom_possible and "implementation" in data:
            impl = data["implementation"]
            custom_spec = CustomToolSpec(
                name=impl.get("name", f"custom_{uuid.uuid4().hex[:8]}"),
                description=impl.get("description", suggestion),
                input_schema=impl.get("input_schema", {}),
                output_schema=impl.get("output_schema", {}),
                dependencies=impl.get("dependencies", []),
                implementation_hint=impl.get("approach", ""),
            )

        return SearchResult(
            query=query,
            found=[],
            not_found=True,
            escalated_to_agent=True,
            agent_suggestion=suggestion,
            custom_tool_possible=custom_possible,
        )

    def _build_agent_prompt(self, query: str) -> str:
        available = self.catalog.available_tools()
        available_names = [t.name for t in available][:50]  # компактно

        return f"""Builder запросил инструмент: "{query}"

Доступные инструменты в системе ({len(available)}): {', '.join(available_names)}

Проанализируй запрос и предложи решение:
1. Можно ли использовать один из доступных инструментов напрямую?
2. Можно ли скомбинировать 2-3 инструмента (composite)?
3. Нужен ли custom Python-инструмент?

Верни JSON:
{{
  "suggestion": "текст рекомендации",
  "custom_tool_possible": true/false,
  "implementation": {{
    "name": "latex_normalizer",
    "description": "Нормализация LaTeX",
    "input_schema": {{"latex": "string"}},
    "output_schema": {{"normalized": "string", "is_valid": "boolean"}},
    "dependencies": ["pylatexenc"],
    "approach": "Использовать pylatexenc.latex_parse для AST walk"
  }}
}}

Если custom_tool_possible=false, omit "implementation".
"""

    def _agent_model(self) -> str:
        """Имя модели для Steward-agent."""
        # Используем первую модель провайдера
        models = self.llm.list_models() if self.llm else []
        if models:
            return models[0].name
        return "gemma-4-26b"

    # ─── 3. create_custom ───────────────────────────────────

    def create_custom(
        self,
        spec: CustomToolSpec,
        *,
        code: str | None = None,
        skip_human_approval: bool = False,
    ) -> CustomToolResult:
        """
        Создаёт custom Python-инструмент.

        Если code=None — генерирует код через LLM.
        Если code предоставлен (например, Builder написал сам) — использует его.

        Шаги:
        1. Генерация или приём кода
        2. SafetyAgent анализ
        3. HITL approval (если не skip)
        4. Установка зависимостей
        5. Тестовый запуск (опционально)
        6. Регистрация в catalog
        """
        tool_id = f"custom:{spec.name}_v1"
        script_path = self.catalog.custom_dir() / f"{spec.name}_v1.py"
        meta_path = script_path.with_suffix(".meta.yaml")

        # Шаг 1: код
        if code is None:
            code = self._generate_code(spec)
            if not code:
                return CustomToolResult(
                    tool_id=tool_id,
                    name=spec.name,
                    version="1",
                    script_path=str(script_path),
                    safety_report=SafetyReport(verdict="unsafe", notes="Code generation failed"),
                    status="failed",
                    error="LLM failed to generate code",
                )

        # Шаг 2: safety check
        report = self.safety.analyze(code, script_path=str(script_path))

        # Шаг 3: HITL approval
        if not skip_human_approval and self.human_approval:
            approved = self._hitl_approval(spec, code, report)
            if not approved:
                return CustomToolResult(
                    tool_id=tool_id,
                    name=spec.name,
                    version="1",
                    script_path=str(script_path),
                    safety_report=report,
                    status="failed",
                    error="User rejected",
                )

        # Если safety = unsafe и не skip — отказ
        if report.verdict == "unsafe" and not skip_human_approval:
            return CustomToolResult(
                tool_id=tool_id,
                name=spec.name,
                version="1",
                script_path=str(script_path),
                safety_report=report,
                status="failed",
                error="Safety verdict: unsafe",
            )

        # Шаг 4: сохраняем код
        script_path.write_text(code, encoding="utf-8")

        # Шаг 5: устанавливаем зависимости
        if spec.dependencies:
            for dep in spec.dependencies:
                install_result = self.install(
                    InstallSpec(managers={"pip": dep}, notes=f"Dependency of {spec.name}")
                )
                if not install_result.success:
                    return CustomToolResult(
                        tool_id=tool_id,
                        name=spec.name,
                        version="1",
                        script_path=str(script_path),
                        safety_report=report,
                        status="failed",
                        error=f"Failed to install dependency {dep}: {install_result.error}",
                    )

        # Шаг 6: сохраняем meta
        meta = {
            "name": spec.name,
            "version": "1",
            "description": spec.description,
            "category": "custom",
            "input_schema": spec.input_schema,
            "output_schema": spec.output_schema,
            "dependencies": spec.dependencies,
            "keywords": [spec.name],
            "safety_verdict": report.verdict,
            "network_access": report.network_access_used,
            "fs_access": report.fs_access,
        }
        meta_path.write_text(yaml.safe_dump(meta, allow_unicode=True), encoding="utf-8")

        # Регистрируем в catalog
        descriptor = ToolDescriptor(
            name=spec.name,
            layer=ToolLayer.CUSTOM,
            category=ToolCategory.CUSTOM,
            description=spec.description,
            available=True,
            input_schema=spec.input_schema,
            output_schema=spec.output_schema,
            example_usage=f'python_run("{tool_id}", {{...}})',
            script_path=str(script_path),
            dependencies=spec.dependencies,
            version="1",
            keywords=[spec.name],
            safety_report=report,
        )
        self.catalog.add_custom(descriptor)

        return CustomToolResult(
            tool_id=tool_id,
            name=spec.name,
            version="1",
            script_path=str(script_path),
            safety_report=report,
            status="available",
        )

    def _generate_code(self, spec: CustomToolSpec) -> str | None:
        """Генерирует Python-код через LLM."""
        if self.llm is None:
            return None

        prompt = f"""Напиши Python-функцию для инструмента.

Имя: {spec.name}
Описание: {spec.description}
Input schema: {json.dumps(spec.input_schema, ensure_ascii=False)}
Output schema: {json.dumps(spec.output_schema, ensure_ascii=False)}
Зависимости: {spec.dependencies}
Подсказка: {spec.implementation_hint}

Требования:
1. Функция main(input_data: dict) -> dict
2. Без global state
3. Без print() — только return
4. Обработка ошибок через try/except, возвращай {{"error": "..."}}
5. Только стандартные Python-иды

Верни ТОЛЬКО код, без markdown-обёртки."""

        try:
            response = self.llm.chat(
                messages=[Message(role="user", content=prompt)],
                model=self._agent_model(),
                temperature=0.2,
                max_tokens=2048,
            )
            code = response.content.strip()
            # Убираем markdown обёртку если есть
            if code.startswith("```"):
                lines = code.splitlines()
                code = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
            return code
        except Exception:
            return None

    def _hitl_approval(
        self,
        spec: CustomToolSpec,
        code: str,
        report: SafetyReport,
    ) -> bool:
        """Показывает код и safety-report пользователю, ждёт approval."""
        report_text = self.safety.format_report_for_human(report, code)
        print("\n" + report_text)
        print(f"\nTool: {spec.name}")
        print(f"Description: {spec.description}")
        print(f"Dependencies: {spec.dependencies}")
        print(f"\nCode ({len(code)} chars):\n")
        print(code[:2000])
        if len(code) > 2000:
            print(f"\n... ({len(code) - 2000} more chars)")

        try:
            answer = input("\n\nApprove this tool? [y/N]: ").strip().lower()
            return answer in ("y", "yes", "да")
        except (EOFError, KeyboardInterrupt):
            return False

    # ─── 4. install ─────────────────────────────────────────

    def install(self, spec: InstallSpec) -> InstallResult:
        """
        Устанавливает системный пакет через менеджер.
        pacman (Arch), apt (Debian), dnf (Fedora), pip (Python).
        """
        manager = self.catalog.scanner.detect_package_manager()
        if manager is None:
            return InstallResult(
                success=False,
                manager="unknown",
                package=str(spec.managers),
                error="No package manager detected",
            )

        # Если есть pip — приоритет для Python-пакетов
        if "pip" in spec.managers and spec.managers.get("pip"):
            return self._install_pip(spec.managers["pip"])

        # Системный менеджер
        package = spec.managers.get(manager)
        if not package:
            # Fallback на любой доступный
            for mgr, pkg in spec.managers.items():
                if mgr in ("pacman", "apt", "dnf", "yum", "brew", "pip"):
                    manager = mgr
                    package = pkg
                    break

        if not package:
            return InstallResult(
                success=False,
                manager=manager,
                package="?",
                error=f"Package not specified for {manager}",
            )

        if manager == "pacman":
            return self._install_pacman(package)
        elif manager == "apt":
            return self._install_apt(package)
        elif manager == "dnf":
            return self._install_dnf(package)
        elif manager == "pip":
            return self._install_pip(package)
        else:
            return InstallResult(
                success=False,
                manager=manager,
                package=package,
                error=f"Unsupported manager: {manager}",
            )

    def _install_pacman(self, package: str) -> InstallResult:
        """Arch Linux: pacman -S package."""
        if self.human_approval:
            print(f"\n📦 Install via pacman: {package}")
            try:
                answer = input("Approve? [y/N]: ").strip().lower()
                if answer not in ("y", "yes", "да"):
                    return InstallResult(
                        success=False, manager="pacman", package=package,
                        error="User rejected",
                    )
            except (EOFError, KeyboardInterrupt):
                return InstallResult(
                    success=False, manager="pacman", package=package,
                    error="User interrupted",
                )

        try:
            r = subprocess.run(
                ["sudo", "pacman", "-S", "--noconfirm", package],
                capture_output=True, text=True, timeout=300,
            )
            success = r.returncode == 0
            return InstallResult(
                success=success,
                manager="pacman",
                package=package,
                output=r.stdout,
                error=None if success else r.stderr,
                needs_sudo=True,
            )
        except subprocess.TimeoutExpired:
            return InstallResult(
                success=False, manager="pacman", package=package,
                error="Timeout",
            )

    def _install_apt(self, package: str) -> InstallResult:
        """Debian/Ubuntu: apt install package."""
        if self.human_approval:
            try:
                answer = input(f"\nInstall via apt: {package}? [y/N]: ").strip().lower()
                if answer not in ("y", "yes", "да"):
                    return InstallResult(
                        success=False, manager="apt", package=package,
                        error="User rejected",
                    )
            except (EOFError, KeyboardInterrupt):
                return InstallResult(
                    success=False, manager="apt", package=package,
                    error="User interrupted",
                )

        try:
            r = subprocess.run(
                ["sudo", "apt", "install", "-y", package],
                capture_output=True, text=True, timeout=300,
            )
            return InstallResult(
                success=r.returncode == 0,
                manager="apt",
                package=package,
                output=r.stdout,
                error=None if r.returncode == 0 else r.stderr,
                needs_sudo=True,
            )
        except subprocess.TimeoutExpired:
            return InstallResult(
                success=False, manager="apt", package=package,
                error="Timeout",
            )

    def _install_dnf(self, package: str) -> InstallResult:
        """Fedora/RHEL: dnf install package."""
        try:
            r = subprocess.run(
                ["sudo", "dnf", "install", "-y", package],
                capture_output=True, text=True, timeout=300,
            )
            return InstallResult(
                success=r.returncode == 0,
                manager="dnf",
                package=package,
                output=r.stdout,
                error=None if r.returncode == 0 else r.stderr,
                needs_sudo=True,
            )
        except subprocess.TimeoutExpired:
            return InstallResult(
                success=False, manager="dnf", package=package,
                error="Timeout",
            )

    def _install_pip(self, package: str) -> InstallResult:
        """Python: pip install package."""
        try:
            r = subprocess.run(
                ["pip", "install", package],
                capture_output=True, text=True, timeout=300,
            )
            return InstallResult(
                success=r.returncode == 0,
                manager="pip",
                package=package,
                output=r.stdout,
                error=None if r.returncode == 0 else r.stderr,
                needs_sudo=False,
            )
        except subprocess.TimeoutExpired:
            return InstallResult(
                success=False, manager="pip", package=package,
                error="Timeout",
            )

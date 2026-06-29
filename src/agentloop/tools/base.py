"""
Tool Catalog — типы данных.

ToolDescriptor описывает любой инструмент (Layer 1/2/3).
SearchResult — ответ Steward.search().
CustomToolSpec — запрос на создание custom Python-инструмента.
SafetyReport — результат анализа SafetyAgent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ToolLayer(str, Enum):
    """Слой инструмента в каталоге."""

    CORE = "core"  # Layer 1: 7 примитивов
    DISCOVERED = "discovered"  # Layer 2: системные утилиты
    CUSTOM = "custom"  # Layer 3: созданные Python-скрипты


class ToolCategory(str, Enum):
    """Категория инструмента для поиска."""

    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    MATH = "math"
    AUDIO = "audio"
    VIDEO = "video"
    SYSTEM = "system"
    NETWORK = "network"
    LLM = "llm"
    FILE = "file"
    PYTHON = "python"
    CUSTOM = "custom"
    OTHER = "other"


@dataclass
class ToolDescriptor:
    """
    Описание инструмента в каталоге.

    Для Layer 1 (core) — захардкожено в core.py.
    Для Layer 2 (discovered) — генерируется scanner.py.
    Для Layer 3 (custom) — создаётся steward.py при установке.
    """

    name: str  # уникальное имя, напр. "pdftoppm"
    layer: ToolLayer
    category: ToolCategory
    description: str  # короткое описание для Builder'а

    # Доступность
    available: bool = True  # установлен ли в системе
    install_spec: InstallSpec | None = None  # как установить, если не available

    # Interface (для Builder'а, чтобы понимать как вызывать)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    example_usage: str = ""

    # Для Layer 3 (custom)
    script_path: str | None = None  # путь к .py файлу
    dependencies: list[str] = field(default_factory=list)  # pip-зависимости
    version: str = "1.0"
    safety_report: SafetyReport | None = None

    # Метаданные для поиска
    keywords: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)  # альтернативные имена

    # Для composite tools (например, "pdftoppm + tesseract = OCR")
    composite_of: list[str] = field(default_factory=list)  # имена компонентов

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer.value,
            "category": self.category.value,
            "description": self.description,
            "available": self.available,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "example_usage": self.example_usage,
            "script_path": self.script_path,
            "dependencies": self.dependencies,
            "version": self.version,
            "keywords": self.keywords,
            "aliases": self.aliases,
            "composite_of": self.composite_of,
        }


@dataclass
class InstallSpec:
    """Как установить инструмент."""

    managers: dict[str, str]  # {"pacman": "poppler", "apt": "poppler-utils", "pip": "pylatexenc"}
    notes: str = ""

    def for_manager(self, manager: str) -> str | None:
        return self.managers.get(manager)


@dataclass
class SearchResult:
    """Результат поиска Steward.search()."""

    query: str
    found: list[ToolDescriptor]
    not_found: bool = False
    escalated_to_agent: bool = False
    agent_suggestion: str | None = None  # если steward-agent включился
    custom_tool_possible: bool = False  # если можно создать custom


@dataclass
class CustomToolSpec:
    """
    Спецификация custom Python-инструмента для создания через Steward.
    """

    name: str  # напр. "latex_normalizer"
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    dependencies: list[str] = field(default_factory=list)  # pip-пакеты
    implementation_hint: str = ""  # подсказка для LLM-генератора
    example_usage: str = ""


@dataclass
class CustomToolResult:
    """Результат создания custom-инструмента."""

    tool_id: str  # "custom:latex_normalizer_v1"
    name: str
    version: str
    script_path: str
    safety_report: SafetyReport
    status: str  # "available" | "pending_safety" | "failed"
    error: str | None = None


@dataclass
class SafetyReport:
    """Отчёт SafetyAgent о коде custom-инструмента."""

    verdict: str  # "safe" | "needs_review" | "unsafe"
    concerns: list[str] = field(default_factory=list)
    network_access_used: bool = False
    fs_access: str = "unknown"  # "none" | "read" | "write" | "read_write"
    dangerous_calls: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def is_safe(self) -> bool:
        return self.verdict == "safe"


@dataclass
class InstallResult:
    """Результат установки пакета."""

    success: bool
    manager: str  # "pacman" | "apt" | "pip"
    package: str
    output: str = ""
    error: str | None = None
    needs_sudo: bool = False

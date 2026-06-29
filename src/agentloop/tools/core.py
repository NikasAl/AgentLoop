"""
Layer 1 — Core tools (7 примитивов, захардкожено).

Builder знает только эти инструменты. Всё остальное — производное.
"""

from __future__ import annotations

from .base import (
    InstallSpec,
    ToolCategory,
    ToolDescriptor,
    ToolLayer,
)


def _core(
    name: str,
    description: str,
    category: ToolCategory,
    *,
    input_schema: dict | None = None,
    output_schema: dict | None = None,
    example_usage: str = "",
    keywords: list[str] | None = None,
    aliases: list[str] | None = None,
) -> ToolDescriptor:
    """Хелпер для создания core-инструмента."""
    return ToolDescriptor(
        name=name,
        layer=ToolLayer.CORE,
        category=category,
        description=description,
        available=True,  # core всегда доступен
        input_schema=input_schema or {},
        output_schema=output_schema or {},
        example_usage=example_usage,
        keywords=keywords or [],
        aliases=aliases or [],
    )


CORE_TOOLS: list[ToolDescriptor] = [
    _core(
        name="bash_run",
        description="Выполнить shell-команду в sandbox с timeout. Любая Linux-утилита доступна через этот примитив (pdftotext, ffmpeg, jq, ...).",
        category=ToolCategory.SYSTEM,
        input_schema={
            "command": "str (шаблон с {var})",
            "input_vars": "dict (необязательно)",
            "timeout_sec": "int (default 60)",
        },
        output_schema={
            "stdout": "str",
            "stderr": "str",
            "exit_code": "int",
            "files": "list[str] (если созданы)",
        },
        example_usage='bash_run("pdftoppm -png -r 300 input.pdf output/page")',
        keywords=["shell", "command", "execute", "terminal", "linux"],
        aliases=["sh", "exec", "cmd"],
    ),
    _core(
        name="python_run",
        description="Выполнить Python-скрипт (core или custom через Steward). Скрипт может быть core-модулем (json_merge, select_best_by_score) или custom (custom:latex_validator_v1).",
        category=ToolCategory.PYTHON,
        input_schema={
            "script_ref": "str (напр. 'core:json_merge' или 'custom:latex_validator_v1')",
            "input": "dict (параметры скрипта)",
            "timeout_sec": "int (default 60)",
        },
        output_schema={"result": "dict (по schema скрипта)"},
        example_usage='python_run("custom:latex_validator_v1", {"items": [...]})',
        keywords=["python", "script", "code", "run"],
        aliases=["py", "exec_python"],
    ),
    _core(
        name="llm_call",
        description="Вызов LLM через Provider Layer. Поддерживает 4 провайдера: local (gemma-4-26b), openrouter, zai, human. Vision, JSON mode, tools — через capabilities.",
        category=ToolCategory.LLM,
        input_schema={
            "model": "str (напр. 'local:gemma-4-26b')",
            "messages": "list[Message]",
            "temperature": "float (default 0.7)",
            "max_tokens": "int (default 2048)",
            "json_mode": "bool (default False)",
        },
        output_schema={
            "content": "str",
            "input_tokens": "int",
            "output_tokens": "int",
            "cost_usd": "float",
            "latency_ms": "int",
        },
        example_usage='llm_call("local:gemma-4-26b", [Message(role="user", content="...")])',
        keywords=["llm", "ai", "chat", "completion", "gpt", "gemma", "glm"],
        aliases=["chat", "complete"],
    ),
    _core(
        name="wait_human",
        description="Запросить человеческий ввод. Копирует промпт в буфер обмена, открывает subl, ждёт ответа. Для debug-режима, дистилляции (через браузерную модель) и subjective judge.",
        category=ToolCategory.SYSTEM,
        input_schema={
            "prompt": "str (или messages)",
            "model": "str ('browser' | 'self')",
            "node_id": "str (контекст)",
            "reason": "str (зачем нужен человек)",
            "timeout_sec": "int (default 1800)",
        },
        output_schema={
            "response": "str",
            "human_time_sec": "int",
        },
        example_usage='wait_human(prompt="...", model="browser", reason="distillation teacher")',
        keywords=["human", "manual", "input", "human-in-the-loop", "hitl"],
        aliases=["ask_human", "human_input"],
    ),
    _core(
        name="web_search",
        description="Семантический веб-поиск. Возвращает список результатов (title, url, snippet). Реализация: DuckDuckGo HTML или Searx.",
        category=ToolCategory.NETWORK,
        input_schema={
            "query": "str",
            "max_results": "int (default 10)",
        },
        output_schema={
            "results": "list[{title, url, snippet}]",
        },
        example_usage='web_search("python latex parser library")',
        keywords=["search", "web", "google", "duckduckgo", "internet"],
        aliases=["google", "search_web"],
    ),
    _core(
        name="web_fetch",
        description="Загрузка URL с парсингом. HTML → markdown, PDF → text, JSON → dict. Для research-задач.",
        category=ToolCategory.NETWORK,
        input_schema={
            "url": "str",
            "format": "str ('markdown' | 'text' | 'json' | 'raw')",
            "timeout_sec": "int (default 30)",
        },
        output_schema={
            "content": "str (или dict для json)",
            "status_code": "int",
            "content_type": "str",
        },
        example_usage='web_fetch("https://arxiv.org/abs/2401.00001", format="markdown")',
        keywords=["fetch", "url", "download", "http", "scrape"],
        aliases=["http_get", "download"],
    ),
    _core(
        name="file_op",
        description="Файловые операции как first-class. read/write/append/list/move/copy/delete. Типизированный input/output, не просто bash.",
        category=ToolCategory.FILE,
        input_schema={
            "operation": "str ('read' | 'write' | 'append' | 'list' | 'move' | 'copy' | 'delete' | 'exists')",
            "path": "str",
            "content": "str (для write/append)",
            "pattern": "str (для list, glob)",
        },
        output_schema={
            "result": "str | list[str] | bool (зависит от operation)",
        },
        example_usage='file_op("read", "/path/to/file.txt")',
        keywords=["file", "read", "write", "fs", "filesystem"],
        aliases=["file", "fs"],
    ),
]


def get_core_tool(name: str) -> ToolDescriptor | None:
    """Возвращает core-инструмент по имени."""
    for t in CORE_TOOLS:
        if t.name == name or name in t.aliases:
            return t
    return None


def list_core_tools() -> list[ToolDescriptor]:
    """Возвращает все 7 core-инструментов."""
    return CORE_TOOLS.copy()


def core_tools_for_builder_prompt() -> str:
    """
    Форматирует core-инструменты для системного промпта Builder'а.
    Компактно — только name, description, example_usage.
    """
    lines = ["# Доступные базовые инструменты (Layer 1):"]
    for t in CORE_TOOLS:
        lines.append(f"\n## {t.name}")
        lines.append(f"{t.description}")
        if t.example_usage:
            lines.append(f"Пример: {t.example_usage}")
    lines.append(
        "\n---\n"
        "Если нужного функционала нет среди базовых инструментов — "
        "обратись к Steward через tool_catalog.search() для поиска в Layer 2, "
        "или steward.create_custom() для создания custom Python-инструмента (Layer 3)."
    )
    return "\n".join(lines)

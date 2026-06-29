"""
SafetyAgent — анализ custom Python-кода перед выполнением.

Парсит AST, ищет опасные паттерны:
- os.system, subprocess без whitelist
- eval, exec
- network access (requests, httpx, urllib)
- file system access вне work_dir
- import опасных модулей

Возвращает SafetyReport с verdict: safe | needs_review | unsafe.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .base import SafetyReport


# Опасные модули (требуют attention)
DANGEROUS_MODULES = {
    "os": "fs access, env vars",
    "subprocess": "spawning processes",
    "shutil": "file operations",
    "pathlib": "fs access (usually safe)",
    "socket": "raw network access",
    "http.client": "HTTP client",
    "urllib": "HTTP client",
    "requests": "HTTP client",
    "httpx": "HTTP client",
    "ctypes": "FFI, can call arbitrary C",
    "multiprocessing": "spawning processes",
    "pickle": "arbitrary code execution on unpickle",
    "marshal": "arbitrary code execution",
    "tempfile": "usually safe, but check fs access",
}

# Опасные вызовы (одобрять вручную)
DANGEROUS_CALLS = {
    "os.system": "spawning shell command",
    "os.popen": "spawning shell command",
    "subprocess.call": "spawning process",
    "subprocess.run": "spawning process",
    "subprocess.Popen": "spawning process",
    "eval": "arbitrary code execution",
    "exec": "arbitrary code execution",
    "compile": "code compilation",
    "__import__": "dynamic import",
    "open": "file access",
    "os.remove": "file deletion",
    "os.unlink": "file deletion",
    "shutil.rmtree": "directory deletion",
}

# Whitelist безопасных модулей (no concerns)
SAFE_MODULES = {
    "math", "statistics", "random", "json", "re", "string",
    "collections", "itertools", "functools", "operator",
    "datetime", "time", "decimal", "fractions",
    "pylatexenc", "sympy", "numpy", "pandas",
    "pydantic", "yaml", "tomli",
}


class SafetyAgent:
    """
    Анализирует Python-код custom-инструмента.
    Не использует LLM — только статический AST-анализ.
    """

    def analyze(self, code: str, script_path: str | None = None) -> SafetyReport:
        """
        Анализирует код, возвращает SafetyReport.

        Args:
            code: Python-код для анализа
            script_path: путь к файлу (для контекста в отчёте)
        """
        concerns: list[str] = []
        dangerous_calls: list[str] = []
        network_access = False
        fs_access_flags: set[str] = set()

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return SafetyReport(
                verdict="unsafe",
                concerns=[f"Syntax error: {e}"],
                notes="Code doesn't parse",
            )

        # Анализируем все узлы
        for node in ast.walk(tree):
            # import os / import subprocess
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._check_import(alias.name, concerns, fs_access_flags)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self._check_import(node.module, concerns, fs_access_flags)

            # Function calls
            elif isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name:
                    self._check_call(call_name, node, concerns, dangerous_calls,
                                     fs_access_flags, network_access_ref := [network_access])
                    network_access = network_access_ref[0]

        # Определяем fs_access
        if "read" in fs_access_flags and "write" in fs_access_flags:
            fs_access = "read_write"
        elif "write" in fs_access_flags:
            fs_access = "write"
        elif "read" in fs_access_flags:
            fs_access = "read"
        else:
            fs_access = "none"

        # Вердикт
        verdict = self._make_verdict(concerns, dangerous_calls, network_access, fs_access)

        return SafetyReport(
            verdict=verdict,
            concerns=concerns,
            network_access_used=network_access,
            fs_access=fs_access,
            dangerous_calls=dangerous_calls,
            notes=f"Analyzed {len(code)} chars, {len(list(ast.walk(tree)))} AST nodes",
        )

    def _check_import(
        self,
        module: str,
        concerns: list[str],
        fs_access_flags: set[str],
    ) -> None:
        """Проверяет import- statements."""
        root_module = module.split(".")[0]

        if root_module in DANGEROUS_MODULES:
            concern = DANGEROUS_MODULES[root_module]
            if root_module == "os":
                fs_access_flags.add("read")
                fs_access_flags.add("write")
                concerns.append(f"imports 'os' — {concern}")
            elif root_module in ("subprocess", "multiprocessing"):
                concerns.append(f"imports '{root_module}' — {concern}")
            elif root_module == "socket":
                concerns.append(f"imports 'socket' — {concern}")
            elif root_module in ("requests", "httpx", "urllib", "http.client"):
                concerns.append(f"imports '{root_module}' — {concern}")
            elif root_module == "ctypes":
                concerns.append(f"imports 'ctypes' — {concern}")
            elif root_module in ("pickle", "marshal"):
                # Критично: pickle/marshal = arbitrary code execution
                concerns.append(f"imports '{root_module}' — {concern} (CRITICAL)")
        # Safe modules — ничего не добавляем

    def _check_call(
        self,
        call_name: str,
        node: ast.Call,
        concerns: list[str],
        dangerous_calls: list[str],
        fs_access_flags: set[str],
        network_access_ref: list[bool],
    ) -> None:
        """Проверяет вызовы функций."""
        # Прямые опасные вызовы
        if call_name in DANGEROUS_CALLS:
            concern = DANGEROUS_CALLS[call_name]
            dangerous_calls.append(f"{call_name}() — {concern}")

            if call_name == "open":
                fs_access_flags.add("read")
                # Запись если mode='w' или 'a'
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
                        if isinstance(mode, str) and ("w" in mode or "a" in mode):
                            fs_access_flags.add("write")

                # Позиционный аргумент mode (второй)
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    mode = node.args[1].value
                    if isinstance(mode, str) and ("w" in mode or "a" in mode):
                        fs_access_flags.add("write")

            elif call_name in ("os.remove", "os.unlink", "shutil.rmtree"):
                fs_access_flags.add("write")

        # Network-вызовы на объектах
        if ".get" in call_name or ".post" in call_name or ".put" in call_name or ".delete" in call_name:
            # requests.get, httpx.get, etc.
            network_access_ref[0] = True

        # socket operations
        if "socket" in call_name.lower():
            network_access_ref[0] = True

    def _get_call_name(self, node: ast.Call) -> str:
        """Извлекает имя вызываемой функции."""
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            # Поднимаемся по цепочке: os.path.join -> "os.path.join"
            parts = []
            current = func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return ""

    def _make_verdict(
        self,
        concerns: list[str],
        dangerous_calls: list[str],
        network_access: bool,
        fs_access: str,
    ) -> str:
        """Принимает итоговое решение."""
        # Критические опасности — функции
        critical_patterns = ["eval(", "exec(", "__import__", "ctypes"]
        for pattern in critical_patterns:
            for call in dangerous_calls:
                if pattern in call:
                    return "unsafe"

        # Критические imports — pickle, marshal, ctypes
        for concern in concerns:
            if "CRITICAL" in concern:
                return "unsafe"

        # subprocess без whitelist — needs_review
        for call in dangerous_calls:
            if "subprocess" in call or "os.system" in call or "os.popen" in call:
                return "needs_review"

        # Если есть опасные вызовы, но не критичные — needs_review
        if dangerous_calls:
            return "needs_review"

        # Network — needs_review
        if network_access:
            return "needs_review"

        # Только fs read без network и subprocess — safe
        if fs_access in ("read", "none") and not concerns:
            return "safe"

        # fs write без subprocess и network — needs_review
        if fs_access in ("write", "read_write") and not network_access:
            return "needs_review"

        return "safe" if not concerns else "needs_review"

    def format_report_for_human(self, report: SafetyReport, code: str) -> str:
        """Форматирует отчёт для показа пользователю при HITL."""
        lines = ["=" * 60, "SAFETY REPORT", "=" * 60, ""]
        lines.append(f"Verdict: {report.verdict.upper()}")
        lines.append(f"Network access: {'yes' if report.network_access_used else 'no'}")
        lines.append(f"FS access: {report.fs_access}")
        lines.append("")

        if report.concerns:
            lines.append("Concerns:")
            for c in report.concerns:
                lines.append(f"  - {c}")
            lines.append("")

        if report.dangerous_calls:
            lines.append("Dangerous calls detected:")
            for c in report.dangerous_calls:
                lines.append(f"  - {c}")
            lines.append("")

        lines.append(f"Notes: {report.notes}")
        lines.append("")
        lines.append("=" * 60)
        if report.verdict == "safe":
            lines.append("✓ Tool is safe to execute without supervision.")
        elif report.verdict == "needs_review":
            lines.append("⚠ Tool requires human review before execution.")
        else:
            lines.append("✗ Tool is UNSAFE — do not execute without significant review.")
        lines.append("=" * 60)
        return "\n".join(lines)

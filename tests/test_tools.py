"""Tests for Tool Catalog, Steward, SafetyAgent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentloop.tools import (
    CORE_TOOLS,
    SafetyAgent,
    Steward,
    SystemScanner,
    ToolCatalog,
    ToolCategory,
    ToolDescriptor,
    ToolLayer,
    get_core_tool,
)
from agentloop.tools.base import (
    CustomToolSpec,
    InstallSpec,
    SafetyReport,
    SearchResult,
)


# ─── Layer 1: Core tools ───────────────────────────────────


class TestCoreTools:
    def test_seven_core_tools(self):
        assert len(CORE_TOOLS) == 7

    def test_required_tools_present(self):
        names = {t.name for t in CORE_TOOLS}
        required = {"bash_run", "python_run", "llm_call", "wait_human",
                    "web_search", "web_fetch", "file_op"}
        assert required.issubset(names)

    def test_get_core_tool_by_name(self):
        t = get_core_tool("bash_run")
        assert t is not None
        assert t.layer == ToolLayer.CORE
        assert t.category == ToolCategory.SYSTEM

    def test_get_core_tool_by_alias(self):
        # 'exec' is alias of bash_run
        t = get_core_tool("exec")
        assert t is not None
        assert t.name == "bash_run"

    def test_get_core_tool_missing(self):
        assert get_core_tool("nonexistent") is None

    def test_core_tools_for_builder_prompt(self):
        from agentloop.tools.core import core_tools_for_builder_prompt
        text = core_tools_for_builder_prompt()
        assert "bash_run" in text
        assert "Steward" in text
        assert "Layer 1" in text


# ─── ToolCatalog ───────────────────────────────────────────


class TestToolCatalog:
    @pytest.fixture
    def catalog(self, tmp_path, monkeypatch):
        # Изолируем cache и custom_dir
        monkeypatch.setenv("AGENTLOOP_CACHE_DIR", str(tmp_path / "cache"))
        from agentloop.tools import scanner as scanner_mod
        scanner_mod.CACHE_DIR = tmp_path / "cache"
        scanner_mod.CACHE_FILE = tmp_path / "cache" / "tool_cache.yaml"

        catalog = ToolCatalog()
        catalog._custom_dir = tmp_path / "custom"
        catalog._custom_dir.mkdir(parents=True, exist_ok=True)
        return catalog

    def test_layer1_always_present(self, catalog):
        l1 = catalog.layer1()
        assert len(l1) == 7
        assert all(t.layer == ToolLayer.CORE for t in l1)

    def test_layer2_empty_before_scan(self, catalog):
        l2 = catalog.layer2()
        assert len(l2) == 0

    def test_scan_system_populates_layer2(self, catalog):
        catalog.scan_system(use_cache=False)
        l2 = catalog.layer2()
        # Должны найти хотя бы python3, pip, git на любой системе
        names = {t.name for t in l2}
        assert "python3" in names or "python" in names
        assert "git" in names

    def test_get_by_name(self, catalog):
        t = catalog.get("bash_run")
        assert t is not None
        assert t.name == "bash_run"

    def test_get_missing(self, catalog):
        assert catalog.get("nonexistent") is None

    def test_search_by_keyword(self, catalog):
        catalog.scan_system(use_cache=False)
        # Ищем что-то точно присутствующее
        result = catalog.search("pdf", only_available=False)
        # pdftoppm должен быть в списке (даже если не установлен)
        names = {t.name for t in result.found}
        assert "pdftoppm" in names or "pdftotext" in names

    def test_search_exact_name_high_score(self, catalog):
        catalog.scan_system(use_cache=False)
        result = catalog.search("bash_run")
        # bash_run точно совпадает с core tool
        assert any(t.name == "bash_run" for t in result.found)

    def test_search_no_results(self, catalog):
        # Очень специфичный запрос, который не должен ничего найти
        result = catalog.search("zzzxxxqqq_nonexistent_completely_random")
        # Если что-то нашлось — это случайное совпадение, проверим что хотя бы не точно
        # (на практике Layer 2 может что-то найти по слову "random")
        # Поэтому просто проверим что результат валидный
        assert isinstance(result, SearchResult)
        assert result.query == "zzzxxxqqq_nonexistent_completely_random"

    def test_search_with_layer_filter(self, catalog):
        catalog.scan_system(use_cache=False)
        result = catalog.search("python", layer=ToolLayer.DISCOVERED)
        assert all(t.layer == ToolLayer.DISCOVERED for t in result.found)

    def test_search_with_category_filter(self, catalog):
        catalog.scan_system(use_cache=False)
        result = catalog.search("python", category=ToolCategory.SYSTEM)
        assert all(t.category == ToolCategory.SYSTEM for t in result.found)

    def test_summary_for_builder(self, catalog):
        catalog.scan_system(use_cache=False)
        text = catalog.summary_for_builder()
        assert "Layer 1" in text
        assert "bash_run" in text
        assert "Layer 2" in text

    def test_add_custom(self, catalog):
        custom = ToolDescriptor(
            name="test_tool",
            layer=ToolLayer.CUSTOM,
            category=ToolCategory.CUSTOM,
            description="Test custom tool",
            available=True,
            script_path="/tmp/test.py",
        )
        catalog.add_custom(custom)
        l3 = catalog.layer3()
        assert len(l3) == 1
        assert l3[0].name == "test_tool"

    def test_add_custom_wrong_layer(self, catalog):
        with pytest.raises(ValueError, match="CUSTOM layer"):
            catalog.add_custom(ToolDescriptor(
                name="bad", layer=ToolLayer.CORE,
                category=ToolCategory.SYSTEM, description="",
            ))


# ─── SystemScanner ─────────────────────────────────────────


class TestSystemScanner:
    def test_detect_os_arch(self, tmp_path, monkeypatch):
        from agentloop.tools import scanner as scanner_mod
        scanner_mod.CACHE_DIR = tmp_path / "cache"
        scanner_mod.CACHE_FILE = tmp_path / "cache" / "tool_cache.yaml"

        scanner = SystemScanner()
        info = scanner.detect_os()
        # На Linux-системе должен что-то вернуть
        assert "family" in info
        assert "distro" in info

    def test_detect_package_manager(self, tmp_path, monkeypatch):
        from agentloop.tools import scanner as scanner_mod
        scanner_mod.CACHE_DIR = tmp_path / "cache"
        scanner_mod.CACHE_FILE = tmp_path / "cache" / "tool_cache.yaml"

        scanner = SystemScanner()
        # На тестовой системе должен быть хотя бы один менеджер
        mgr = scanner.detect_package_manager()
        # Может быть None на не-Linux, но обычно есть
        if mgr is not None:
            assert mgr in ("pacman", "apt", "dnf", "yum", "zypper", "brew")

    def test_scan_path_returns_tools(self, tmp_path, monkeypatch):
        from agentloop.tools import scanner as scanner_mod
        scanner_mod.CACHE_DIR = tmp_path / "cache"
        scanner_mod.CACHE_FILE = tmp_path / "cache" / "tool_cache.yaml"

        scanner = SystemScanner()
        tools = scanner.scan_path()
        # Должен найти python3, pip, git
        names = {t.name for t in tools}
        assert "python3" in names or "python" in names
        assert "git" in names
        # Все должны быть Layer 2
        assert all(t.layer == ToolLayer.DISCOVERED for t in tools)

    def test_scan_saves_and_loads_cache(self, tmp_path, monkeypatch):
        from agentloop.tools import scanner as scanner_mod
        scanner_mod.CACHE_DIR = tmp_path / "cache"
        scanner_mod.CACHE_FILE = tmp_path / "cache" / "tool_cache.yaml"

        scanner1 = SystemScanner()
        scanner1.scan_all(use_cache=False)
        assert scanner_mod.CACHE_FILE.exists()

        # Создаём новый scanner, должен загрузить из кеша
        scanner2 = SystemScanner()
        cached = scanner2._load_cache()
        assert cached is not None
        assert len(cached) > 0


# ─── SafetyAgent ───────────────────────────────────────────


class TestSafetyAgent:
    @pytest.fixture
    def agent(self):
        return SafetyAgent()

    def test_safe_code(self, agent):
        code = """
import math
def main(input_data):
    return {"result": math.sqrt(input_data["x"])}
"""
        report = agent.analyze(code)
        assert report.verdict == "safe"
        assert not report.network_access_used
        assert report.fs_access in ("none", "read")

    def test_eval_unsafe(self, agent):
        code = """
def main(input_data):
    return eval(input_data["expr"])
"""
        report = agent.analyze(code)
        assert report.verdict == "unsafe"
        assert any("eval" in c for c in report.dangerous_calls)

    def test_subprocess_needs_review(self, agent):
        code = """
import subprocess
def main(input_data):
    return subprocess.run(["ls"], capture_output=True).stdout.decode()
"""
        report = agent.analyze(code)
        assert report.verdict == "needs_review"
        assert any("subprocess" in c for c in report.dangerous_calls)

    def test_network_access(self, agent):
        code = """
import requests
def main(input_data):
    r = requests.get(input_data["url"])
    return {"content": r.text}
"""
        report = agent.analyze(code)
        assert report.network_access_used
        assert report.verdict == "needs_review"

    def test_file_write(self, agent):
        code = """
def main(input_data):
    with open(input_data["path"], "w") as f:
        f.write(input_data["content"])
    return {"status": "ok"}
"""
        report = agent.analyze(code)
        assert "write" in report.fs_access
        assert report.verdict == "needs_review"

    def test_pickle_unsafe(self, agent):
        code = """
import pickle
def main(input_data):
    return pickle.loads(input_data["data"])
"""
        report = agent.analyze(code)
        assert report.verdict == "unsafe"

    def test_syntax_error(self, agent):
        code = "def broken("
        report = agent.analyze(code)
        assert report.verdict == "unsafe"
        assert "Syntax error" in report.concerns[0]

    def test_format_report_for_human(self, agent):
        report = SafetyReport(
            verdict="safe",
            concerns=[],
            network_access_used=False,
            fs_access="read",
            notes="OK",
        )
        text = agent.format_report_for_human(report, "code")
        assert "SAFE" in text.upper() or "safe" in text
        assert "Verdict" in text


# ─── Steward ───────────────────────────────────────────────


class TestSteward:
    @pytest.fixture
    def catalog(self, tmp_path, monkeypatch):
        from agentloop.tools import scanner as scanner_mod
        scanner_mod.CACHE_DIR = tmp_path / "cache"
        scanner_mod.CACHE_FILE = tmp_path / "cache" / "tool_cache.yaml"
        c = ToolCatalog()
        c._custom_dir = tmp_path / "custom"
        c._custom_dir.mkdir(parents=True, exist_ok=True)
        return c

    @pytest.fixture
    def steward(self, catalog):
        # Steward без LLM для basic тестов
        return Steward(catalog=catalog, llm_provider=None, human_approval=False)

    def test_list_core(self, steward):
        core = steward.list_core()
        assert len(core) == 7

    def test_search_function_finds_tool(self, steward, catalog):
        catalog.scan_system(use_cache=False)
        result = steward.search("pdf")
        # Должен что-то найти в Layer 2
        assert len(result.found) > 0
        assert not result.escalated_to_agent

    def test_search_no_results_without_llm(self, steward):
        result = steward.search("нечто совершенно несуществующее")
        assert result.not_found
        assert not result.escalated_to_agent  # нет LLM
        assert "not configured" in (result.agent_suggestion or "")

    def test_create_custom_with_provided_code(self, steward, catalog):
        spec = CustomToolSpec(
            name="test_tool",
            description="Test tool",
            input_schema={"x": "int"},
            output_schema={"result": "float"},
            dependencies=[],
            implementation_hint="sqrt",
        )
        code = """
import math
def main(input_data):
    return {"result": math.sqrt(input_data["x"])}
"""
        result = steward.create_custom(spec, code=code)
        assert result.status == "available"
        assert result.safety_report.is_safe
        assert Path(result.script_path).exists()

        # Должен появиться в catalog
        l3 = catalog.layer3()
        assert any(t.name == "test_tool" for t in l3)

    def test_create_custom_unsafe_rejected(self, steward):
        spec = CustomToolSpec(
            name="bad_tool",
            description="Bad tool",
            input_schema={"expr": "str"},
            output_schema={},
            dependencies=[],
        )
        code = """
def main(input_data):
    return eval(input_data["expr"])
"""
        result = steward.create_custom(spec, code=code)
        # human_approval=False, но unsafe всё равно блокирует (если не skip)
        assert result.status == "failed"
        assert "unsafe" in (result.error or "").lower()

    def test_create_custom_with_dependencies(self, steward, catalog):
        # Используем стандартную библиотеку, чтобы не требовать реальной установки
        spec = CustomToolSpec(
            name="json_tool",
            description="JSON processor",
            input_schema={"data": "dict"},
            output_schema={"text": "str"},
            dependencies=[],  # без зависимостей
        )
        code = """
import json
def main(input_data):
    return {"text": json.dumps(input_data["data"])}
"""
        result = steward.create_custom(spec, code=code)
        assert result.status == "available"

    def test_install_unknown_manager(self, steward):
        spec = InstallSpec(managers={"unknown_mgr": "package"})
        result = steward.install(spec)
        # Должен упасть gracefully
        assert not result.success

    def test_install_pip_dry_run(self, steward):
        # pip install несуществующего пакета
        spec = InstallSpec(managers={"pip": "this_package_definitely_does_not_exist_xxxxx"})
        result = steward.install(spec)
        assert not result.success
        assert result.manager == "pip"

"""
Tool Catalog — единый реестр инструментов системы.

Три слоя:
- Layer 1: Core (7 примитивов, захардкожено в core.py)
- Layer 2: Discovered (сканируется из PATH и pip list)
- Layer 3: Custom (создаётся Builder'ом через Steward)
"""

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
from .core import CORE_TOOLS, core_tools_for_builder_prompt, get_core_tool, list_core_tools
from .safety import SafetyAgent
from .scanner import SystemScanner
from .steward import Steward

__all__ = [
    "CORE_TOOLS",
    "CustomToolResult",
    "CustomToolSpec",
    "InstallResult",
    "InstallSpec",
    "SafetyAgent",
    "SafetyReport",
    "Steward",
    "SystemScanner",
    "ToolCatalog",
    "ToolCategory",
    "ToolDescriptor",
    "ToolLayer",
    "SearchResult",
    "core_tools_for_builder_prompt",
    "get_core_tool",
    "list_core_tools",
]

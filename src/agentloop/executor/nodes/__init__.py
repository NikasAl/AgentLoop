"""
Node types — реализации всех типов узлов DAG.

Каждый узел реализует интерфейс BaseNode.execute(state) -> NodeResult.
"""

from .base import BaseNode, NodeResult, NodeError, NodeFactory
from .bash import BashNode
from .llm import LLMNode
from .python import PythonNode
from .file import FileNode
from .gate import GateNode
from .loop import LoopNode

__all__ = [
    "BaseNode",
    "BashNode",
    "FileNode",
    "GateNode",
    "LLMNode",
    "LoopNode",
    "NodeError",
    "NodeFactory",
    "NodeResult",
    "PythonNode",
]

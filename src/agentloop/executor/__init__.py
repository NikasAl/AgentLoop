"""
Pipeline Executor — интерпретирует DAG из design/2_dag.json и выполняет узлы.

Главные компоненты:
- PipelineState: состояние между узлами (переменные, файлы)
- PipelineExecutor: топологическая сортировка + последовательное выполнение
- Node types: bash, llm, python, file, loop, gate
- Checkpoint: сохранение прогресса для resume
"""

from .state import PipelineState, VariableResolver
from .executor import PipelineExecutor, ExecutionResult
from .checkpoint import Checkpoint

__all__ = [
    "Checkpoint",
    "ExecutionResult",
    "PipelineExecutor",
    "PipelineState",
    "VariableResolver",
]

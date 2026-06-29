"""
BaseNode — базовый класс для всех типов узлов DAG.

Каждый узел:
- Имеет id, type, timeout_sec, max_retries
- Может иметь condition (опциональное выполнение)
- Реализует execute(state) → NodeResult
- Поддерживает retry при неудаче
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..state import PipelineState, VariableResolver


class NodeError(Exception):
    """Ошибка выполнения узла."""

    def __init__(self, node_id: str, message: str, retryable: bool = False):
        super().__init__(f"[{node_id}] {message}")
        self.node_id = node_id
        self.retryable = retryable


@dataclass
class NodeResult:
    """Результат выполнения узла."""

    node_id: str
    success: bool
    output: dict[str, Any] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    error: str | None = None
    duration_sec: float = 0.0
    skipped: bool = False  # если condition = false
    retries_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseNode:
    """
    Базовый класс для всех типов узлов.

    Подклассы должны реализовать _execute(state, resolver) -> NodeResult.
    """

    def __init__(
        self,
        node_id: str,
        timeout_sec: int = 60,
        max_retries: int = 0,
        condition: str | None = None,
        fallback_on_failure: str = "error",  # "error" | "skip_and_log" | "use_default"
        default_output: dict[str, Any] | None = None,
    ):
        self.node_id = node_id
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.condition = condition
        self.fallback_on_failure = fallback_on_failure
        self.default_output = default_output or {}

    def execute(self, state: PipelineState) -> NodeResult:
        """
        Главный метод выполнения. Управляет condition, retry, fallback.
        """
        resolver = state.resolver()

        # Проверяем condition
        if self.condition:
            try:
                should_run = resolver.resolve_condition(self.condition)
            except Exception as e:
                return NodeResult(
                    node_id=self.node_id,
                    success=False,
                    error=f"Condition eval failed: {e}",
                )
            if not should_run:
                return NodeResult(
                    node_id=self.node_id,
                    success=True,
                    skipped=True,
                    output={"skipped": True, "reason": "condition_false"},
                )

        # Выполняем с retry
        last_error: str | None = None
        last_result: NodeResult | None = None
        for attempt in range(self.max_retries + 1):
            start = time.time()
            try:
                result = self._execute(state, resolver)
                result.duration_sec = time.time() - start
                result.retries_used = attempt
                last_result = result
                if result.success:
                    # Сохраняем output в state
                    state.set_output(self.node_id, result.output)
                    if result.files:
                        state.add_files(self.node_id, result.files)
                    return result
                else:
                    last_error = result.error or "Unknown error"
            except NodeError as e:
                last_error = str(e)
                if not e.retryable:
                    break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"

            if attempt < self.max_retries:
                time.sleep(1 * (attempt + 1))  # exponential-ish backoff

        # Все попытки провалились
        if self.fallback_on_failure == "skip_and_log":
            state.set_error(self.node_id, last_error or "failed")
            state.set_output(self.node_id, self.default_output)
            return NodeResult(
                node_id=self.node_id,
                success=True,  # soft success
                skipped=True,
                error=last_error,
                output=self.default_output,
                retries_used=self.max_retries,
                metadata={"fallback_used": True},
            )
        elif self.fallback_on_failure == "use_default":
            state.set_output(self.node_id, self.default_output)
            return NodeResult(
                node_id=self.node_id,
                success=True,
                output=self.default_output,
                error=last_error,
                retries_used=self.max_retries,
                metadata={"fallback_used": True, "default_used": True},
            )
        else:  # "error"
            state.set_error(self.node_id, last_error or "failed")
            return NodeResult(
                node_id=self.node_id,
                success=False,
                error=last_error,
                retries_used=self.max_retries,
                output=last_result.output if last_result else {},
            )

    def _execute(self, state: PipelineState, resolver: VariableResolver) -> NodeResult:
        """Переопределяется в подклассах. Реальная логика узла."""
        raise NotImplementedError


class NodeFactory:
    """Фабрика для создания узлов из JSON-описания DAG."""

    @staticmethod
    def from_dict(node_dict: dict[str, Any]) -> BaseNode:
        """Создаёт узел нужного типа из dict."""
        node_type = node_dict.get("type")
        if not node_type:
            raise ValueError(f"Node missing 'type' field: {node_dict}")

        # Импортируем здесь чтобы избежать циклических зависимостей
        from .bash import BashNode
        from .file import FileNode
        from .gate import GateNode
        from .llm import LLMNode
        from .loop import LoopNode
        from .python import PythonNode

        builders: dict[str, Callable[[dict], BaseNode]] = {
            "bash": BashNode.from_dict,
            "llm": LLMNode.from_dict,
            "python": PythonNode.from_dict,
            "file": FileNode.from_dict,
            "gate": GateNode.from_dict,
            "loop": LoopNode.from_dict,
        }

        builder = builders.get(node_type)
        if not builder:
            raise ValueError(f"Unknown node type: {node_type}")

        return builder(node_dict)

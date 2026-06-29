"""
LoopNode — цикл над sub-graph.

Поля:
- body: list of node dicts (sub-graph)
- exit_condition: условие выхода
- max_iterations: лимит
- on_max_iterations: "continue_with_warning" | "abort"
- carry_over_state: файлы, которые обновляются между итерациями
"""

from __future__ import annotations

from typing import Any

from ..state import PipelineState, VariableResolver
from .base import BaseNode, NodeResult, NodeFactory


class LoopNode(BaseNode):
    """Узел цикла над sub-graph."""

    def __init__(
        self,
        node_id: str,
        body: list[dict[str, Any]],
        exit_condition: str,
        max_iterations: int = 3,
        on_max_iterations: str = "continue_with_warning",
        carry_over_state: list[str] | None = None,
        loop_kind: str = "until_condition",
        timeout_sec: int = 3600,
        condition: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(node_id, timeout_sec, 0, condition, **kwargs)
        self.body = body
        self.exit_condition = exit_condition
        self.max_iterations = max_iterations
        self.on_max_iterations = on_max_iterations
        self.carry_over_state = carry_over_state or []
        self.loop_kind = loop_kind

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LoopNode":
        return cls(
            node_id=d["id"],
            body=d.get("body", []),
            exit_condition=d.get("exit_condition", "true"),
            max_iterations=d.get("max_iterations", 3),
            on_max_iterations=d.get("on_max_iterations", "continue_with_warning"),
            carry_over_state=d.get("carry_over_state"),
            loop_kind=d.get("loop_kind", "until_condition"),
            timeout_sec=d.get("timeout_sec", 3600),
            condition=d.get("condition"),
            fallback_on_failure=d.get("fallback_on_failure", "error"),
        )

    def _execute(self, state: PipelineState, resolver: VariableResolver) -> NodeResult:
        """Выполняет sub-graph в цикле."""
        # Создаём под-узлы из body
        sub_nodes = []
        for node_dict in self.body:
            try:
                node = NodeFactory.from_dict(node_dict)
                sub_nodes.append(node)
            except Exception as e:
                return NodeResult(
                    node_id=self.node_id,
                    success=False,
                    error=f"Failed to create sub-node {node_dict.get('id')}: {e}",
                )

        iterations_run = 0
        last_outputs: list[dict[str, Any]] = []
        score_history: list[float] = []

        for iteration in range(self.max_iterations):
            iterations_run = iteration + 1
            # Устанавливаем iteration_count в state (для использования в exit_condition)
            state.runtime_vars[f"{self.node_id}_iteration_count"] = iteration + 1

            # Выполняем все под-узлы последовательно
            iteration_outputs: dict[str, Any] = {}
            for node in sub_nodes:
                result = node.execute(state)
                if not result.success:
                    return NodeResult(
                        node_id=self.node_id,
                        success=False,
                        error=f"Sub-node {node.node_id} failed: {result.error}",
                        output={"iterations_run": iterations_run, "last_outputs": last_outputs},
                        metadata={"iterations": iterations_run},
                    )
                iteration_outputs[node.node_id] = result.output

            last_outputs.append(iteration_outputs)

            # Проверяем exit_condition
            try:
                # exit_condition может ссылаться на {node_id.output.field}
                # Используем текущий resolver (он видит обновлённые outputs)
                should_exit = resolver.resolve_condition(self.exit_condition)
            except Exception:
                should_exit = False

            # Сохраняем score если есть
            for output in iteration_outputs.values():
                if isinstance(output, dict) and "composite_score" in output:
                    score_history.append(float(output["composite_score"]))
                    break

            if should_exit:
                return NodeResult(
                    node_id=self.node_id,
                    success=True,
                    output={
                        "iterations_run": iterations_run,
                        "exit_reason": "condition_met",
                        "last_outputs": iteration_outputs,
                        "score_history": score_history,
                    },
                    metadata={"iterations": iterations_run, "exit_reason": "condition_met"},
                )

        # Достигли max_iterations
        if self.on_max_iterations == "continue_with_warning":
            return NodeResult(
                node_id=self.node_id,
                success=True,
                output={
                    "iterations_run": iterations_run,
                    "exit_reason": "max_iterations_reached",
                    "last_outputs": iteration_outputs,
                    "score_history": score_history,
                },
                metadata={
                    "iterations": iterations_run,
                    "exit_reason": "max_iterations_reached",
                    "warning": "Loop reached max iterations without meeting exit condition",
                },
            )
        else:  # abort
            return NodeResult(
                node_id=self.node_id,
                success=False,
                error=f"Loop reached max_iterations ({self.max_iterations}) without exit condition",
                output={"iterations_run": iterations_run, "score_history": score_history},
                metadata={"iterations": iterations_run, "exit_reason": "max_iterations_abort"},
            )

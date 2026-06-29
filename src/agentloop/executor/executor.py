"""
PipelineExecutor — главный интерпретатор DAG.

Загружает DAG из JSON, выполняет узлы в топологическом порядке,
обрабатывает condition, iterate_over, loop, gate.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..cost_tracker import CostTracker
from .checkpoint import Checkpoint
from .nodes import NodeFactory, NodeResult
from .nodes.base import BaseNode
from .state import PipelineState


@dataclass
class ExecutionResult:
    """Итог выполнения pipeline."""

    run_id: str
    success: bool
    completed_nodes: list[str] = field(default_factory=list)
    failed_nodes: list[str] = field(default_factory=list)
    skipped_nodes: list[str] = field(default_factory=list)
    final_output: dict[str, Any] = field(default_factory=dict)
    total_duration_sec: float = 0.0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    node_results: dict[str, NodeResult] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    exit_node_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "success": self.success,
            "completed_nodes": self.completed_nodes,
            "failed_nodes": self.failed_nodes,
            "skipped_nodes": self.skipped_nodes,
            "final_output": self.final_output,
            "total_duration_sec": self.total_duration_sec,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "errors": self.errors,
            "exit_node_id": self.exit_node_id,
        }


class PipelineExecutor:
    """
    Интерпретатор DAG.

    Usage:
        executor = PipelineExecutor(work_dir="/tmp/run1")
        result = executor.execute(
            dag_path="path/to/dag.json",
            task_id="my_task",
            hypothesis_id="h1",
            input_vars={"$INPUT": "/data/file.pdf"},
        )
    """

    def __init__(
        self,
        work_dir: Path | str,
        cost_tracker: CostTracker | None = None,
        checkpoint_enabled: bool = True,
    ):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cost_tracker = cost_tracker
        self.checkpoint_enabled = checkpoint_enabled
        self.checkpoint = Checkpoint(self.work_dir)

    def execute(
        self,
        dag_path: Path | str | None = None,
        dag_dict: dict[str, Any] | None = None,
        task_id: str = "unknown",
        hypothesis_id: str = "unknown",
        mode: str = "research",
        input_vars: dict[str, Any] | None = None,
        resume: bool = False,
    ) -> ExecutionResult:
        """
        Выполняет DAG.

        Args:
            dag_path: путь к JSON-файлу DAG
            dag_dict: DAG как dict (альтернатива dag_path)
            task_id: ID задачи
            hypothesis_id: ID гипотезы
            mode: "research" | "production"
            input_vars: runtime variables ($INPUT, $PAGE_NUM, ...)
            resume: продолжить с последнего checkpoint

        Returns:
            ExecutionResult
        """
        # Загружаем DAG
        if dag_dict is None:
            if dag_path is None:
                return ExecutionResult(
                    run_id="error",
                    success=False,
                    errors={"_": "Either dag_path or dag_dict required"},
                )
            dag_dict = json.loads(Path(dag_path).read_text(encoding="utf-8"))

        # Создаём state
        run_id = f"run_{int(time.time())}"
        state = PipelineState(
            work_dir=self.work_dir,
            runtime_vars=input_vars,
            run_id=run_id,
            task_id=task_id,
            hypothesis_id=hypothesis_id,
            mode=mode,
        )

        # Resume из checkpoint
        last_completed = None
        if resume and self.checkpoint_enabled:
            last_completed = self.checkpoint.restore_state(state)
            if last_completed:
                print(f"📦 Resumed from checkpoint. Last completed: {last_completed}")

        # Топологическая сортировка узлов
        try:
            ordered_nodes = self._topological_sort(dag_dict)
        except Exception as e:
            return ExecutionResult(
                run_id=run_id,
                success=False,
                errors={"_": f"Topological sort failed: {e}"},
            )

        # Создаём инстансы узлов
        node_instances: dict[str, BaseNode] = {}
        for node_dict in dag_dict.get("nodes", []):
            try:
                node = NodeFactory.from_dict(node_dict)
                node_instances[node.node_id] = node
            except Exception as e:
                return ExecutionResult(
                    run_id=run_id,
                    success=False,
                    errors={node_dict.get("id", "?"): f"Node creation failed: {e}"},
                )

        # Загружаем prompts для LLM-узлов
        prompts = dag_dict.get("prompts", {})
        for node in node_instances.values():
            if hasattr(node, "system_prompt") and node.system_prompt:
                # Если это reference на prompts
                if node.system_prompt in prompts:
                    node.system_prompt = prompts[node.system_prompt].get("text", node.system_prompt)
                elif isinstance(node.system_prompt, str) and not node.system_prompt.startswith("Ты") and not node.system_prompt.startswith("You"):
                    # Может быть prompt_ref
                    ref = prompts.get(node.system_prompt)
                    if ref:
                        node.system_prompt = ref.get("text", node.system_prompt)

        # Выполняем узлы
        result = ExecutionResult(run_id=run_id, success=True)
        start_time = time.time()
        skip_until_resumed = last_completed is not None

        for node_id in ordered_nodes:
            # Если resume — пропускаем уже выполненные
            if skip_until_resumed:
                if node_id == last_completed:
                    skip_until_resumed = False
                    result.completed_nodes.append(node_id)
                    continue
                else:
                    result.completed_nodes.append(node_id)
                    continue

            node = node_instances.get(node_id)
            if node is None:
                result.failed_nodes.append(node_id)
                result.errors[node_id] = "Node not found in instances"
                continue

            print(f"\n▶ Executing: {node_id} ({type(node).__name__})")
            node_result = node.execute(state)
            result.node_results[node_id] = node_result

            # Собираем метрики
            if node_result.metadata:
                result.total_cost_usd += node_result.metadata.get("cost_usd", 0.0) or node_result.metadata.get("total_cost_usd", 0.0)
                result.total_tokens += (
                    node_result.metadata.get("input_tokens", 0)
                    + node_result.metadata.get("output_tokens", 0)
                )

            if node_result.success:
                if node_result.skipped:
                    result.skipped_nodes.append(node_id)
                else:
                    result.completed_nodes.append(node_id)

                # Checkpoint после каждого успешного узла
                if self.checkpoint_enabled:
                    self.checkpoint.save(state, last_completed_node=node_id)

                # Логируем в cost_tracker
                if self.cost_tracker and node_result.metadata:
                    self._log_to_tracker(
                        task_id=task_id,
                        run_id=run_id,
                        node_id=node_id,
                        metadata=node_result.metadata,
                    )
            else:
                result.failed_nodes.append(node_id)
                result.errors[node_id] = node_result.error or "Unknown error"
                result.success = False

                # Проверяем, должен ли gate вернуть назад (go_back_to)
                if isinstance(node, type) and hasattr(node_result, "metadata"):
                    go_back = node_result.metadata.get("go_back_to")
                    if go_back and go_back in node_instances:
                        # Сбрасываем output узла, к которому возвращаемся
                        if go_back in state.outputs:
                            del state.outputs[go_back]
                        # Продолжаем с этого узла (упрощённо — просто продолжаем)
                        # В полной реализации здесь нужен rollback состояния
                        print(f"⤴ Gate rejected, would go back to: {go_back}")
                        break
                break

        result.total_duration_sec = time.time() - start_time

        # Финальный output = output exit-узла
        exit_node_id = dag_dict.get("exit")
        if exit_node_id and exit_node_id in state.outputs:
            result.final_output = state.outputs[exit_node_id]
            result.exit_node_id = exit_node_id

        # Очищаем checkpoint при успешном завершении
        if result.success and self.checkpoint_enabled:
            self.checkpoint.clear()

        return result

    def _topological_sort(self, dag: dict[str, Any]) -> list[str]:
        """
        Топологическая сортировка узлов по edges.
        Использует Kahn's algorithm.
        """
        nodes = [n["id"] for n in dag.get("nodes", [])]
        edges = dag.get("edges", [])

        # Строим adjacency list и in-degree
        adj: dict[str, list[str]] = {n: [] for n in nodes}
        in_degree: dict[str, int] = {n: 0 for n in nodes}

        for edge in edges:
            src = edge.get("from")
            dst = edge.get("to")
            if src in adj and dst in adj:
                adj[src].append(dst)
                in_degree[dst] += 1

        # Kahn's algorithm
        queue = [n for n in nodes if in_degree[n] == 0]
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Если есть циклы — result будет меньше nodes
        if len(result) != len(nodes):
            # Добавляем оставшиеся (например, для loop/gate без явных edges)
            remaining = [n for n in nodes if n not in result]
            result.extend(remaining)

        return result

    def _log_to_tracker(
        self,
        task_id: str,
        run_id: str,
        node_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Логирует вызов в cost_tracker."""
        if not self.cost_tracker:
            return

        provider = metadata.get("provider", "unknown")
        model = metadata.get("model", "unknown")
        input_tokens = metadata.get("input_tokens", 0)
        output_tokens = metadata.get("output_tokens", 0)
        cost = metadata.get("cost_usd", 0.0)
        latency_ms = metadata.get("latency_ms", 0)
        human_time = metadata.get("human_time_sec", 0)

        self.cost_tracker.log(
            task_id=task_id,
            run_id=run_id,
            node_id=node_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            human_time_sec=human_time,
        )

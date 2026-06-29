"""
Checkpoint — сохранение и загрузка прогресса pipeline.

Сохраняет snapshot PipelineState в {work_dir}/checkpoint.json.
При resume — загружает snapshot и пропускает уже выполненные узлы.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state import PipelineState


class Checkpoint:
    """Управление чекпойнтами pipeline."""

    def __init__(self, work_dir: Path | str):
        self.work_dir = Path(work_dir)
        self.checkpoint_file = self.work_dir / "checkpoint.json"

    def save(self, state: PipelineState, last_completed_node: str | None = None) -> None:
        """
        Сохраняет snapshot состояния.

        Args:
            state: текущее PipelineState
            last_completed_node: ID последнего успешно выполненного узла
        """
        snapshot = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "run_id": state.run_id,
            "task_id": state.task_id,
            "hypothesis_id": state.hypothesis_id,
            "mode": state.mode,
            "runtime_vars": state.runtime_vars,
            "outputs": state.outputs,
            "errors": state.errors,
            "gate_decisions": state.gate_decisions,
            "last_completed_node": last_completed_node,
            "completed_nodes": list(state.outputs.keys()),
        }

        self.checkpoint_file.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def load(self) -> dict[str, Any] | None:
        """Загружает snapshot из чекпойнта. None если нет."""
        if not self.checkpoint_file.exists():
            return None
        try:
            return json.loads(self.checkpoint_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def restore_state(self, state: PipelineState) -> str | None:
        """
        Восстанавливает состояние из чекпойнта.

        Returns:
            ID последнего выполненного узла (для resume), или None
        """
        snapshot = self.load()
        if not snapshot:
            return None

        state.outputs = snapshot.get("outputs", {})
        state.errors = snapshot.get("errors", {})
        state.gate_decisions = snapshot.get("gate_decisions", {})
        state.runtime_vars.update(snapshot.get("runtime_vars", {}))

        return snapshot.get("last_completed_node")

    def clear(self) -> None:
        """Удаляет чекпойнт (после успешного завершения)."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

    def exists(self) -> bool:
        return self.checkpoint_file.exists()

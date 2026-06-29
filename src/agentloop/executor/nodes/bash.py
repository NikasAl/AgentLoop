"""
BashNode — выполнение shell-команды в sandbox с timeout.

Использует subprocess.run с timeout. На первом этапе без bwrap
(добавим sandbox в PipelineExecutor, когда он будет готов).

Output: {stdout, stderr, exit_code, files?}
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

from .base import BaseNode, NodeResult
from ..state import PipelineState, VariableResolver


class BashNode(BaseNode):
    """Узел выполнения shell-команды."""

    def __init__(
        self,
        node_id: str,
        command: str,
        input_vars: dict[str, Any] | None = None,
        output_pattern: str | None = None,
        output_list_as: str | None = None,
        output_kind: str = "text",
        working_dir: str | None = None,
        timeout_sec: int = 60,
        max_retries: int = 0,
        condition: str | None = None,
        env_vars: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(node_id, timeout_sec, max_retries, condition, **kwargs)
        self.command = command
        self.input_vars = input_vars or {}
        self.output_pattern = output_pattern
        self.output_list_as = output_list_as
        self.output_kind = output_kind
        self.working_dir = working_dir
        self.env_vars = env_vars or {}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BashNode":
        return cls(
            node_id=d["id"],
            command=d["command"],
            input_vars=d.get("input_vars"),
            output_pattern=d.get("output", {}).get("pattern") if isinstance(d.get("output"), dict) else d.get("output_pattern"),
            output_list_as=d.get("output", {}).get("list_as") if isinstance(d.get("output"), dict) else d.get("output_list_as"),
            output_kind=d.get("output", {}).get("kind", "text") if isinstance(d.get("output"), dict) else d.get("output_kind", "text"),
            working_dir=d.get("working_dir"),
            timeout_sec=d.get("timeout_sec", 60),
            max_retries=d.get("max_retries", 0),
            condition=d.get("condition"),
            env_vars=d.get("env_vars"),
            fallback_on_failure=d.get("fallback_on_failure", "error"),
            default_output=d.get("default_output"),
        )

    def _execute(self, state: PipelineState, resolver: VariableResolver) -> NodeResult:
        # Разрешаем переменные в command
        cmd_resolved = resolver.resolve(self.command)
        work_dir = resolver.resolve(self.working_dir) if self.working_dir else str(state.work_dir)

        # Подготавливаем env
        env = {"PATH": "/usr/bin:/bin:/usr/local/bin"}
        env.update({k: str(v) for k, v in self.env_vars.items()})

        try:
            r = subprocess.run(
                cmd_resolved,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                cwd=work_dir,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return NodeResult(
                node_id=self.node_id,
                success=False,
                error=f"Timeout after {self.timeout_sec}s",
                output={"stdout": "", "stderr": f"Timeout after {self.timeout_sec}s", "exit_code": -1, "files": []},
            )
        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                success=False,
                error=f"Execution error: {e}",
                output={"stdout": "", "stderr": str(e), "exit_code": -1, "files": []},
            )

        output: dict[str, Any] = {
            "stdout": r.stdout,
            "stderr": r.stderr,
            "exit_code": r.returncode,
        }

        # Если задан pattern для files — собираем файлы
        files: list[str] = []
        if self.output_pattern:
            pattern_resolved = resolver.resolve(self.output_pattern)
            # Поддерживаем glob
            import glob
            matches = sorted(glob.glob(pattern_resolved))
            files = matches
            if self.output_list_as:
                output[self.output_list_as] = matches

        output["files"] = files

        success = r.returncode == 0
        return NodeResult(
            node_id=self.node_id,
            success=success,
            output=output,
            files=files,
            error=None if success else f"Exit code {r.returncode}: {r.stderr[:500]}",
        )

"""
PythonNode — выполнение Python-скриптов (core или custom).

script_ref может быть:
- "core:json_merge" — встроенный core-модуль
- "custom:latex_validator_v1" — custom-инструмент из Layer 3
- "/abs/path/to/script.py" — прямой путь к файлу

Скрипт должен иметь функцию main(input_data: dict) -> dict.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from ..state import PipelineState, VariableResolver
from .base import BaseNode, NodeResult


# Core Python-модули (встроенные)
CORE_SCRIPTS_DIR = Path(__file__).parent / "core_scripts"


class PythonNode(BaseNode):
    """Узел выполнения Python-скрипта."""

    def __init__(
        self,
        node_id: str,
        script_ref: str,
        input_data: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        iterate_over: str | None = None,
        timeout_sec: int = 60,
        max_retries: int = 0,
        condition: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(node_id, timeout_sec, max_retries, condition, **kwargs)
        self.script_ref = script_ref
        self.input_data = input_data or {}
        self.output_schema = output_schema
        self.iterate_over = iterate_over

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PythonNode":
        return cls(
            node_id=d["id"],
            script_ref=d["script_ref"],
            input_data=d.get("input"),
            output_schema=d.get("output_schema") or d.get("output", {}).get("schema") if isinstance(d.get("output"), dict) else d.get("output_schema"),
            iterate_over=d.get("iterate_over"),
            timeout_sec=d.get("timeout_sec", 60),
            max_retries=d.get("max_retries", 0),
            condition=d.get("condition"),
            fallback_on_failure=d.get("fallback_on_failure", "error"),
            default_output=d.get("default_output"),
        )

    def _execute(self, state: PipelineState, resolver: VariableResolver) -> NodeResult:
        # Разрешаем переменные в input_data
        resolved_input = resolver.resolve(self.input_data)

        # Если iterate_over — цикл
        if self.iterate_over:
            return self._execute_iterated(state, resolver, resolved_input)

        return self._run_script(resolved_input)

    def _execute_iterated(
        self,
        state: PipelineState,
        resolver: VariableResolver,
        base_input: dict[str, Any],
    ) -> NodeResult:
        """Запускает скрипт для каждого элемента из iterate_over."""
        items_ref = resolver.resolve(self.iterate_over)
        if isinstance(items_ref, str):
            # Это glob pattern
            import glob
            items = sorted(glob.glob(items_ref))
        elif isinstance(items_ref, list):
            items = items_ref
        else:
            return NodeResult(
                node_id=self.node_id,
                success=False,
                error=f"iterate_over resolved to {type(items_ref)}, expected list or glob",
            )

        results = []
        for item in items:
            # Подставляем item в input_data
            iter_input = dict(base_input)
            iter_input["_current_item"] = item
            iter_input["_items"] = items

            # Если в input_data было поле с ссылками на iterate_over, подставляем item
            for key, val in list(iter_input.items()):
                if val == self.iterate_over:
                    iter_input[key] = item

            result = self._run_script(iter_input)
            if not result.success:
                return result
            results.append(result.output)

        return NodeResult(
            node_id=self.node_id,
            success=True,
            output={"results": results, "count": len(results)},
            metadata={"iterations": len(items)},
        )

    def _run_script(self, input_data: dict[str, Any]) -> NodeResult:
        """Загружает и запускает скрипт."""
        script_path = self._resolve_script_path()
        if not script_path:
            return NodeResult(
                node_id=self.node_id,
                success=False,
                error=f"Cannot find script: {self.script_ref}",
            )

        try:
            # Загружаем модуль динамически
            module_name = f"_agentloop_script_{self.node_id}_{hash(str(script_path)) & 0xFFFFFF:x}"
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            if spec is None or spec.loader is None:
                return NodeResult(
                    node_id=self.node_id,
                    success=False,
                    error=f"Cannot load module from {script_path}",
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Вызываем main(input_data)
            if not hasattr(module, "main"):
                return NodeResult(
                    node_id=self.node_id,
                    success=False,
                    error=f"Script {script_path} has no main() function",
                )

            result = module.main(input_data)
            if not isinstance(result, dict):
                return NodeResult(
                    node_id=self.node_id,
                    success=False,
                    error=f"Script returned {type(result)}, expected dict",
                )

            # Если в результате есть "error" — считаем неудачей
            if "error" in result:
                return NodeResult(
                    node_id=self.node_id,
                    success=False,
                    output=result,
                    error=str(result["error"]),
                )

            return NodeResult(
                node_id=self.node_id,
                success=True,
                output=result,
            )

        except Exception as e:
            return NodeResult(
                node_id=self.node_id,
                success=False,
                error=f"Script execution failed: {type(e).__name__}: {e}",
            )

    def _resolve_script_path(self) -> Path | None:
        """Разрешает script_ref в путь к файлу."""
        # "core:json_merge" → core_scripts/json_merge.py
        if self.script_ref.startswith("core:"):
            name = self.script_ref[5:]
            path = CORE_SCRIPTS_DIR / f"{name}.py"
            return path if path.exists() else None

        # "custom:latex_validator_v1" → ~/.agentloop/custom_tools/latex_validator_v1.py
        elif self.script_ref.startswith("custom:"):
            name = self.script_ref[7:]
            custom_dir = Path.home() / ".agentloop" / "custom_tools"
            path = custom_dir / f"{name}.py"
            return path if path.exists() else None

        # Прямой путь
        else:
            path = Path(self.script_ref)
            return path if path.exists() else None

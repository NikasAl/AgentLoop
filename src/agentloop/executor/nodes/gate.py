"""
GateNode — промежуточная точка контроля.

Поддерживает режимы: human_approval, quality_check, budget_check
Approval modes: approve, reject, modify, accept_as_is
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..state import PipelineState, VariableResolver
from .base import BaseNode, NodeResult


class GateNode(BaseNode):
    """Узел-шлюз с human approval."""

    def __init__(
        self,
        node_id: str,
        gate_kind: str = "human_approval",  # human_approval | quality_check | budget_check
        prompt_template: str = "",
        show_artifacts: list[str] | None = None,
        approval_modes: list[str] | None = None,
        on_reject: dict[str, Any] | None = None,
        on_modify: dict[str, Any] | None = None,
        on_accept_as_is: dict[str, Any] | None = None,
        hitl_mode: str = "manual",
        timeout_sec: int = 1800,
        condition: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(node_id, timeout_sec, 0, condition, **kwargs)
        self.gate_kind = gate_kind
        self.prompt_template = prompt_template
        self.show_artifacts = show_artifacts or []
        self.approval_modes = approval_modes or ["approve", "reject"]
        self.on_reject = on_reject or {"action": "abort"}
        self.on_modify = on_modify or {"action": "use_modified_input"}
        self.on_accept_as_is = on_accept_as_is or {"action": "continue"}
        self.hitl_mode = hitl_mode

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GateNode":
        return cls(
            node_id=d["id"],
            gate_kind=d.get("gate_kind", "human_approval"),
            prompt_template=d.get("prompt_template", ""),
            show_artifacts=d.get("show_artifacts"),
            approval_modes=d.get("approval_modes"),
            on_reject=d.get("on_reject"),
            on_modify=d.get("on_modify"),
            on_accept_as_is=d.get("on_accept_as_is"),
            hitl_mode=d.get("hitl_mode", "manual"),
            timeout_sec=d.get("timeout_sec", 1800),
            condition=d.get("condition"),
            fallback_on_failure=d.get("fallback_on_failure", "error"),
        )

    def _execute(self, state: PipelineState, resolver: VariableResolver) -> NodeResult:
        # Разрешаем промпт
        prompt = resolver.resolve(self.prompt_template)

        # Если auto-режим — автоматически approve
        if self.hitl_mode == "auto":
            state.set_gate_decision(self.node_id, "approve")
            return NodeResult(
                node_id=self.node_id,
                success=True,
                output={"decision": "approve", "auto": True},
                metadata={"gate_kind": self.gate_kind},
            )

        # Показываем артефакты
        print(f"\n{'='*60}")
        print(f"🔒 GATE: {self.node_id} ({self.gate_kind})")
        print(f"{'='*60}")
        print(prompt)

        for artifact_ref in self.show_artifacts:
            artifact_path = resolver.resolve(artifact_ref)
            if isinstance(artifact_path, str):
                p = Path(artifact_path)
                if p.exists():
                    print(f"\n--- {p} ---")
                    try:
                        content = p.read_text(encoding="utf-8")
                        # Truncate long content
                        if len(content) > 2000:
                            print(content[:2000])
                            print(f"... ({len(content) - 2000} more chars)")
                        else:
                            print(content)
                    except Exception as e:
                        print(f"[error reading: {e}]")

        # Спрашиваем решение
        modes_str = "/".join(self.approval_modes)
        try:
            decision = input(f"\nDecision [{modes_str}] (default: {self.approval_modes[0]}): ").strip().lower()
            if not decision:
                decision = self.approval_modes[0]
            if decision not in self.approval_modes:
                print(f"Unknown mode '{decision}', using default: {self.approval_modes[0]}")
                decision = self.approval_modes[0]
        except (EOFError, KeyboardInterrupt):
            decision = "reject"

        state.set_gate_decision(self.node_id, decision)

        # Обрабатываем решение
        if decision == "approve":
            return NodeResult(
                node_id=self.node_id,
                success=True,
                output={"decision": "approve"},
                metadata={"gate_kind": self.gate_kind},
            )
        elif decision == "reject":
            action = self.on_reject.get("action", "abort")
            target = self.on_reject.get("target")
            feedback_prompt = self.on_reject.get("feedback_prompt")
            return NodeResult(
                node_id=self.node_id,
                success=False,  # reject = failure для executor
                output={"decision": "reject", "action": action, "target": target},
                error=f"Gate rejected: {action} {target or ''}".strip(),
                metadata={
                    "gate_kind": self.gate_kind,
                    "on_reject_action": action,
                    "go_back_to": target,
                    "feedback_prompt": feedback_prompt,
                },
            )
        elif decision == "modify":
            modify_target = self.on_modify.get("modify_target")
            if modify_target:
                target_path = resolver.resolve(modify_target)
                if isinstance(target_path, str):
                    print(f"\n📝 Editing: {target_path}")
                    print("Open the file in your editor, modify, save. Press Enter when done.")
                    try:
                        input()
                    except (EOFError, KeyboardInterrupt):
                        pass
            return NodeResult(
                node_id=self.node_id,
                success=True,
                output={"decision": "modify", "modified_target": modify_target},
                metadata={"gate_kind": self.gate_kind},
            )
        elif decision == "accept_as_is":
            return NodeResult(
                node_id=self.node_id,
                success=True,
                output={"decision": "accept_as_is"},
                metadata={"gate_kind": self.gate_kind, "note": self.on_accept_as_is.get("note", "")},
            )
        else:
            return NodeResult(
                node_id=self.node_id,
                success=False,
                output={"decision": decision, "error": "unknown decision"},
                error=f"Unknown decision: {decision}",
            )

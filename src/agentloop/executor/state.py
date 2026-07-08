"""
PipelineState — состояние между узлами.

Хранит:
- outputs: dict[node_id, dict] — выходы каждого узла в памяти
- file_outputs: dict[node_id, list[Path]] — файлы, созданные узлом
- runtime_vars: $INPUT, $WORKDIR, $RUN_ID, $RUN_TIMESTAMP
- run_metadata: task_id, hypothesis_id, mode (research/production)
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..tools.base import ToolDescriptor


class VariableResolver:
    """
    Разрешает переменные в строках вида {node.output.field} и $VARS.

    Поддерживает фильтры: {var|length}, {var|is_empty}, {var|contains:'substr'}
    """

    VAR_PATTERN = re.compile(r"\{([^{}|]+)(\|[^{}]*)?\}")
    DOLLAR_PATTERN = re.compile(r"\$([A-Z_][A-Z0-9_]*)")

    def __init__(self, state: PipelineState):
        self.state = state

    def resolve(self, text: Any) -> Any:
        """
        Разрешает все переменные в тексте (или dict/list).
        Возвращает тот же тип, что и input.
        """
        if isinstance(text, str):
            return self._resolve_str(text)
        elif isinstance(text, dict):
            return {k: self.resolve(v) for k, v in text.items()}
        elif isinstance(text, list):
            return [self.resolve(item) for item in text]
        return text

    def _resolve_str(self, text: str) -> Any:
        # Сначала $VARS (runtime variables)
        def dollar_repl(m: re.Match) -> str:
            name = m.group(1)
            val = self.state.runtime_vars.get(name, m.group(0))
            return str(val) if not isinstance(val, Path) else str(val)

        text = self.DOLLAR_PATTERN.sub(dollar_repl, text)

        # Затем {node.output.field} ссылки
        matches = list(self.VAR_PATTERN.finditer(text))
        if not matches:
            return text

        # Если вся строка — одна переменная, возвращаем как есть (может быть не строка)
        if len(matches) == 1 and matches[0].group(0) == text:
            return self._resolve_var(matches[0].group(1), matches[0].group(2))

        # Иначе подставляем как строки
        def var_repl(m: re.Match) -> str:
            val = self._resolve_var(m.group(1), m.group(2))
            if isinstance(val, (dict, list)):
                return json.dumps(val, ensure_ascii=False, default=str)
            return str(val)

        return self.VAR_PATTERN.sub(var_repl, text)

    def _resolve_var(self, path: str, filter_str: str | None) -> Any:
        """Разрешает одну переменную вида 'node_id.output.field'."""
        parts = path.strip().split(".")
        if len(parts) < 2:
            return ""  # не смогли разрешить → пусто

        node_id = parts[0]
        field_path = parts[1:]

        # Достаём output узла
        node_output = self.state.outputs.get(node_id)
        if node_output is None:
            return ""

        # Если field_path начинается с "output" — пропускаем его
        # (формат {node.output.field} совместим с прямым доступом к полям)
        if field_path and field_path[0] == "output":
            field_path = field_path[1:]

        # Если после пропуска "output" ничего не осталось — возвращаем весь output
        if not field_path:
            return node_output

        # Идём по field_path
        current: Any = node_output
        for field in field_path:
            if isinstance(current, dict) and field in current:
                current = current[field]
            else:
                return ""  # поля нет → пусто, не literal

        # Применяем фильтр если есть
        if filter_str:
            current = self._apply_filter(current, filter_str.strip("|"))

        return current

    def _apply_filter(self, value: Any, filter_expr: str) -> Any:
        """Применяет фильтр: length, is_empty, contains:'...'."""
        if filter_expr == "length":
            if isinstance(value, (list, dict, str)):
                return len(value)
            return 0
        elif filter_expr == "is_empty":
            if isinstance(value, (list, dict, str)):
                return len(value) == 0
            return value is None
        elif filter_expr.startswith("contains:"):
            substr = filter_expr.split(":", 1)[1].strip("'\"")
            if isinstance(value, str):
                return substr in value
            if isinstance(value, list):
                return any(substr in str(item) for item in value)
            return False
        return value

    def resolve_condition(self, condition: str) -> bool:
        """
        Вычисляет условие вида:
        {node.field} == false
        {node.field} == true && {node.other|length} > 5
        {node.field|is_empty} == false
        """
        # Заменяем все переменные на их значения
        resolved = self._resolve_str(condition)
        if isinstance(resolved, bool):
            return resolved

        # Теперь парсим булевое выражение
        # Простая поддержка ==, !=, >, >=, <, <=, &&, ||
        return self._eval_bool_expr(str(resolved))

    def _eval_bool_expr(self, expr: str) -> bool:
        """Простейший eval булевых выражений."""
        # Разделяем по || (or)
        or_parts = expr.split("||")
        for or_part in or_parts:
            # Разделяем по && (and)
            and_parts = or_part.split("&&")
            all_true = True
            for part in and_parts:
                if not self._eval_single_condition(part.strip()):
                    all_true = False
                    break
            if all_true:
                return True
        return False

    def _eval_single_condition(self, cond: str) -> bool:
        """Вычисляет одно условие: A == B, A != B, A > B, etc."""
        cond = cond.strip()
        if not cond:
            return True

        for op in ["==", "!=", ">=", "<=", ">", "<"]:
            if op in cond:
                left, right = cond.split(op, 1)
                left = left.strip().strip("'\"")
                right = right.strip().strip("'\"")
                return self._compare(left, right, op)

        # Без оператора — трактуем как bool
        return cond.lower() in ("true", "1", "yes")


    def _compare(self, left: str, right: str, op: str) -> bool:
        """Сравнивает два значения."""
        # Нормализуем boolean
        def to_num_or_bool(s: str):
            sl = s.lower()
            if sl in ("true", "yes"):
                return True
            if sl in ("false", "no"):
                return False
            try:
                if "." in s:
                    return float(s)
                return int(s)
            except ValueError:
                return s

        l, r = to_num_or_bool(left), to_num_or_bool(right)

        # Если типы разные (bool vs str) — приводим к строке для ==
        if type(l) != type(r):
            if op in ("==", "!="):
                # Пробуем сравнить как строки
                ls, rs = str(l).lower(), str(r).lower()
                if op == "==":
                    return ls == rs
                else:
                    return ls != rs

        if op == "==":
            return l == r
        elif op == "!=":
            return l != r
        elif op == ">":
            return l > r
        elif op == ">=":
            return l >= r
        elif op == "<":
            return l < r
        elif op == "<=":
            return l <= r
        return False


class PipelineState:
    """
    Состояние pipeline во время выполнения.

    Хранит:
    - outputs: dict[node_id, dict] — выходы узлов в памяти
    - file_outputs: dict[node_id, list[Path]]
    - runtime_vars: $WORKDIR, $RUN_ID, $INPUT и т.д.
    - run_metadata: task_id, hypothesis_id, mode
    """

    def __init__(
        self,
        work_dir: Path | str,
        runtime_vars: dict[str, Any] | None = None,
        run_id: str | None = None,
        task_id: str | None = None,
        hypothesis_id: str | None = None,
        mode: str = "research",
    ):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id or f"run_{int(time.time())}"
        self.task_id = task_id or "unknown"
        self.hypothesis_id = hypothesis_id or "unknown"
        self.mode = mode

        self.outputs: dict[str, dict[str, Any]] = {}
        self.file_outputs: dict[str, list[Path]] = {}
        self.errors: dict[str, str] = {}  # node_id → error message
        self.gate_decisions: dict[str, str] = {}  # node_id → approve/reject/modify

        # Runtime vars
        self.runtime_vars: dict[str, Any] = {
            "WORKDIR": str(self.work_dir),
            "RUN_ID": self.run_id,
            "RUN_TIMESTAMP": datetime.now(timezone.utc).isoformat(),
            "INPUT": "",  # должен быть установлен externally
            "PAGE_NUM": "1",
        }
        if runtime_vars:
            self.runtime_vars.update(runtime_vars)

    def set_input(self, input_value: Any) -> None:
        """Устанавливает $INPUT."""
        self.runtime_vars["INPUT"] = str(input_value) if not isinstance(input_value, str) else input_value

    def set_var(self, name: str, value: Any) -> None:
        """Устанавливает runtime-переменную."""
        self.runtime_vars[name] = value

    def set_output(self, node_id: str, output: dict[str, Any]) -> None:
        """Сохраняет выход узла."""
        self.outputs[node_id] = output
        # Также сохраняем на диск для resume
        out_file = self.work_dir / f"node_{node_id}_output.json"
        try:
            out_file.write_text(
                json.dumps(output, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    def get_output(self, node_id: str) -> dict[str, Any] | None:
        return self.outputs.get(node_id)

    def add_files(self, node_id: str, files: list[Path | str]) -> None:
        """Регистрирует файлы, созданные узлом."""
        self.file_outputs[node_id] = [Path(f) for f in files]

    def set_error(self, node_id: str, error: str) -> None:
        self.errors[node_id] = error

    def set_gate_decision(self, node_id: str, decision: str) -> None:
        self.gate_decisions[node_id] = decision

    def resolver(self) -> VariableResolver:
        return VariableResolver(self)

    def snapshot(self) -> dict[str, Any]:
        """Снимок состояния для checkpoint."""
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "hypothesis_id": self.hypothesis_id,
            "mode": self.mode,
            "runtime_vars": self.runtime_vars,
            "outputs": self.outputs,
            "errors": self.errors,
            "gate_decisions": self.gate_decisions,
            "completed_nodes": list(self.outputs.keys()),
        }

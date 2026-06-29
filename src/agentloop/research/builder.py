"""
PipelineBuilder — LLM-агент, строящий DAG из гипотезы.

Вход: гипотеза + catalog инструментов
Выход: DAG JSON (формат из design/2_dag.json)

Может обращаться к Steward для поиска/создания инструментов.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..providers import Message, Provider
from ..providers.base import Response
from ..tools import Steward, ToolCatalog, core_tools_for_builder_prompt
from .hypothesis import Hypothesis


@dataclass
class BuildResult:
    """Результат построения DAG."""

    hypothesis_id: str
    dag: dict[str, Any]
    success: bool
    steward_requests: list[dict[str, Any]] = field(default_factory=list)
    custom_tools_created: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineBuilder:
    """
    LLM-агент для построения DAG из гипотезы.

    Использует Steward для:
    - Поиска доступных инструментов (Layer 2)
    - Создания custom Python-инструментов (Layer 3)
    """

    SYSTEM_PROMPT = """Ты — PipelineBuilder в адаптивной системе пайплайнов.
Твоя задача: построить DAG (Directed Acyclic Graph) пайплайна на основе гипотезы.

Принципы:
1. Используй БАЗОВЫЕ инструменты (Layer 1) где возможно
2. Для специализированных задач — запрашивай инструменты через Steward
3. Узлы DAG: bash, llm, python, file, loop, gate
4. Передавай данные между узлами через {node_id.output.field}
5. Для LLM-узлов указывай model в формате "provider:model_name"
6. Для циклов используй узел типа "loop" с body и exit_condition
7. Для human approval — узел "gate"

Формат ответа: JSON с полями:
- nodes: массив узлов
- edges: массив {from, to}
- entry: id стартового узла
- exit: id финального узла
- prompts: {prompt_name: {text: "...", version: "v1"}}
- steward_requests: массив {purpose, query} — если нужны custom tools
- prerequisites: массив {type, request_id, status}

Каждый узел имеет:
- id: уникальное имя
- type: "bash" | "llm" | "python" | "file" | "loop" | "gate"
- специфичные для типа поля

Примеры узлов:

bash: {"id": "n1", "type": "bash", "command": "echo $MSG", "timeout_sec": 5}
llm: {"id": "n2", "type": "llm", "model": "local:gemma-4-26b", "system_prompt_ref": "extractor_v1", "user_prompt_template": "Извлеки: {$INPUT}", "json_mode": true}
python: {"id": "n3", "type": "python", "script_ref": "core:json_merge", "input": {"a": "{n2.output}"}}
file: {"id": "n4", "type": "file", "operation": "write", "path": "/tmp/out.txt", "content_from": "{n2.output.content}"}
loop: {"id": "n5", "type": "loop", "body": [...], "exit_condition": "{n6.output.done} == true", "max_iterations": 3}
gate: {"id": "n7", "type": "gate", "gate_kind": "human_approval", "prompt_template": "Approve?"}

Возвращай ТОЛЬКО JSON, без markdown обёртки."""

    def __init__(
        self,
        llm_provider: Provider,
        model: str = "gemma-4-26b",
        catalog: ToolCatalog | None = None,
        steward: Steward | None = None,
        max_steward_calls: int = 3,
    ):
        self.llm = llm_provider
        self.model = model
        self.catalog = catalog
        self.steward = steward
        self.max_steward_calls = max_steward_calls

    def build(self, hypothesis: Hypothesis, task_id: str = "task") -> BuildResult:
        """
        Строит DAG для гипотезы.

        Args:
            hypothesis: гипотеза с approach, model_assignments, custom_tools_needed
            task_id: ID задачи

        Returns:
            BuildResult с DAG или ошибкой
        """
        # Сначала обрабатываем custom_tools_needed из гипотезы
        steward_requests_log: list[dict[str, Any]] = []
        custom_tools_created: list[str] = []

        if hypothesis.custom_tools_needed and self.steward:
            for tool_spec in hypothesis.custom_tools_needed[:self.max_steward_calls]:
                # Ищем существующий
                query = tool_spec.get("purpose", tool_spec.get("proposed_name", "tool"))
                search_result = self.steward.search(query)

                if search_result.found:
                    # Нашли в Layer 2/3 — добавляем в лог
                    steward_requests_log.append({
                        "purpose": query,
                        "found": [t.name for t in search_result.found[:3]],
                        "created": False,
                    })
                elif search_result.custom_tool_possible or tool_spec.get("proposed_name"):
                    # Создаём custom Python-инструмент
                    from ..tools.base import CustomToolSpec

                    spec = CustomToolSpec(
                        name=tool_spec.get("proposed_name", f"custom_{int(time.time())}"),
                        description=tool_spec.get("purpose", query),
                        input_schema=tool_spec.get("input_schema", {}),
                        output_schema=tool_spec.get("output_schema", {}),
                        dependencies=tool_spec.get("estimated_dependencies", []),
                        implementation_hint=tool_spec.get("steward_request_hint", ""),
                    )
                    try:
                        result = self.steward.create_custom(spec)
                        if result.status == "available":
                            custom_tools_created.append(result.tool_id)
                            steward_requests_log.append({
                                "purpose": query,
                                "found": [],
                                "created": True,
                                "tool_id": result.tool_id,
                            })
                        else:
                            steward_requests_log.append({
                                "purpose": query,
                                "found": [],
                                "created": False,
                                "error": result.error,
                            })
                    except Exception as e:
                        steward_requests_log.append({
                            "purpose": query,
                            "found": [],
                            "created": False,
                            "error": str(e),
                        })

        # Строим промпт для LLM
        user_prompt = self._build_user_prompt(hypothesis, custom_tools_created)

        # Вызываем LLM
        start = time.time()
        try:
            response: Response = self.llm.chat(
                messages=[
                    Message(role="system", content=self.SYSTEM_PROMPT),
                    Message(role="user", content=user_prompt),
                ],
                model=self.model,
                temperature=0.3,  # ниже температура для детерминированности
                max_tokens=4096,
                json_mode=True,
                timeout_sec=300,
            )
        except Exception as e:
            return BuildResult(
                hypothesis_id=hypothesis.id,
                dag={},
                success=False,
                steward_requests=steward_requests_log,
                custom_tools_created=custom_tools_created,
                error=f"LLM call failed: {e}",
            )

        # Парсим DAG
        dag = self._parse_dag(response.content)
        if not dag:
            return BuildResult(
                hypothesis_id=hypothesis.id,
                dag={},
                success=False,
                steward_requests=steward_requests_log,
                custom_tools_created=custom_tools_created,
                error="Failed to parse DAG from LLM response",
            )

        # Валидируем DAG
        validation_error = self._validate_dag(dag)
        if validation_error:
            return BuildResult(
                hypothesis_id=hypothesis.id,
                dag=dag,
                success=False,
                steward_requests=steward_requests_log,
                custom_tools_created=custom_tools_created,
                error=f"DAG validation failed: {validation_error}",
            )

        # Добавляем steward_requests в DAG
        if "steward_requests" not in dag:
            dag["steward_requests"] = []
        dag["steward_requests"].extend(steward_requests_log)

        return BuildResult(
            hypothesis_id=hypothesis.id,
            dag=dag,
            success=True,
            steward_requests=steward_requests_log,
            custom_tools_created=custom_tools_created,
            metadata={
                "builder_model": self.model,
                "build_time_sec": int(time.time() - start),
                "llm_tokens": response.output_tokens,
                "llm_cost_usd": response.cost_usd,
                "built_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _build_user_prompt(self, hypothesis: Hypothesis, custom_tools: list[str]) -> str:
        """Строит промпт для LLM-билдера."""
        parts = [f"## Гипотеза: {hypothesis.title}\n"]
        parts.append(f"**Rationale:** {hypothesis.rationale}\n")

        parts.append("\n## Подход\n")
        for i, step in enumerate(hypothesis.approach, 1):
            parts.append(f"{i}. {step}\n")

        parts.append("\n## Назначения моделей\n")
        for ma in hypothesis.model_assignments:
            parts.append(f"- **{ma.get('role', '?')}**: `{ma.get('model', '?')}` — {ma.get('reason', '')}\n")

        # Базовые инструменты
        parts.append(f"\n## Базовые инструменты (Layer 1)\n{core_tools_for_builder_prompt()}\n")

        # Custom tools (если созданы)
        if custom_tools:
            parts.append("\n## Доступные custom инструменты\n")
            for tool_id in custom_tools:
                parts.append(f"- `{tool_id}`\n")

        # Задача для Builder'а
        parts.append(
            "\n## Задача\n"
            "Построй DAG пайплайна на основе этой гипотезы.\n"
            "Используй назначенные модели и подход.\n"
            "Все шаги из approach должны быть отражены в узлах DAG.\n"
            "Возвращай JSON."
        )

        return "\n".join(parts)

    def _parse_dag(self, content: str) -> dict[str, Any] | None:
        """Парсит DAG из ответа LLM."""
        # Пробуем прямой JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Пробуем извлечь JSON из текста
        start = content.find("{")
        if start == -1:
            return None
        end = content.rfind("}")
        if end == -1 or end <= start:
            return None

        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return None

    def _validate_dag(self, dag: dict[str, Any]) -> str | None:
        """Валидирует структуру DAG. Возвращает ошибку или None."""
        if not isinstance(dag, dict):
            return "DAG is not a dict"

        nodes = dag.get("nodes")
        if not nodes or not isinstance(nodes, list):
            return "DAG missing 'nodes' array"

        if len(nodes) == 0:
            return "DAG has no nodes"

        # Проверяем, что все узлы имеют id и type
        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                return f"Node {i} is not a dict"
            if "id" not in node:
                return f"Node {i} missing 'id'"
            if "type" not in node:
                return f"Node {node['id']} missing 'type'"

        # Проверяем entry и exit
        node_ids = {n["id"] for n in nodes}
        if "entry" in dag and dag["entry"] not in node_ids:
            return f"Entry node '{dag['entry']}' not in nodes"
        if "exit" in dag and dag["exit"] not in node_ids:
            return f"Exit node '{dag['exit']}' not in nodes"

        # Проверяем edges
        edges = dag.get("edges", [])
        for edge in edges:
            if not isinstance(edge, dict):
                return f"Edge is not a dict: {edge}"
            if "from" not in edge or "to" not in edge:
                return f"Edge missing from/to: {edge}"
            if edge["from"] not in node_ids:
                return f"Edge from unknown node: {edge['from']}"
            if edge["to"] not in node_ids:
                return f"Edge to unknown node: {edge['to']}"

        return None

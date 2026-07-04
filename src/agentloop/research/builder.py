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
from ..executor.nodes.python import resolve_script_path
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

КРИТИЧЕСКИЕ ПРАВИЛА ДЛЯ LLM-УЗЛОВ:
- ВСЕГДА указывай "user_prompt_template" — это текст, который отправляется как user-сообщение.
  БЕЗ user_prompt_template модель получит пустой запрос и вернёт пустой результат.
- "system_prompt_ref" задаёт роль/поведение модели, "user_prompt_template" — конкретную задачу.
- user_prompt_template может ссылаться на переменные: {$INPUT}, {prev_node.output.field}.
- Если нужно вызвать LLM без контекста — user_prompt_template = повторение сути задачи.

КРИТИЧЕСКИЕ ПРАВИЛА ДЛЯ BASH-УЗЛОВ:
- НЕ ВЫДУМЫВАЙ внешние утилиты. Если задаче нужен сторонний инструмент (ollama, docker, pandoc,
  libreoffice, tesseract и т.п.), которого может не быть в окружении — НЕ добавляй bash-узлы
  для его запуска/установки. Вместо этого реши задачу доступными средствами (LLM-узлы, core-скрипты,
  стандартные утилиты вроде echo/cat/grep/python3).
- Если внешний инструмент ОБЯЗАТЕЛЕНЕН — добавь его в prerequisites со status "required",
  но предпочти альтернативу без внешних зависимостей.

КРИТИЧЕСКИЕ ПРАВИЛА ДЛЯ PYTHON-УЗЛОВ:
- script_ref с префиксом "core:" может ссылаться ТОЛЬКО на существующие core-скрипты:
  {available_core_scripts}. НЕ ВЫДУМЫВАЙ несуществующие core-скрипты (например, core:json_validate).
- Для специализированной логики используй "custom:" (Steward создаст инструмент) или "llm"-узлы.

Пример ПРАВИЛЬНОГО LLM-узла:
{"id": "n1", "type": "llm", "model": "local:gemma-4-26b",
 "system_prompt_ref": "extractor_v1",
 "user_prompt_template": "Извлеки все задачи из следующего текста: {$INPUT}",
 "json_mode": true}

Пример НЕПРАВИЛЬНОГО (запрещено — нет user_prompt_template):
{"id": "n1", "type": "llm", "model": "local:gemma-4-26b",
 "system_prompt_ref": "extractor_v1", "json_mode": true}  ← БЕЗ user_prompt_template!

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
        # Подставляем реальный список доступных core-скриптов в системный промпт,
        # чтобы LLM не выдумывал несуществующие вроде core:json_validate.
        self.SYSTEM_PROMPT = self.SYSTEM_PROMPT.replace(
            "{available_core_scripts}", ", ".join(self._available_core_scripts())
        )

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

        # Вызываем LLM с retry на reasoning-перерасход и транзиентные ошибки.
        # Reasoning-модели иногда тратят весь бюджет на chain-of-thought,
        # оставляя content пустым (finish_reason=length).
        # Локальная LLM (llama-server) иногда таймаутит на сложных задачах —
        # повторяем с тем же max_tokens после паузы.
        start = time.time()
        response: Response | None = None
        current_max_tokens = 8192  # reasoning-моделям нужно больше (было 4096)
        last_error: str | None = None
        for attempt in range(3):
            try:
                response = self.llm.chat(
                    messages=[
                        Message(role="system", content=self.SYSTEM_PROMPT),
                        Message(role="user", content=user_prompt),
                    ],
                    model=self.model,
                    temperature=0.3,  # ниже температура для детерминированности
                    max_tokens=current_max_tokens,
                    json_mode=True,
                    timeout_sec=300,
                )
                last_error = None
            except Exception as e:
                last_error = str(e)
                error_str = str(e).lower()
                # Транзиентные ошибки — retry с тем же max_tokens
                if any(p in error_str for p in [
                    "timeout", "timed out", "connection error",
                    "connection reset", "429", "502", "503", "504",
                ]):
                    import time as _time
                    _time.sleep(2 ** attempt)
                    continue
                # Нетранзиентная — сразу возвращаем ошибку
                return BuildResult(
                    hypothesis_id=hypothesis.id,
                    dag={},
                    success=False,
                    steward_requests=steward_requests_log,
                    custom_tools_created=custom_tools_created,
                    error=f"LLM call failed: {e}",
                )

            content = response.content or ""
            # Если content непустой или не reasoning-перерасход — выходим
            if content.strip() or response.finish_reason != "length":
                break

            # Retry с увеличенным max_tokens
            current_max_tokens = min(current_max_tokens * 2, 32768)

        # Если все попытки провалились с ошибкой
        if last_error is not None and (response is None or not (response.content or "").strip()):
            return BuildResult(
                hypothesis_id=hypothesis.id,
                dag={},
                success=False,
                steward_requests=steward_requests_log,
                custom_tools_created=custom_tools_created,
                error=f"LLM call failed after retries: {last_error}",
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
        """Парсит DAG из ответа LLM.

        Устойчив к:
        - markdown-обёрткам (```json ... ```)
        - обрезанному JSON (reasoning-модель не успела закрыть скобки)
        - JSON, встроенному в текст (ищет первый { ... })
        """
        # Strip markdown code fences
        stripped = self._strip_code_fences(content)
        if stripped != content:
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass

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

        candidate = content[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Repair: возможно JSON обрезан — попытаемся закрыть незакрытые скобки
            repaired = self._repair_truncated_json(candidate)
            if repaired != candidate:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass

        return None

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Убирает ```json / ``` обёртки, которые LLM иногда добавляет."""
        import re
        # ```json ... ``` или ``` ... ```
        pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
        m = pattern.search(text)
        if m and m.group(1).strip():
            return m.group(1).strip()
        return text

    @staticmethod
    def _repair_truncated_json(text: str) -> str:
        """Пытается закрыть незакрытые скобки/кавычки в обрезанном JSON.

        Максимальная глубина ремонта — 5 уровней. Не пытается вставлять
        отсутствующие значения (это задача retry с бОльшим max_tokens).
        """
        # Убираем висячую запятую перед обрезом
        text = text.rstrip()
        while text and text[-1] in (",", " ", "\n", "\t"):
            text = text[:-1]

        # Считаем незакрытые { [ " (в порядке, обратном появлению)
        open_braces = 0
        open_brackets = 0
        open_quotes = False
        i = 0
        while i < len(text):
            c = text[i]
            if c == '"' and (i == 0 or text[i - 1] != "\\"):
                open_quotes = not open_quotes
            elif not open_quotes:
                if c == "{":
                    open_braces += 1
                elif c == "}":
                    open_braces -= 1
                elif c == "[":
                    open_brackets += 1
                elif c == "]":
                    open_brackets -= 1
            i += 1

        if open_quotes:
            text += '"'
        # Закрываем в обратном порядке
        for _ in range(min(open_brackets, 5)):
            text += "]"
        for _ in range(min(open_braces, 5)):
            text += "}"

        return text

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

        # Проверяем LLM-узлы на обязательные поля
        for node in nodes:
            if node.get("type") == "llm":
                node_id = node.get("id", "?")
                if not node.get("user_prompt_template"):
                    return (
                        f"LLM node '{node_id}' missing 'user_prompt_template'. "
                        f"Without it, the LLM receives an empty prompt and returns empty output. "
                        f"Add user_prompt_template with the actual task text."
                    )
                if not node.get("model"):
                    return f"LLM node '{node_id}' missing 'model'"

        # Проверяем python-узлы: script_ref должен разрешаться в существующий файл.
        # Готовых core-скриптов мало, и LLM нередко выдумывает несуществующие вроде
        # 'core:json_validate' — такой узел всё равно упадёт в runtime, поэтому ловим заранее.
        # Для custom: проверяем, что Steward создал инструмент (иначе runtime упадёт с
        # "Cannot find script: custom:json_parser_v1", как было в реальном прогоне).
        for node in nodes:
            if node.get("type") == "python":
                node_id = node.get("id", "?")
                script_ref = node.get("script_ref")
                if not script_ref:
                    return f"Python node '{node_id}' missing 'script_ref'"
                if script_ref.startswith("core:"):
                    if resolve_script_path(script_ref) is None:
                        name = script_ref[5:]
                        return (
                            f"Python node '{node_id}' references unknown core script '{script_ref}'. "
                            f"No 'core_scripts/{name}.py' exists. "
                            f"Available core scripts: {self._available_core_scripts()}."
                        )
                elif script_ref.startswith("custom:"):
                    # Если Steward недоступен или не создал этот инструмент — упадём в runtime.
                    # Ловим здесь, чтобы дать осмысленную ошибку и подсказку.
                    if resolve_script_path(script_ref) is None:
                        name = script_ref[7:]
                        if self.steward is None:
                            return (
                                f"Python node '{node_id}' references custom script '{script_ref}', "
                                f"but Steward is not enabled (no --steward flag). "
                                f"Either enable Steward to create custom tools, or use only core: scripts."
                            )
                        else:
                            return (
                                f"Python node '{node_id}' references custom script '{script_ref}' "
                                f"which was not created by Steward. "
                                f"Check steward_requests_log for creation errors. "
                                f"Available core scripts: {self._available_core_scripts()}."
                            )

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

        # Проверяем prerequisites: если билдер сам отметил зависимость как
        # неудовлетворённую (status != satisfied), пайплайн всё равно упадёт в runtime
        # (например, 'ollama: command not found'). Фейлим сразу с понятной причиной.
        prerequisites = dag.get("prerequisites", [])
        for prereq in prerequisites:
            if not isinstance(prereq, dict):
                continue
            status = prereq.get("status")
            request_id = prereq.get("request_id", "?")
            if status and status != "satisfied":
                return (
                    f"Unsatisfied prerequisite '{request_id}' (status='{status}'). "
                    f"The pipeline depends on '{request_id}' which is not available in this environment. "
                    f"Reformulate the hypothesis without external dependencies, or satisfy the prerequisite."
                )

        return None

    @staticmethod
    def _available_core_scripts() -> list[str]:
        """Возвращает список доступных core-скриптов для подсказки в ошибке."""
        from ..executor.nodes.python import CORE_SCRIPTS_DIR
        return sorted(p.stem for p in CORE_SCRIPTS_DIR.glob("*.py") if p.stem != "__init__")

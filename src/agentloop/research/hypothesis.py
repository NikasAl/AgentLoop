"""
HypothesisGenerator — LLM-агент, генерирующий гипотезы по задаче.

Вход: задача + критерии + реестр моделей + история прошлых попыток
Выход: 3-5 гипотез в JSON (формат из design/1_hypothesis.json)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..providers import Message, Provider, get_provider
from ..providers.base import Response
from ..tools import ToolCatalog


@dataclass
class Hypothesis:
    """Одна гипотеза о том, как решать задачу."""

    id: str
    title: str
    rationale: str
    approach: list[str]
    model_assignments: list[dict[str, str]]
    custom_tools_needed: list[dict[str, Any]]
    estimated: dict[str, Any]
    risks: list[str]
    test_sample_size: int = 1
    user_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "rationale": self.rationale,
            "approach": self.approach,
            "model_assignments": self.model_assignments,
            "custom_tools_needed": self.custom_tools_needed,
            "estimated": self.estimated,
            "risks": self.risks,
            "test_sample_size": self.test_sample_size,
            "user_hint": self.user_hint,
        }


@dataclass
class HypothesisSet:
    """Набор гипотез для одной research-итерации."""

    task_id: str
    task_description: str
    sample_input: dict[str, Any]
    user_constraints: dict[str, Any]
    user_hint: str | None
    hypotheses: list[Hypothesis]
    metadata: dict[str, Any] = field(default_factory=dict)


class HypothesisGenerator:
    """
    LLM-агент для генерации гипотез.

    Использует локальную модель (gemma-4-26b) или human-as-API для отладки.
    """

    SYSTEM_PROMPT = """Ты — HypothesisGenerator в адаптивной системе пайплайнов.
Твоя задача: проанализировать задачу пользователя и предложить 3 разные гипотезы о том,
как её можно решить через pipeline из LLM-вызовов и bash/python инструментов.

Принципы:
1. Каждая гипотеза — это **другой подход**, не вариация одной идеи
2. Учитывай доступные модели и их стоимость
3. Учитывай бюджет (если задан)
4. Описывай риски честно
5. Если у пользователя есть hint — учитывай его

Формат ответа: JSON с полем "hypotheses" — массив из 3 объектов.
Каждый объект имеет поля:
- id: "h1", "h2", "h3"
- title: краткое название (1 строка)
- rationale: почему этот подход может сработать (2-3 предложения)
- approach: массив шагов пайплайна (на естественном языке)
- model_assignments: массив {role, model, reason}
- custom_tools_needed: массив {purpose, proposed_name, type, estimated_dependencies}
- estimated: {tokens_per_page, latency_per_page_sec, cost_per_page_usd, expected_quality, confidence_in_estimate}
- risks: массив строк
- test_sample_size: int (обычно 1-5)

Возвращай ТОЛЬКО JSON, без markdown обёртки."""

    def __init__(
        self,
        llm_provider: Provider,
        model: str = "gemma-4-26b",
        catalog: ToolCatalog | None = None,
    ):
        self.llm = llm_provider
        self.model = model
        self.catalog = catalog

    def generate(
        self,
        task_description: str,
        sample_input: dict[str, Any] | None = None,
        user_constraints: dict[str, Any] | None = None,
        user_hint: str | None = None,
        task_id: str = "task",
        history: list[dict[str, Any]] | None = None,
    ) -> HypothesisSet:
        """
        Генерирует 3 гипотезы для задачи.

        Args:
            task_description: что нужно сделать
            sample_input: пример входных данных
            user_constraints: бюджет, лимиты
            user_hint: подсказка от пользователя
            task_id: ID задачи
            history: история прошлых попыток [{hypothesis_id, score, feedback}]

        Returns:
            HypothesisSet с 3 гипотезами
        """
        sample_input = sample_input or {}
        user_constraints = user_constraints or {}
        history = history or []

        user_prompt = self._build_user_prompt(
            task_description=task_description,
            sample_input=sample_input,
            user_constraints=user_constraints,
            user_hint=user_hint,
            history=history,
        )

        start = time.time()
        try:
            response: Response = self.llm.chat(
                messages=[
                    Message(role="system", content=self.SYSTEM_PROMPT),
                    Message(role="user", content=user_prompt),
                ],
                model=self.model,
                temperature=0.8,  # выше температура для разнообразия
                max_tokens=4096,
                json_mode=True,
                timeout_sec=300,
            )
        except Exception as e:
            # Fallback: возвращаем одну тривиальную гипотезу
            return HypothesisSet(
                task_id=task_id,
                task_description=task_description,
                sample_input=sample_input,
                user_constraints=user_constraints,
                user_hint=user_hint,
                hypotheses=[self._fallback_hypothesis(task_description, str(e))],
                metadata={"error": str(e), "fallback": True},
            )

        # Парсим JSON
        try:
            data = json.loads(response.content)
            hypotheses_data = data.get("hypotheses", [])
        except json.JSONDecodeError:
            # Пробуем извлечь JSON из текста
            json_str = self._extract_json(response.content)
            if json_str:
                try:
                    data = json.loads(json_str)
                    hypotheses_data = data.get("hypotheses", [])
                except json.JSONDecodeError:
                    hypotheses_data = []
            else:
                hypotheses_data = []

        hypotheses: list[Hypothesis] = []
        for i, h_data in enumerate(hypotheses_data[:5]):  # максимум 5 гипотез
            h = self._parse_hypothesis(h_data, i)
            hypotheses.append(h)

        if not hypotheses:
            hypotheses.append(self._fallback_hypothesis(task_description, "No hypotheses parsed"))

        return HypothesisSet(
            task_id=task_id,
            task_description=task_description,
            sample_input=sample_input,
            user_constraints=user_constraints,
            user_hint=user_hint,
            hypotheses=hypotheses,
            metadata={
                "generated_by": "HypothesisGenerator",
                "model": self.model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generation_tokens": response.output_tokens,
                "generation_cost_usd": response.cost_usd,
                "generation_latency_sec": int(time.time() - start),
                "history_aware": len(history) > 0,
                "previous_attempts_considered": len(history),
            },
        )

    def _build_user_prompt(
        self,
        task_description: str,
        sample_input: dict[str, Any],
        user_constraints: dict[str, Any],
        user_hint: str | None,
        history: list[dict[str, Any]],
    ) -> str:
        parts = [f"## Задача\n{task_description}\n"]

        if sample_input:
            parts.append(f"## Пример входных данных\n```json\n{json.dumps(sample_input, ensure_ascii=False, indent=2)}\n```\n")

        if user_constraints:
            parts.append(f"## Ограничения\n```json\n{json.dumps(user_constraints, ensure_ascii=False, indent=2)}\n```\n")

        if user_hint:
            parts.append(f"## Подсказка от пользователя\n{user_hint}\n")

        # Доступные модели (компактно)
        if self.catalog:
            parts.append("## Доступные инструменты\nСм. ToolCatalog. Базовые: bash_run, python_run, llm_call, wait_human, web_search, web_fetch, file_op. Для остальных — обратись к Steward.\n")

        # История
        if history:
            parts.append("## История прошлых попыток\n")
            for h in history[-5:]:  # последние 5
                parts.append(f"- {h.get('hypothesis_id', '?')}: score={h.get('score', '?')}, feedback={h.get('feedback', '')[:200]}\n")
            parts.append("\nУчти эти ошибки при генерации новых гипотез. Не повторяй подходы, которые уже провалились.\n")
        else:
            parts.append("## История\nЭто первая итерация. Генерируй максимально разные подходы.\n")

        parts.append("\nСгенерируй 3 гипотезы в формате JSON.")
        return "\n".join(parts)

    def _parse_hypothesis(self, data: dict[str, Any], idx: int) -> Hypothesis:
        """Парсит dict в Hypothesis с дефолтами."""
        return Hypothesis(
            id=data.get("id", f"h{idx + 1}"),
            title=data.get("title", f"Hypothesis {idx + 1}"),
            rationale=data.get("rationale", ""),
            approach=data.get("approach", []),
            model_assignments=data.get("model_assignments", []),
            custom_tools_needed=data.get("custom_tools_needed", []),
            estimated=data.get("estimated", {}),
            risks=data.get("risks", []),
            test_sample_size=data.get("test_sample_size", 1),
            user_hint=data.get("user_hint"),
        )

    def _fallback_hypothesis(self, task_description: str, error: str) -> Hypothesis:
        """Простая гипотеза-заглушка при ошибке LLM."""
        return Hypothesis(
            id="h_fallback",
            title="Fallback: single LLM call",
            rationale=f"LLM generation failed ({error}). Fallback: один LLM-вызов для всей задачи.",
            approach=[
                "Один LLM-вызов к локальной модели с описанием задачи",
                "Возврат результата напрямую",
            ],
            model_assignments=[
                {"role": "main", "model": "local:gemma-4-26b", "reason": "fallback"}
            ],
            custom_tools_needed=[],
            estimated={
                "tokens_per_page": 2000,
                "latency_per_page_sec": 60,
                "cost_per_page_usd": 0.0,
                "expected_quality": "low",
                "confidence_in_estimate": "low",
            },
            risks=["LLM generation failed, fallback may not be optimal"],
            test_sample_size=1,
        )

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Пытается найти JSON в тексте."""
        start = text.find("{")
        if start == -1:
            return None
        # Идём от конца, ищем последнюю }
        end = text.rfind("}")
        if end == -1 or end <= start:
            return None
        return text[start : end + 1]

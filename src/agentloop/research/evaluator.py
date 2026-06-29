"""
Evaluator — оценивает результат выполнения pipeline.

Поддерживает:
- Структурные проверки (exit_code, JSON schema, file existence)
- LLM-as-judge (опционально, для субъективных метрик)
- Composite score с весами из evaluator config
- Actionable feedback для следующей итерации
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..executor.executor import ExecutionResult
from ..providers import Message, Provider
from ..providers.base import Response


@dataclass
class Metric:
    """Одна метрика оценки."""

    name: str
    value: float
    weight: float = 1.0
    direction: str = "max"  # "max" | "min"
    evaluator: str = "unknown"
    description: str = ""
    automated: bool = True
    passed_threshold: bool = True
    threshold: float | None = None


@dataclass
class EvaluationResult:
    """Результат оценки pipeline."""

    run_id: str
    hypothesis_id: str
    success: bool
    metrics: list[Metric] = field(default_factory=list)
    composite_score: float = 0.0
    feedback: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Evaluator:
    """
    Оценивает результат выполнения pipeline.

    По умолчанию использует структурные проверки (без LLM).
    При наличии llm_provider — может использовать LLM-as-judge.
    """

    def __init__(
        self,
        llm_provider: Provider | None = None,
        judge_model: str = "gemma-4-26b",
        evaluator_config: dict[str, Any] | None = None,
    ):
        self.llm = llm_provider
        self.judge_model = judge_model
        self.config = evaluator_config or self._default_config()

    def _default_config(self) -> dict[str, Any]:
        """Конфиг по умолчанию (универсальный)."""
        return {
            "metrics": [
                {
                    "name": "execution_success",
                    "evaluator": "from_logs",
                    "direction": "max",
                    "weight": 0.4,
                    "threshold": 1.0,
                },
                {
                    "name": "output_completeness",
                    "evaluator": "structural",
                    "direction": "max",
                    "weight": 0.3,
                },
                {
                    "name": "cost_efficiency",
                    "evaluator": "from_logs",
                    "direction": "min",
                    "weight": 0.15,
                },
                {
                    "name": "latency_efficiency",
                    "evaluator": "from_logs",
                    "direction": "min",
                    "weight": 0.15,
                },
            ],
            "composite_score": {"method": "weighted_sum"},
        }

    def evaluate(
        self,
        execution_result: ExecutionResult,
        hypothesis_id: str = "unknown",
    ) -> EvaluationResult:
        """
        Оценивает результат выполнения.

        Args:
            execution_result: результат PipelineExecutor
            hypothesis_id: ID гипотезы (для логирования)

        Returns:
            EvaluationResult с метриками и composite score
        """
        metrics: list[Metric] = []

        # 1. Execution success (бинарная)
        success_metric = Metric(
            name="execution_success",
            value=1.0 if execution_result.success else 0.0,
            weight=0.4,
            direction="max",
            evaluator="from_logs",
            description="Pipeline завершился успешно",
            automated=True,
            threshold=1.0,
            passed_threshold=execution_result.success,
        )
        metrics.append(success_metric)

        # 2. Output completeness (есть ли финальный output)
        has_output = bool(execution_result.final_output)
        output_keys = len(execution_result.final_output) if isinstance(execution_result.final_output, dict) else 0
        completeness = min(1.0, output_keys / 3.0) if has_output else 0.0  # ожидаем хотя бы 3 поля

        metrics.append(Metric(
            name="output_completeness",
            value=completeness,
            weight=0.3,
            direction="max",
            evaluator="structural",
            description="Финальный output содержит данные",
            automated=True,
        ))

        # 3. Cost efficiency (0 = дорого, 1 = бесплатно)
        cost = execution_result.total_cost_usd
        cost_score = max(0.0, 1.0 - cost / 0.50)  # $0.50 = 0 баллов
        metrics.append(Metric(
            name="cost_efficiency",
            value=cost_score,
            weight=0.15,
            direction="max",  # нормализовали — чем больше, тем лучше
            evaluator="from_logs",
            description=f"Cost ${cost:.4f}",
            automated=True,
        ))

        # 4. Latency efficiency
        latency = execution_result.total_duration_sec
        latency_score = max(0.0, 1.0 - latency / 300.0)  # 300 сек = 0 баллов
        metrics.append(Metric(
            name="latency_efficiency",
            value=latency_score,
            weight=0.15,
            direction="max",
            evaluator="from_logs",
            description=f"Latency {latency:.1f}s",
            automated=True,
        ))

        # Composite score
        composite = self._compute_composite(metrics)

        # Feedback
        feedback = self._generate_feedback(execution_result, metrics, composite)

        return EvaluationResult(
            run_id=execution_result.run_id,
            hypothesis_id=hypothesis_id,
            success=execution_result.success,
            metrics=metrics,
            composite_score=composite,
            feedback=feedback,
            metadata={
                "evaluated_at": time.time(),
                "config_used": self.config.get("metrics", []),
            },
        )

    def _compute_composite(self, metrics: list[Metric]) -> float:
        """Вычисляет composite score через weighted_sum."""
        total_weight = sum(m.weight for m in metrics)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(m.value * m.weight for m in metrics)
        return weighted_sum / total_weight

    def _generate_feedback(
        self,
        execution_result: ExecutionResult,
        metrics: list[Metric],
        composite: float,
    ) -> dict[str, Any]:
        """Генерирует actionable feedback для следующей итерации."""
        weaknesses: list[str] = []
        suggestions: list[str] = []

        if not execution_result.success:
            weaknesses.append("Pipeline завершился с ошибкой")
            # Анализируем, какие узлы упали
            for node_id, error in execution_result.errors.items():
                weaknesses.append(f"Узел '{node_id}' упал: {error[:200]}")
            suggestions.append("Проверить упавшие узлы и добавить retry/fallback")
            suggestions.append("Возможно, изменить промпты LLM-узлов")

        # Анализ метрик
        for m in metrics:
            if m.value < 0.5:
                weaknesses.append(f"Низкая метрика '{m.name}': {m.value:.2f}")
                if m.name == "cost_efficiency":
                    suggestions.append("Использовать более дешёвые модели (local:gemma-4-26b)")
                elif m.name == "latency_efficiency":
                    suggestions.append("Уменьшить количество узлов или использовать batching")
                elif m.name == "output_completeness":
                    suggestions.append("Добавить узлы для проверки и обогащения output")

        # Если всё хорошо
        if not weaknesses:
            suggestions.append("Pipeline работает хорошо. Можно использовать в production")

        return {
            "winner_hypothesis": execution_result.run_id,
            "why": f"Composite score: {composite:.3f}",
            "weaknesses": weaknesses,
            "suggestions_for_next_iteration": suggestions,
            "next_action_recommended": "iterate" if composite < 0.85 else "accept",
        }

    def evaluate_with_llm_judge(
        self,
        execution_result: ExecutionResult,
        task_description: str,
        hypothesis_id: str = "unknown",
    ) -> EvaluationResult:
        """
        Расширенная оценка с LLM-as-judge.

        Добавляет субъективные метрики (quality, relevance) через LLM.
        """
        # Сначала структурная оценка
        base_result = self.evaluate(execution_result, hypothesis_id)

        if not self.llm or not execution_result.success:
            return base_result

        # LLM-as-judge
        try:
            output_str = json.dumps(execution_result.final_output, ensure_ascii=False, default=str)[:2000]
            prompt = f"""Оцени результат выполнения pipeline.

Задача: {task_description}

Результат (финальный output):
{output_str}

Метрики лога:
- Успех: {execution_result.success}
- Стоимость: ${execution_result.total_cost_usd:.4f}
- Время: {execution_result.total_duration_sec:.1f}s
- Токенов: {execution_result.total_tokens}

Оцени по критериям (0-1):
1. quality: насколько результат качественный
2. relevance: насколько соответствует задаче
3. completeness: насколько полный

Верни JSON: {{"quality": 0.0, "relevance": 0.0, "completeness": 0.0, "reasoning": "..."}}
"""

            response: Response = self.llm.chat(
                messages=[Message(role="user", content=prompt)],
                model=self.judge_model,
                temperature=0.2,
                max_tokens=512,
                json_mode=True,
                timeout_sec=120,
            )

            judge_data = json.loads(response.content)

            # Добавляем метрики
            for name, weight in [("quality", 0.3), ("relevance", 0.2), ("completeness", 0.15)]:
                value = float(judge_data.get(name, 0.5))
                base_result.metrics.append(Metric(
                    name=name,
                    value=value,
                    weight=weight,
                    direction="max",
                    evaluator=f"llm_judge:{self.judge_model}",
                    description=judge_data.get("reasoning", "")[:200],
                    automated=False,
                ))

            # Пересчитываем composite
            base_result.composite_score = self._compute_composite(base_result.metrics)

        except Exception as e:
            base_result.metadata["llm_judge_error"] = str(e)

        return base_result

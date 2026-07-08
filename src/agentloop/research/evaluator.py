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
                    "weight": 0.25,
                    "threshold": 1.0,
                },
                {
                    "name": "output_completeness",
                    "evaluator": "structural",
                    "direction": "max",
                    "weight": 0.15,
                },
                {
                    "name": "output_content_quality",
                    "evaluator": "structural",
                    "direction": "max",
                    "weight": 0.35,
                    "threshold": 0.3,
                },
                {
                    "name": "cost_efficiency",
                    "evaluator": "from_logs",
                    "direction": "min",
                    "weight": 0.125,
                },
                {
                    "name": "latency_efficiency",
                    "evaluator": "from_logs",
                    "direction": "min",
                    "weight": 0.125,
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
            weight=0.25,
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
            weight=0.15,
            direction="max",
            evaluator="structural",
            description=f"Финальный output содержит данные ({output_keys} ключей)",
            automated=True,
        ))

        # 3. Output content quality — ПРОВЕРКА СОДЕРЖИМОГО
        # Без этой метрики pipeline мог вернуть {} и получить 0.89 score.
        content_quality, content_description = self._evaluate_content_quality(execution_result)
        metrics.append(Metric(
            name="output_content_quality",
            value=content_quality,
            weight=0.35,
            direction="max",
            evaluator="structural",
            description=content_description,
            automated=True,
            threshold=0.3,
            passed_threshold=content_quality >= 0.3,
        ))

        # 4. Cost efficiency (0 = дорого, 1 = бесплатно)
        cost = execution_result.total_cost_usd
        cost_score = max(0.0, 1.0 - cost / 0.50)  # $0.50 = 0 баллов
        metrics.append(Metric(
            name="cost_efficiency",
            value=cost_score,
            weight=0.125,
            direction="max",
            evaluator="from_logs",
            description=f"Cost ${cost:.4f}",
            automated=True,
        ))

        # 5. Latency efficiency
        latency = execution_result.total_duration_sec
        latency_score = max(0.0, 1.0 - latency / 300.0)  # 300 сек = 0 баллов
        metrics.append(Metric(
            name="latency_efficiency",
            value=latency_score,
            weight=0.125,
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

    def _evaluate_content_quality(self, execution_result: ExecutionResult) -> tuple[float, str]:
        """
        Проверяет содержимое финального output.

        Возвращает (score 0-1, описание).
        Шкала:
          0.0 — пустой output, {} или None
          0.2 — output есть, но значения пустые (""/[]/0)
          0.5 — output есть, значения короткие (< 50 char)
          0.8 — output есть, значения содержательные
          1.0 — output есть, несколько содержательных полей
        """
        output = execution_result.final_output

        if not output:
            return 0.0, "Output пустой"

        if not isinstance(output, dict):
            # Строка — проверяем длину
            text = str(output).strip()
            if not text or text in ("{}", "[]", "null", "None"):
                return 0.0, f"Output пустой: {text[:50]!r}"
            # Проверяем error-паттерн в строкке
            if self._looks_like_error(text):
                return 0.05, f"Output выглядит как error: {text[:100]!r}"
            if len(text) < 50:
                return 0.3, f"Output слишком короткий ({len(text)} символов)"
            return 0.7, f"Output содержательный ({len(text)} символов)"

        # dict
        if not output:
            return 0.0, "Output пустой dict {}"

        # Проверяем error-паттерн в dict
        # LLM в json_mode часто возвращает {"error": "...", "message": "..."}
        # когда не может выполнить задачу (нет входных данных, placeholders и т.д.)
        error_indicators = ("error", "Error", "ERROR")
        has_error_field = any(k in output for k in error_indicators)
        if has_error_field:
            error_text = ""
            for k in error_indicators:
                if k in output and isinstance(output[k], str):
                    error_text = output[k][:150]
                    break
            if "message" in output and isinstance(output["message"], str):
                error_text = error_text or output["message"][:150]
            return 0.05, f"Output содержит error: {error_text!r}"

        # Сканируем значения
        meaningful_fields = 0
        empty_fields = 0
        total_chars = 0

        for key, value in output.items():
            # Пропускаем служебные поля executor'а
            if key in ("input_tokens", "output_tokens", "cost_usd", "latency_ms", "exit_code", "stdout", "stderr", "files"):
                continue

            if value is None:
                empty_fields += 1
                continue

            if isinstance(value, str):
                if not value.strip():
                    empty_fields += 1
                elif value.strip() in ("{}", "[]"):
                    empty_fields += 1
                else:
                    # Проверяем, не является ли строка вложенным error-сообщением
                    if self._looks_like_error(value):
                        return 0.05, f"Output content выглядит как error: {value[:100]!r}"
                    meaningful_fields += 1
                    total_chars += len(value)

            elif isinstance(value, (list, dict)):
                if len(value) == 0:
                    empty_fields += 1
                else:
                    meaningful_fields += 1
                    total_chars += len(str(value))

            elif isinstance(value, (int, float, bool)):
                # Числа — нейтральные
                pass

        if meaningful_fields == 0:
            if empty_fields > 0:
                return 0.1, f"Output {len(output)} ключей, но все значения пустые"
            return 0.2, f"Output {len(output)} ключей, но нет содержательных значений"

        # Считаем score
        score = 0.3  # базовый за наличие meaningful полей
        score += min(0.4, meaningful_fields * 0.1)  # до +0.4 за количество
        score += min(0.3, total_chars / 1000.0 * 0.3)  # до +0.3 за объём

        return min(1.0, score), f"{meaningful_fields} содержательных полей, {total_chars} символов"

    @staticmethod
    def _looks_like_error(text: str) -> bool:
        """Детектит error-паттерн в строке.

        LLM в json_mode возвращает сообщения об ошибках когда:
        - Нет входных данных ('No text provided', 'No topic provided')
        - Неразрешённые плейсхолдеры ('Placeholders detected')
        - Не может выполнить задачу ('cannot generate', 'unable to')

        Такие output — не реальный результат, а жалоба модели.
        """
        if not text:
            return False
        text_lower = text.lower()
        error_patterns = [
            '"error"',
            "'error'",
            "no text provided",
            "no topic provided",
            "no input provided",
            "no content provided",
            "placeholders detected",
            "template with variables",
            "please provide",
            "cannot generate",
            "unable to generate",
            "unable to process",
            "i cannot",
            "i can't",
            "i am unable",
            "not enough information",
            "insufficient information",
            "missing required",
            "please paste",
            "please specify",
        ]
        return any(p in text_lower for p in error_patterns)

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
                weaknesses.append(f"Низкая метрика '{m.name}': {m.value:.2f} ({m.description})")
                if m.name == "cost_efficiency":
                    suggestions.append("Использовать более дешёвые модели (local:gemma-4-26b)")
                elif m.name == "latency_efficiency":
                    suggestions.append("Уменьшить количество узлов или использовать batching")
                elif m.name == "output_completeness":
                    suggestions.append("Добавить узлы для проверки и обогащения output")
                elif m.name == "output_content_quality":
                    suggestions.append(
                        "Output не содержит содержательных данных. "
                        "Возможные причины: LLM-узел без user_prompt_template, "
                        "модель вернула {} в json_mode, или pipeline не довёл данные до финального узла."
                    )
                    suggestions.append(
                        "Проверить DAG: каждый LLM-узел должен иметь user_prompt_template "
                        "с конкретным текстом задачи."
                    )

        # Если всё хорошо
        if not weaknesses:
            suggestions.append("Pipeline работает хорошо. Можно использовать в production")

        # Проверка минимальных порогов
        failed_thresholds = [m for m in metrics if m.threshold is not None and not m.passed_threshold]
        threshold_blocked = bool(failed_thresholds)

        return {
            "winner_hypothesis": execution_result.run_id,
            "why": f"Composite score: {composite:.3f}",
            "weaknesses": weaknesses,
            "suggestions_for_next_iteration": suggestions,
            "next_action_recommended": "iterate" if (composite < 0.85 or threshold_blocked) else "accept",
            "threshold_blocked": threshold_blocked,
            "failed_thresholds": [
                {"name": m.name, "threshold": m.threshold, "actual": m.value}
                for m in failed_thresholds
            ],
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

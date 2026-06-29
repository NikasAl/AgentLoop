"""
core:quick_evaluate — быстрая оценка качества без LLM.

Используется в refine-циклах для проверки метрик без дорогих judge-узлов.
"""


def main(input_data: dict) -> dict:
    """
    Вычисляет composite score по доступным метрикам.

    input_data:
        poem/result: dict с полями для оценки
        metrics: list of metric names to compute
        weights: dict {metric: weight}
    """
    item = input_data.get("poem") or input_data.get("result") or {}
    metrics = input_data.get("metrics", ["structural", "rhyme"])
    weights = input_data.get("weights", {})

    scores = {}
    total = 0.0
    total_weight = 0.0

    for metric in metrics:
        # Ищем значение метрики в item
        score = None
        for key in [metric, f"{metric}_score", f"{metric}_consistency"]:
            if key in item:
                try:
                    score = float(item[key])
                except (ValueError, TypeError):
                    score = 0.5
                break

        if score is None:
            score = 0.5  # default neutral

        scores[metric] = score
        weight = float(weights.get(metric, 1.0 / len(metrics)))
        total += score * weight
        total_weight += weight

    composite = total / total_weight if total_weight > 0 else 0.0

    return {
        "composite_score": composite,
        "scores": scores,
        "iteration": input_data.get("iteration", 0),
    }

"""
core:select_best_by_score — выбор лучшего элемента по метрике.

Используется в pipeline для выбора лучшего стиха/задачи/варианта.
"""


def main(input_data: dict) -> dict:
    """
    Выбирает элемент с max/min по заданной метрике.

    input_data:
        aggregated: dict с полем items (list of dicts)
        metric: имя поля-метрики в каждом item
        direction: "max" | "min"

    returns: dict с лучшим элементом
    """
    aggregated = input_data.get("aggregated", {})
    metric = input_data.get("metric", "composite_score")
    direction = input_data.get("direction", "max")

    items = aggregated.get("items") or aggregated.get("poems_with_scores") or aggregated.get("results", [])

    if not items:
        return {"error": "No items to select from"}

    def get_score(item: dict) -> float:
        if isinstance(item, dict):
            return float(item.get(metric, 0))
        return 0.0

    if direction == "max":
        best = max(items, key=get_score)
    else:
        best = min(items, key=get_score)

    return {
        "best": best,
        "best_score": get_score(best),
        "total_candidates": len(items),
    }

"""
core:json_merge — объединение JSON-данных из нескольких источников.

Используется в pipeline для агрегации результатов.
"""


def main(input_data: dict) -> dict:
    """
    Объединяет несколько dict в один.

    input_data: dict с произвольными ключами, значения — dicts
    returns: объединённый dict
    """
    result: dict = {}
    for key, value in input_data.items():
        if key.startswith("_"):
            continue  # служебные поля
        if isinstance(value, dict):
            result[key] = value
        elif isinstance(value, list):
            result[key] = value
        else:
            result[key] = value
    return result

"""
Provider Layer — единый интерфейс к 4 провайдерам LLM.

Принципы:
- Все провайдеры реализуют Protocol `Provider`
- Каждый вызов возвращает `Response` с метриками (tokens, cost, latency)
- Cost tracker логирует все вызовы в SQLite
- Кураторский список моделей в models.yaml
"""

from .base import (
    Capability,
    Message,
    Provider,
    ProviderError,
    Response,
    ModelInfo,
)
from .registry import (
    ALL_MODELS,
    find_model,
    get_provider,
    list_providers,
    models_by_provider,
    models_by_tier,
    PROVIDERS,
)

__all__ = [
    "ALL_MODELS",
    "Capability",
    "Message",
    "Provider",
    "ProviderError",
    "Response",
    "ModelInfo",
    "find_model",
    "get_provider",
    "list_providers",
    "models_by_provider",
    "models_by_tier",
    "PROVIDERS",
]

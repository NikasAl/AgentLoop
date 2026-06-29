"""
Кураторский список моделей + registry провайдеров.

Использование:
    from agentloop.providers import get_provider, list_providers

    local = get_provider("local", base_url="http://turbo:8080")
    response = local.chat([Message(role="user", content="Hello")], model="gemma-4-26b")

    human = get_provider("human")
    response = human.chat([Message(role="user", content="...")], model="browser")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .base import Capability, ModelInfo, Provider, ProviderError

# Load curated models from YAML
_MODELS_YAML = Path(__file__).parent.parent / "models.yaml"


def _load_models() -> list[ModelInfo]:
    if not _MODELS_YAML.exists():
        return []
    data = yaml.safe_load(_MODELS_YAML.read_text(encoding="utf-8"))
    models: list[ModelInfo] = []
    for m in data.get("models", []):
        models.append(
            ModelInfo(
                name=m["name"],
                provider=m["provider"],
                full_id=f"{m['provider']}:{m['name']}",
                tier=m.get("tier", 99),
                capabilities=[Capability(c) for c in m.get("capabilities", ["text"])],
                price_input_usd_per_1m=m.get("price_input_usd_per_1m", 0.0),
                price_output_usd_per_1m=m.get("price_output_usd_per_1m", 0.0),
                max_context=m.get("max_context", 32768),
                max_output=m.get("max_output", 8192),
                description=m.get("description", ""),
                rating=m.get("rating", 0.0),
                notes=m.get("notes", ""),
            )
        )
    return models


ALL_MODELS: list[ModelInfo] = _load_models()


def models_by_provider(provider_name: str) -> list[ModelInfo]:
    """Фильтрует модели по провайдеру."""
    return [m for m in ALL_MODELS if m.provider == provider_name]


def models_by_tier(tier: int) -> list[ModelInfo]:
    """Фильтрует модели по tier (0=local, 1=cheap, 2=mid, 3=strong, 999=human)."""
    return [m for m in ALL_MODELS if m.tier == tier]


def find_model(full_id: str) -> ModelInfo | None:
    """Находит модель по full_id типа 'local:gemma-4-26b'."""
    for m in ALL_MODELS:
        if m.full_id == full_id:
            return m
    return None


# ─── Provider factory ──────────────────────────────────────

_PROV_MODULES = {
    "local": ("local", "LocalProvider"),
    "openrouter": ("openrouter", "OpenRouterProvider"),
    "zai": ("zai", "ZAIProvider"),
    "human": ("human", "HumanProvider"),
}

# Cache instantiated providers
_INSTANCES: dict[str, Provider] = {}


def get_provider(name: str, **kwargs: Any) -> Provider:
    """
    Возвращает провайдер по имени.

    Args:
        name: "local" | "openrouter" | "zai" | "human"
        **kwargs: параметры конкретного провайдера
            - local: base_url (default: http://turbo:8080)
            - openrouter: api_key (default: $OPENROUTER_API_KEY)
            - zai: api_key (default: $ZAI_API_KEY)
            - human: editor (default: "subl"), clipboard_cmd (default: "xclip")

    Кеширует экземпляры (с одинаковыми kwargs) для переиспользования.
    """
    if name in _INSTANCES:
        return _INSTANCES[name]

    if name not in _PROV_MODULES:
        raise ProviderError(
            f"Unknown provider: {name}. Available: {list(_PROV_MODULES.keys())}",
            provider=name,
        )

    module_name, class_name = _PROV_MODULES[name]
    import importlib

    mod = importlib.import_module(f".{module_name}", package=__package__)
    cls = getattr(mod, class_name)

    instance = cls(**kwargs)
    _INSTANCES[name] = instance
    return instance


def list_providers() -> list[str]:
    """Возвращает имена всех доступных провайдеров."""
    return list(_PROV_MODULES.keys())


PROVIDERS = list_providers()

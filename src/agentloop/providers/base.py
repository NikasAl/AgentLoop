"""
Base types for Provider Layer.

Все провайдеры реализуют Protocol `Provider`.
Все вызовы возвращают `Response` с метриками.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, runtime_checkable


class Capability(str, Enum):
    """Возможности модели. Фильтруются при выборе модели под задачу."""

    TEXT = "text"
    VISION = "vision"
    TOOLS = "tools"
    JSON_MODE = "json_mode"
    REASONING = "reasoning"


@dataclass
class Message:
    """Сообщение в чате. Поддерживает text и image (для vision-моделей)."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    images: list[Path | str] | None = None  # пути к изображениям для vision

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.images:
            d["images"] = [str(p) for p in self.images]
        return d


@dataclass
class Response:
    """
    Ответ провайдера + метрики.

    Все поля кроме `content` опциональны для HumanProvider (он не знает токенов).
    """

    content: str
    provider: str
    model: str

    # Token metrics
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    # Cost (USD)
    cost_usd: float = 0.0

    # Latency
    latency_ms: int = 0

    # Human-specific
    human_time_sec: int = 0  # для HumanProvider

    # Raw response (для отладки)
    raw: dict[str, Any] | None = None

    # Error (если был retry/fallback)
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ModelInfo:
    """Информация о модели из кураторского списка."""

    name: str  # локальное имя, напр. "gemma-4-26b"
    provider: str  # "local" | "openrouter" | "zai" | "human"
    full_id: str  # "local:gemma-4-26b" — уникальный идентификатор

    tier: int  # 0=local, 1=cheap, 2=mid, 3=strong, 999=human
    capabilities: list[Capability] = field(default_factory=list)

    # Pricing (USD per 1M tokens)
    price_input_usd_per_1m: float = 0.0
    price_output_usd_per_1m: float = 0.0

    # Limits
    max_context: int = 32768
    max_output: int = 8192

    # Notes
    description: str = ""
    rating: float = 0.0  # 0-1, субъективная оценка
    notes: str = ""

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Примерная стоимость вызова."""
        return (
            self.price_input_usd_per_1m * input_tokens / 1_000_000
            + self.price_output_usd_per_1m * output_tokens / 1_000_000
        )


class ProviderError(Exception):
    """Ошибка провайдера (сеть, авторизация, rate limit и т.д.)."""

    def __init__(
        self,
        message: str,
        provider: str = "",
        status_code: int | None = None,
        retry_after_sec: int | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retry_after_sec = retry_after_sec
        self.retryable = status_code in {429, 500, 502, 503, 504} if status_code else False


@runtime_checkable
class Provider(Protocol):
    """
    Protocol, который реализуют все провайдеры.

    Главный метод — `chat`. Возвращает `Response` с метриками.
    """

    name: str  # "local" | "openrouter" | "zai" | "human"

    def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
        timeout_sec: int = 120,
        **kwargs: Any,
    ) -> Response:
        """
        Вызов LLM.

        Args:
            messages: список сообщений (system/user/assistant/tool)
            model: имя модели (без префикса провайдера)
            temperature: 0..1
            max_tokens: лимит на output
            json_mode: запросить JSON-ответ (если поддерживается)
            timeout_sec: таймаут на вызов

        Returns:
            Response с content и метриками (tokens, cost, latency)
        """
        ...

    def list_models(self) -> list[ModelInfo]:
        """Возвращает список доступных моделей у этого провайдера."""
        ...

    def health_check(self) -> bool:
        """Проверка доступности провайдера (без LLM-вызова)."""
        ...

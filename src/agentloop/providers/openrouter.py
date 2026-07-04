"""
OpenRouter provider — доступ к 300+ моделям через единый API.

Документация: https://openrouter.ai/docs
Цены: $0.10-15 за 1M токенов в зависимости от модели.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from .base import Capability, Message, ModelInfo, ProviderError, Response
from .registry import ALL_MODELS


class OpenRouterProvider:
    """
    Клиент к OpenRouter API.

    Требует OPENROUTER_API_KEY в env или в kwargs.
    """

    name = "openrouter"
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        referer: str | None = None,
        title: str = "AgentLoop",
        timeout_sec: int = 300,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ProviderError(
                "OPENROUTER_API_KEY not set. Pass api_key= or set env var.",
                provider="openrouter",
            )
        self.referer = referer or os.getenv("OPENROUTER_REFERER", "https://github.com/NikasAl/AgentLoop")
        self.title = title
        self.timeout_sec = timeout_sec
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout_sec,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": self.referer,
                    "X-Title": self.title,
                },
            )
        return self._client

    def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
        timeout_sec: int = 300,
        **kwargs: Any,
    ) -> Response:
        url = f"{self.BASE_URL}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [self._encode_message(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        start = time.time()
        try:
            r = self.client.post(url, json=payload, timeout=timeout_sec)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            retry_after = None
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 60))
            raise ProviderError(
                f"OpenRouter {e.response.status_code}: {e.response.text[:500]}",
                provider="openrouter",
                status_code=e.response.status_code,
                retry_after_sec=retry_after,
            ) from e
        except httpx.RequestError as e:
            raise ProviderError(f"OpenRouter connection error: {e}", provider="openrouter") from e

        latency_ms = int((time.time() - start) * 1000)
        data = r.json()

        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage", {})

        # Считаем cost по нашим ценам из models.yaml
        cost = self._compute_cost(model, usage)

        return Response(
            content=content,
            provider="openrouter",
            model=model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cache_read_tokens=usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
            cost_usd=cost,
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    def _encode_message(self, m: Message) -> dict[str, Any]:
        if m.images:
            import base64
            from pathlib import Path

            parts: list[dict[str, Any]] = [{"type": "text", "text": m.content}]
            for img_path in m.images:
                p = Path(img_path)
                if not p.exists():
                    raise ProviderError(f"Image not found: {p}", provider="openrouter")
                mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
                b64 = base64.b64encode(p.read_bytes()).decode()
                parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                )
            return {"role": m.role, "content": parts}
        return {"role": m.role, "content": m.content}

    def _compute_cost(self, model: str, usage: dict[str, Any]) -> float:
        """Считает стоимость по нашим ценам из models.yaml."""
        full_id = f"openrouter:{model}"
        for m in ALL_MODELS:
            if m.full_id == full_id:
                in_tok = usage.get("prompt_tokens", 0)
                out_tok = usage.get("completion_tokens", 0)
                cache_tok = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                # Cache tokens в 10x дешевле для Anthropic-моделей
                return (
                    m.price_input_usd_per_1m * (in_tok - cache_tok) / 1_000_000
                    + m.price_input_usd_per_1m * 0.1 * cache_tok / 1_000_000
                    + m.price_output_usd_per_1m * out_tok / 1_000_000
                )
        # Не нашли в catalog — возвращаем 0, логируем warning
        # В реальной системе здесь должен быть fallback на /api/v1/models
        return 0.0

    def list_models(self) -> list[ModelInfo]:
        """Возвращает модели из кураторского списка для OpenRouter."""
        return [m for m in ALL_MODELS if m.provider == "openrouter"]

    def fetch_remote_models(self) -> list[dict[str, Any]]:
        """
        Запрашивает полный список моделей с OpenRouter (для обновления models.yaml).
        Использовать периодически, не на каждый вызов.
        """
        try:
            r = self.client.get(f"{self.BASE_URL}/models", timeout=30)
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception as e:
            raise ProviderError(f"Failed to fetch OpenRouter models: {e}", provider="openrouter") from e

    def health_check(self) -> bool:
        """Проверка валидности API key."""
        try:
            r = self.client.get(f"{self.BASE_URL}/key", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

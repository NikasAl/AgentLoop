"""
Z.AI provider — доступ к GLM-моделям через официальный API.

Документация: https://z.ai/docs/api
Бесплатные модели: GLM-4.7 Flash (с retry-логикой, может быть недоступна).
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from .base import Capability, Message, ModelInfo, ProviderError, Response
from .registry import ALL_MODELS


class ZAIProvider:
    """
    Клиент к Z.AI API (GLM-модели).

    Требует ZAI_API_KEY в env или в kwargs.
    Бесплатные модели иногда возвращают 429 — нужен retry.
    """

    name = "zai"
    BASE_URL = "https://api.z.ai/api/paas/v4"

    def __init__(
        self,
        api_key: str | None = None,
        timeout_sec: int = 300,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("ZAI_API_KEY")
        if not self.api_key:
            raise ProviderError(
                "ZAI_API_KEY not set. Pass api_key= or set env var.",
                provider="zai",
            )
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout_sec,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
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

        # Z.AI может возвращать 429 для бесплатных моделей — retry
        last_err: ProviderError | None = None
        for attempt in range(self.max_retries):
            start = time.time()
            try:
                r = self.client.post(url, json=payload, timeout=timeout_sec)
                if r.status_code == 429:
                    retry_after = int(r.headers.get("Retry-After", 30 * (attempt + 1)))
                    last_err = ProviderError(
                        f"Z.AI rate limited (429), retry in {retry_after}s",
                        provider="zai",
                        status_code=429,
                        retry_after_sec=retry_after,
                    )
                    time.sleep(min(retry_after, 30))
                    continue
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise ProviderError(
                    f"Z.AI {e.response.status_code}: {e.response.text[:500]}",
                    provider="zai",
                    status_code=e.response.status_code,
                ) from e
            except httpx.RequestError as e:
                last_err = ProviderError(f"Z.AI connection error: {e}", provider="zai")
                time.sleep(2 * (attempt + 1))
                continue

            latency_ms = int((time.time() - start) * 1000)
            data = r.json()

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            cost = self._compute_cost(model, usage)

            return Response(
                content=content,
                provider="zai",
                model=model,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                cost_usd=cost,
                latency_ms=latency_ms,
                raw=data,
            )

        raise last_err or ProviderError("Z.AI failed after retries", provider="zai")

    def _encode_message(self, m: Message) -> dict[str, Any]:
        if m.images:
            import base64
            from pathlib import Path

            parts: list[dict[str, Any]] = [{"type": "text", "text": m.content}]
            for img_path in m.images:
                p = Path(img_path)
                if not p.exists():
                    raise ProviderError(f"Image not found: {p}", provider="zai")
                mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
                b64 = base64.b64encode(p.read_bytes()).decode()
                parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                )
            return {"role": m.role, "content": parts}
        return {"role": m.role, "content": m.content}

    def _compute_cost(self, model: str, usage: dict[str, Any]) -> float:
        full_id = f"zai:{model}"
        for m in ALL_MODELS:
            if m.full_id == full_id:
                in_tok = usage.get("prompt_tokens", 0)
                out_tok = usage.get("completion_tokens", 0)
                return m.cost(in_tok, out_tok)
        return 0.0

    def list_models(self) -> list[ModelInfo]:
        return [m for m in ALL_MODELS if m.provider == "zai"]

    def health_check(self) -> bool:
        try:
            # Простой health-check через список моделей
            r = self.client.get(f"{self.BASE_URL}/models", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

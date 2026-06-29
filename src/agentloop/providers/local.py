"""
Local provider — клиент к llama-server (OpenAI-совместимый протокол).

По умолчанию: http://turbo:8080 (gemma-4-26b на отдельном сервере).
Цена = 0 (бесплатно, локальная модель).
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from .base import Capability, Message, ModelInfo, ProviderError, Response


class LocalProvider:
    """
    Клиент к llama-server через OpenAI-совместимый /v1/chat/completions.

    Не использует API key (llama-server обычно без авторизации).
    """

    name = "local"

    def __init__(
        self,
        base_url: str | None = None,
        timeout_sec: int = 300,
    ):
        self.base_url = (base_url or os.getenv("LOCAL_LLM_URL", "http://turbo:8080")).rstrip("/")
        self.timeout_sec = timeout_sec
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_sec)
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
        url = f"{self.base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [self._encode_message(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        start = time.time()
        try:
            r = self.client.post(url, json=payload, timeout=timeout_sec)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"Local LLM error {e.response.status_code}: {e.response.text[:500]}",
                provider="local",
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            raise ProviderError(f"Local LLM connection error: {e}", provider="local") from e

        latency_ms = int((time.time() - start) * 1000)
        data = r.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return Response(
            content=content,
            provider="local",
            model=model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cost_usd=0.0,  # local = free
            latency_ms=latency_ms,
            raw=data,
        )

    def _encode_message(self, m: Message) -> dict[str, Any]:
        """Кодирует Message в формат OpenAI. Vision-изображения — base64 inline."""
        if m.images:
            # Multimodal: content is array of text + image_url parts
            import base64
            from pathlib import Path

            parts: list[dict[str, Any]] = [{"type": "text", "text": m.content}]
            for img_path in m.images:
                p = Path(img_path)
                if not p.exists():
                    raise ProviderError(f"Image not found: {p}", provider="local")
                mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
                b64 = base64.b64encode(p.read_bytes()).decode()
                parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                )
            return {"role": m.role, "content": parts}
        return {"role": m.role, "content": m.content}

    def list_models(self) -> list[ModelInfo]:
        """Запрашивает /v1/models у llama-server."""
        try:
            r = self.client.get(f"{self.base_url}/v1/models", timeout=10)
            r.raise_for_status()
            data = r.json()
            models: list[ModelInfo] = []
            for m in data.get("data", []):
                mid = m.get("id", "unknown")
                models.append(
                    ModelInfo(
                        name=mid,
                        provider="local",
                        full_id=f"local:{mid}",
                        tier=0,
                        capabilities=[Capability.TEXT, Capability.REASONING],
                        price_input_usd_per_1m=0.0,
                        price_output_usd_per_1m=0.0,
                        max_context=32768,
                        description="Local model via llama-server",
                    )
                )
            return models
        except Exception as e:
            raise ProviderError(f"Failed to list local models: {e}", provider="local") from e

    def health_check(self) -> bool:
        """Проверка доступности сервера."""
        try:
            r = self.client.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

"""
LLMNode — вызов LLM через Provider Layer.

Поддерживает:
- 4 провайдера: local, openrouter, zai, human
- vision (images)
- json_mode (structured output)
- iterate_over (цикл по коллекции)
- system_prompt_ref / user_prompt_template (с ссылками на prompts секцию DAG)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ...providers import Message, ProviderError, get_provider
from ...providers.base import Response
from ..state import PipelineState, VariableResolver
from .base import BaseNode, NodeResult


class LLMNode(BaseNode):
    """Узел вызова LLM."""

    def __init__(
        self,
        node_id: str,
        model: str,
        system_prompt: str = "",
        user_prompt_template: str = "",
        image_input: str | None = None,
        iterate_over: str | None = None,
        iterate_kind: str = "files",  # "files" | "seeds" | "collection" | "json_array"
        iterate_over_seeds: list[int] | None = None,
        iterate_param: str = "seed",
        output_schema: dict[str, Any] | None = None,
        json_mode: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        output_to_file: str | None = None,
        timeout_sec: int = 120,
        max_retries: int = 0,
        condition: str | None = None,
        human_provider_config: dict[str, Any] | None = None,
        # Reasoning-управление (для reasoning-моделей: gemma-4, qwen-3, deepseek-r1).
        # thinking_budget_tokens — отдельный потолок для chain-of-thought (llama-server).
        # reasoning_effort — "low"|"medium"|"high" (OpenAI/o1-style, OpenRouter).
        # auto_retry_on_empty — при пустом content/finish=length повторить с большим max_tokens.
        thinking_budget_tokens: int | None = None,
        reasoning_effort: str | None = None,
        auto_retry_on_empty: bool = True,
        **kwargs: Any,
    ):
        super().__init__(node_id, timeout_sec, max_retries, condition, **kwargs)
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.image_input = image_input
        self.iterate_over = iterate_over
        self.iterate_kind = iterate_kind
        self.iterate_over_seeds = iterate_over_seeds
        self.iterate_param = iterate_param
        self.output_schema = output_schema
        self.json_mode = json_mode
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.output_to_file = output_to_file
        self.human_provider_config = human_provider_config or {}
        self.thinking_budget_tokens = thinking_budget_tokens
        self.reasoning_effort = reasoning_effort
        self.auto_retry_on_empty = auto_retry_on_empty

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LLMNode":
        return cls(
            node_id=d["id"],
            model=d["model"],
            system_prompt=d.get("system_prompt", ""),
            user_prompt_template=d.get("user_prompt_template", ""),
            image_input=d.get("image_input"),
            iterate_over=d.get("iterate_over"),
            iterate_kind=d.get("iterate_kind", "files"),
            iterate_over_seeds=d.get("iterate_over_seeds"),
            iterate_param=d.get("iterate_param", "seed"),
            output_schema=d.get("output_schema") or d.get("output", {}).get("schema") if isinstance(d.get("output"), dict) else d.get("output_schema"),
            json_mode=d.get("json_mode", False),
            temperature=d.get("temperature", 0.7),
            max_tokens=d.get("max_tokens", 2048),
            output_to_file=d.get("output_to_file"),
            timeout_sec=d.get("timeout_sec", 120),
            max_retries=d.get("max_retries", 0),
            condition=d.get("condition"),
            human_provider_config=d.get("human_provider_config"),
            thinking_budget_tokens=d.get("thinking_budget_tokens"),
            reasoning_effort=d.get("reasoning_effort"),
            auto_retry_on_empty=d.get("auto_retry_on_empty", True),
            fallback_on_failure=d.get("fallback_on_failure", "error"),
            default_output=d.get("default_output"),
        )

    def _execute(self, state: PipelineState, resolver: VariableResolver) -> NodeResult:
        """Выполняет LLM-вызов. Если iterate_over — вызывает несколько раз."""
        # Парсим model в provider + name
        if ":" in self.model:
            provider_name, model_name = self.model.split(":", 1)
        else:
            provider_name = "local"
            model_name = self.model

        try:
            provider = get_provider(provider_name)
        except ProviderError as e:
            return NodeResult(
                node_id=self.node_id,
                success=False,
                error=f"Provider error: {e}",
            )

        # Если iterate_over — цикл
        if self.iterate_over:
            return self._execute_iterated(state, resolver, provider, model_name)

        # Одиночный вызов
        return self._call_once(state, resolver, provider, model_name, None)

    def _execute_iterated(
        self,
        state: PipelineState,
        resolver: VariableResolver,
        provider: Any,
        model_name: str,
    ) -> NodeResult:
        """Вызывает LLM несколько раз для каждого элемента коллекции."""
        items = self._collect_iteration_items(state, resolver)
        if items is None:
            return NodeResult(
                node_id=self.node_id,
                success=False,
                error=f"Failed to resolve iterate_over: {self.iterate_over}",
            )

        results: list[dict[str, Any]] = []
        total_tokens_in = 0
        total_tokens_out = 0
        total_cost = 0.0

        for i, item in enumerate(items):
            result = self._call_once(state, resolver, provider, model_name, item)
            if not result.success:
                return result
            results.append(result.output)
            total_tokens_in += result.metadata.get("input_tokens", 0)
            total_tokens_out += result.metadata.get("output_tokens", 0)
            total_cost += result.metadata.get("cost_usd", 0.0)

        # Объединяем результаты
        # Если все результаты имеют поле "problems" — объединяем списки
        merged: dict[str, Any] = {"results": results, "count": len(results)}
        if results and isinstance(results[0], dict):
            for key in results[0]:
                if isinstance(results[0][key], list):
                    merged[key] = []
                    for r in results:
                        if isinstance(r.get(key), list):
                            merged[key].extend(r[key])

        merged["total_tokens_in"] = total_tokens_in
        merged["total_tokens_out"] = total_tokens_out
        merged["total_cost_usd"] = total_cost

        return NodeResult(
            node_id=self.node_id,
            success=True,
            output=merged,
            metadata={"iterations": len(items), "total_cost_usd": total_cost},
        )

    def _collect_iteration_items(self, state: PipelineState, resolver: VariableResolver) -> list[Any] | None:
        """Собирает элементы для итерации в зависимости от iterate_kind."""
        if self.iterate_kind == "seeds":
            return self.iterate_over_seeds or [1, 2, 3]

        # Разрешаем переменную
        items_ref = resolver.resolve(self.iterate_over)
        if isinstance(items_ref, str) and items_ref == self.iterate_over:
            return None  # не разрешено

        if self.iterate_kind == "files":
            if isinstance(items_ref, list):
                return items_ref
            if isinstance(items_ref, str):
                import glob
                return sorted(glob.glob(items_ref))

        elif self.iterate_kind == "collection":
            if isinstance(items_ref, list):
                return items_ref
            if isinstance(items_ref, dict) and "poems" in items_ref:
                return items_ref["poems"]

        elif self.iterate_kind == "json_array":
            if isinstance(items_ref, list):
                return items_ref

        return None

    def _call_once(
        self,
        state: PipelineState,
        resolver: VariableResolver,
        provider: Any,
        model_name: str,
        iteration_item: Any,
    ) -> NodeResult:
        """Один LLM-вызов с auto-retry на reasoning-перерасход.

        Reasoning-модели (gemma-4, qwen-3, deepseek-r1) иногда тратят весь токен-бюджет
        на chain-of-thought, оставляя content пустым (finish_reason=length). В этом случае
        повторяем вызов с увеличенным max_tokens (до 2x, не более 32768).
        """
        # Готовим user prompt
        user_prompt = resolver.resolve(self.user_prompt_template)
        if iteration_item is not None and self.iterate_kind == "seeds":
            seed = iteration_item
            user_prompt = user_prompt.replace("{seed}", str(seed))

        # Готовим messages
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))

        # Images для vision
        images: list[Path | str] = []
        if self.image_input:
            img_ref = resolver.resolve(self.image_input)
            if isinstance(img_ref, list):
                images = [Path(p) for p in img_ref]
            elif isinstance(img_ref, str):
                import glob
                images = [Path(p) for p in glob.glob(img_ref)]

        messages.append(Message(role="user", content=user_prompt, images=images or None))

        # Пробуем с исходным max_tokens, при необходимости — retry с увеличенным
        current_max_tokens = self.max_tokens
        for attempt in range(3 if self.auto_retry_on_empty else 1):
            response = self._do_provider_call(
                provider, model_name, messages, current_max_tokens,
            )

            # Если content непустой или finish_reason не length — успех
            content = response.content or ""
            if content.strip() and response.finish_reason != "length":
                return self._build_result(response, resolver)

            # Пустой content при finish_reason=length — reasoning-перерасход
            if not self.auto_retry_on_empty or response.finish_reason != "length":
                # Не retry-ситуация — возвращаем как есть (пустой content, но не ошибка)
                return self._build_result(response, resolver)

            # Retry с увеличенным max_tokens
            current_max_tokens = min(current_max_tokens * 2, 32768)

        # Все retry-попытки исчерпаны — возвращаем последний результат
        return self._build_result(response, resolver)

    def _do_provider_call(
        self,
        provider: Any,
        model_name: str,
        messages: list[Message],
        max_tokens: int,
    ) -> Response:
        """Выполняет один HTTP-вызов к провайдеру с пробросом reasoning-параметров."""
        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "json_mode": self.json_mode,
            "timeout_sec": self.timeout_sec,
        }
        # Reasoning-параметры (для reasoning-моделей)
        if self.thinking_budget_tokens is not None:
            kwargs["thinking_budget_tokens"] = self.thinking_budget_tokens
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        # Human provider особый
        if provider.name == "human":
            kwargs["node_id"] = self.node_id
            kwargs["reason"] = self.human_provider_config.get("reason", "LLM call")
            kwargs["model"] = model_name

        try:
            return provider.chat(messages=messages, model=model_name, **kwargs)
        except ProviderError as e:
            # Оборачиваем в Response с ошибкой, чтобы retry-логика могла отличить
            return Response(
                content="",
                provider=provider.name,
                model=model_name,
                finish_reason="error",
                error=str(e),
            )

    def _build_result(
        self,
        response: Response,
        resolver: VariableResolver,
    ) -> NodeResult:
        """Строит NodeResult из Response (общий для первой попытки и retry)."""
        content = response.content or ""
        output: dict[str, Any] = {"content": content}

        # Если json_mode или в output_schema — пробуем распарсить
        if self.json_mode or self.output_schema:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    output.update(parsed)
                else:
                    output["parsed"] = parsed
            except json.JSONDecodeError:
                json_str = self._extract_json(content)
                if json_str:
                    try:
                        output.update(json.loads(json_str))
                    except json.JSONDecodeError:
                        output["json_parse_error"] = "Failed to parse LLM response as JSON"

        # Метрики
        output["input_tokens"] = response.input_tokens
        output["output_tokens"] = response.output_tokens
        output["cost_usd"] = response.cost_usd
        output["latency_ms"] = response.latency_ms

        # Сохраняем в файл если нужно
        if self.output_to_file:
            out_path = resolver.resolve(self.output_to_file)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(
                json.dumps(output, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            output["output_file"] = out_path

        return NodeResult(
            node_id=self.node_id,
            success=True,
            output=output,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "latency_ms": response.latency_ms,
                "model": response.model,
                "provider": response.provider,
                "finish_reason": response.finish_reason,
            },
        )

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Пытается найти JSON в тексте (между { } или [ ])."""
        # Ищем самый внешний { ... }
        start = text.find("{")
        if start == -1:
            start = text.find("[")
            if start == -1:
                return None
            end_char = "]"
        else:
            end_char = "}"

        # Идём от конца, ищем последнюю закрывающую скобку
        end = text.rfind(end_char)
        if end == -1 or end <= start:
            return None

        return text[start : end + 1]

"""Tests for Provider Layer."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentloop.providers import (
    Capability,
    Message,
    ProviderError,
    Response,
    get_provider,
    list_providers,
    find_model,
    ALL_MODELS,
)
from agentloop.providers.base import ModelInfo


# ─── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def sample_messages():
    return [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello, world!"),
    ]


# ─── Base types ────────────────────────────────────────────


class TestMessage:
    def test_text_message(self):
        m = Message(role="user", content="hello")
        d = m.to_dict()
        assert d == {"role": "user", "content": "hello"}
        assert d.get("images") is None

    def test_image_message(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"fake-png")
        m = Message(role="user", content="describe", images=[img])
        d = m.to_dict()
        assert d["images"] == [str(img)]


class TestModelInfo:
    def test_cost_zero_for_local(self):
        m = ModelInfo(
            name="gemma-4-26b",
            provider="local",
            full_id="local:gemma-4-26b",
            tier=0,
        )
        assert m.cost(1000, 500) == 0.0

    def test_cost_paid_model(self):
        m = ModelInfo(
            name="gpt-4",
            provider="openrouter",
            full_id="openrouter:gpt-4",
            tier=3,
            price_input_usd_per_1m=10.0,
            price_output_usd_per_1m=30.0,
        )
        # 1000 input + 500 output = 0.01 + 0.015 = 0.025
        assert m.cost(1000, 500) == pytest.approx(0.025)


# ─── Registry ──────────────────────────────────────────────


class TestRegistry:
    def test_list_providers(self):
        providers = list_providers()
        assert "local" in providers
        assert "openrouter" in providers
        assert "zai" in providers
        assert "human" in providers

    def test_get_unknown_provider(self):
        with pytest.raises(ProviderError, match="Unknown provider"):
            get_provider("nonexistent")

    def test_find_model_local(self):
        m = find_model("local:gemma-4-26b")
        assert m is not None
        assert m.tier == 0
        assert Capability.VISION in m.capabilities

    def test_find_model_missing(self):
        assert find_model("local:nonexistent") is None

    def test_curated_models_loaded(self):
        assert len(ALL_MODELS) > 0
        providers = {m.provider for m in ALL_MODELS}
        assert "local" in providers
        assert "openrouter" in providers
        assert "zai" in providers
        assert "human" in providers

    def test_tiers_distribution(self):
        tiers = {m.tier for m in ALL_MODELS}
        assert 0 in tiers  # local
        assert 999 in tiers  # human


# ─── Local provider ────────────────────────────────────────


class TestLocalProvider:
    def test_init_default_url(self):
        from agentloop.providers.local import LocalProvider

        p = LocalProvider()
        assert p.base_url == "http://turbo:8080"
        assert p.name == "local"

    def test_init_custom_url(self):
        from agentloop.providers.local import LocalProvider

        p = LocalProvider(base_url="http://localhost:1234")
        assert p.base_url == "http://localhost:1234"

    def test_init_from_env(self, monkeypatch):
        from agentloop.providers.local import LocalProvider

        monkeypatch.setenv("LOCAL_LLM_URL", "http://custom:9999")
        p = LocalProvider()
        assert p.base_url == "http://custom:9999"

    def test_health_check_unreachable(self):
        from agentloop.providers.local import LocalProvider

        p = LocalProvider(base_url="http://127.0.0.1:1")  # unreachable port
        assert p.health_check() is False


# ─── OpenRouter provider ───────────────────────────────────


class TestOpenRouterProvider:
    def test_init_requires_api_key(self, monkeypatch):
        from agentloop.providers.openrouter import OpenRouterProvider

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ProviderError, match="OPENROUTER_API_KEY"):
            OpenRouterProvider()

    def test_init_with_api_key(self):
        from agentloop.providers.openrouter import OpenRouterProvider

        p = OpenRouterProvider(api_key="test-key")
        assert p.api_key == "test-key"
        assert p.name == "openrouter"

    def test_cost_computation(self):
        from agentloop.providers.openrouter import OpenRouterProvider

        p = OpenRouterProvider(api_key="test-key")
        # Use gemini-3.1-flash-lite from models.yaml: $0.30/$2.50 per 1M
        cost = p._compute_cost(
            "google/gemini-3.1-flash-lite",
            {"prompt_tokens": 1000, "completion_tokens": 500},
        )
        # Expected: 0.30 * 1000/1M + 2.50 * 500/1M = 0.0003 + 0.00125 = 0.00155
        assert cost == pytest.approx(0.00155, rel=0.01)


# ─── Z.AI provider ────────────────────────────────────────


class TestZAIProvider:
    def test_init_requires_api_key(self, monkeypatch):
        from agentloop.providers.zai import ZAIProvider

        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        with pytest.raises(ProviderError, match="ZAI_API_KEY"):
            ZAIProvider()

    def test_init_with_api_key(self):
        from agentloop.providers.zai import ZAIProvider

        p = ZAIProvider(api_key="test-key")
        assert p.api_key == "test-key"
        assert p.name == "zai"
        assert p.max_retries == 3


# ─── Human provider ───────────────────────────────────────


class TestHumanProvider:
    def test_init_defaults(self):
        from agentloop.providers.human import HumanProvider

        p = HumanProvider()
        assert p.name == "human"
        assert p.timeout_min == 30

    def test_list_models(self):
        from agentloop.providers.human import HumanProvider

        p = HumanProvider()
        models = p.list_models()
        assert len(models) == 2  # browser, self
        assert models[0].provider == "human"
        assert models[0].tier == 999

    def test_format_prompt(self):
        from agentloop.providers.human import HumanProvider

        p = HumanProvider()
        msgs = [
            Message(role="system", content="Be brief."),
            Message(role="user", content="Hello"),
        ]
        text = p._format_prompt(msgs)
        assert "### SYSTEM" in text
        assert "Be brief." in text
        assert "### USER" in text
        assert "Hello" in text

    def test_extract_response_normal(self):
        from agentloop.providers.human import HumanProvider

        p = HumanProvider()
        content = """# header
# more header

# ─── PASTE RESPONSE BELOW THIS LINE ───────────────────────

Hello, this is my response.
"""
        result = p._extract_response(content)
        assert "Hello, this is my response." in result

    def test_extract_response_empty_returns_skip(self):
        from agentloop.providers.human import HumanProvider

        p = HumanProvider()
        content = """# header only, no response"""
        result = p._extract_response(content)
        assert result == "SKIP"


# ─── Cost Tracker ─────────────────────────────────────────


class TestCostTracker:
    @pytest.fixture
    def tracker(self, tmp_path):
        from agentloop.cost_tracker import CostTracker

        return CostTracker(tmp_path / "test_usage.sqlite")

    def test_log_and_summary(self, tracker):
        tracker.log(
            task_id="task1",
            provider="local",
            model="gemma-4-26b",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.0,
        )
        tracker.log(
            task_id="task1",
            provider="openrouter",
            model="gemini-flash",
            input_tokens=500,
            output_tokens=100,
            cost_usd=0.001,
        )

        s = tracker.summary(task_id="task1")
        assert s.total_calls == 2
        assert s.total_tokens_in == 600
        assert s.total_tokens_out == 300
        assert s.total_cost_usd == pytest.approx(0.001)
        assert "local" in s.by_provider
        assert "openrouter" in s.by_provider
        assert s.by_provider["openrouter"]["cost_usd"] == pytest.approx(0.001)

    def test_budget_check(self, tracker):
        tracker.log(
            task_id="task1",
            provider="openrouter",
            model="gemini-flash",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.005,
        )

        spent, budget, exceeded = tracker.budget_check("task1", budget_usd=0.01)
        assert spent == pytest.approx(0.005)
        assert budget == 0.01
        assert not exceeded

        spent, budget, exceeded = tracker.budget_check("task1", budget_usd=0.001)
        assert exceeded

    def test_recent(self, tracker):
        for i in range(5):
            tracker.log(
                task_id="task1",
                provider="local",
                model="gemma-4-26b",
                input_tokens=i * 100,
                output_tokens=0,
            )
        recent = tracker.recent(limit=3, task_id="task1")
        assert len(recent) == 3
        # Most recent first (descending by timestamp)
        assert recent[0]["input_tokens"] == 400

    def test_filter_by_provider(self, tracker):
        tracker.log(
            task_id="task1",
            provider="local",
            model="gemma",
            input_tokens=100,
        )
        tracker.log(
            task_id="task1",
            provider="openrouter",
            model="gemini",
            input_tokens=200,
        )

        s = tracker.summary(provider="local")
        assert s.total_calls == 1
        assert s.total_tokens_in == 100

        s = tracker.summary(provider="openrouter")
        assert s.total_calls == 1
        assert s.total_tokens_in == 200

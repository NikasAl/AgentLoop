# AgentLoop

Adaptive pipeline system with hypothesis-driven pipeline optimization.

## Status

🚧 **Early development** — Provider Layer (Day 1-2 of MVP plan).

## Goals

- Universal adaptive pipeline system (not domain-specific)
- Multiple LLM providers: local (llama-server), OpenRouter, Z.AI, Human-as-API (via subl)
- Hypothesis generation → DAG building → execution → evaluation → distillation
- Token/cost tracking with per-task budgets
- Skill Library for reusable pipelines
- Research / Production mode separation

## Architecture overview

See `design/` directory for detailed format specifications and worked examples
across two domains (math extraction from PDF, poetry generation).

### Layer 1 — Core Tools (always in Builder's context)

| Tool | Description |
|---|---|
| `bash_run` | Shell command in sandbox with timeout |
| `python_run` | Python script (core or custom via Steward) |
| `llm_call` | LLM via Provider Layer (local/openrouter/zai/human) |
| `wait_human` | Human input via subl + clipboard |
| `web_search` | Semantic web search |
| `web_fetch` | Fetch URL with parsing (HTML→markdown, PDF→text) |
| `file_op` | First-class file operations (read/write/append/list/move/copy) |

### Provider Tiers

| Tier | Use case | Examples |
|---|---|---|
| 0 — Local | Bulk work, free | `local:gemma-4-26b` (llama-server) |
| 1 — Cheap API | Routine meta-agents | `gemini-3.1-flash-lite`, `glm-4.7-flash`, `mistral-small` |
| 2 — Mid API | Hypothesis gen, builder, judge | `gemini-3.1-flash`, `glm-5-turbo`, `deepseek-v3` |
| 3 — Strong API | Distillation teacher | `gemini-3.1-pro`, `glm-5.2`, `deepseek-r1` |
| ∞ — Human | Debug, expensive teacher | `human:browser` (subl + clipboard) |

## Project structure

```
agentloop/
├── design/             # Format specifications and worked examples
├── src/
│   └── agentloop/
│       ├── providers/  # Provider Layer (4 providers)
│       ├── cost_tracker.py
│       └── models.yaml # Curated model list
└── tests/
```

## Current module: Provider Layer

### Usage

```python
from agentloop.providers import get_provider
from agentloop.providers.base import Message, Response

provider = get_provider("local")  # or "openrouter", "zai", "human"
response = provider.chat(
    messages=[Message(role="user", content="Hello")],
    model="gemma-4-26b",
)
print(response.content)
print(f"Cost: ${response.cost_usd:.6f}")
print(f"Tokens: {response.input_tokens} in / {response.output_tokens} out")
```

### Providers

- **local** — Connects to llama-server (e.g., `http://turbo:8080`)
- **openrouter** — OpenRouter API with curated models
- **zai** — Z.AI API (GLM models)
- **human** — Opens subl, copies prompt to clipboard, waits for user response

## License

MIT

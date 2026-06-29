# AgentLoop

Адаптивная система пайплайнов с гипотезо-ориентированной оптимизацией.

## Статус

🚧 **Ранняя разработка** — Provider Layer + Tool Catalog (День 1-3 MVP плана).

## Цели

- Универсальная адаптивная система пайплайнов (не привязана к домену)
- Несколько LLM-провайдеров: локальный (llama-server), OpenRouter, Z.AI, Human-as-API (через subl)
- Генерация гипотез → построение DAG → выполнение → оценка → дистилляция
- Учёт токенов и стоимости с бюджетами per-task
- Skill Library для переиспользуемых пайплайнов
- Разделение режимов Research / Production

## Обзор архитектуры

См. директорию `design/` — подробные спецификации форматов и проработанные примеры
на двух доменах (извлечение задач из PDF, написание стихов).

### Layer 1 — Базовые инструменты (всегда в контексте Builder'а)

| Инструмент | Описание |
|---|---|
| `bash_run` | Shell-команда в sandbox с timeout |
| `python_run` | Python-скрипт (core или custom через Steward) |
| `llm_call` | LLM через Provider Layer (local/openrouter/zai/human) |
| `wait_human` | Человеческий ввод через subl + clipboard |
| `web_search` | Семантический веб-поиск |
| `web_fetch` | Загрузка URL с парсингом (HTML→markdown, PDF→text) |
| `file_op` | Файловые операции как first-class (read/write/append/list/move/copy) |

### Уровни провайдеров

| Уровень | Назначение | Примеры |
|---|---|---|
| 0 — Локальный | Основная работа, бесплатно | `local:gemma-4-26b` (llama-server) |
| 1 — Дешёвый API | Рутина и meta-агенты | `gemini-3.1-flash-lite`, `glm-4.7-flash`, `mistral-small` |
| 2 — Средний API | Hypothesis gen, builder, judge | `gemini-3.1-flash`, `glm-5-turbo`, `deepseek-v3` |
| 3 — Сильный API | Дистилляция-учитель | `gemini-3.1-pro`, `glm-5.2`, `deepseek-r1` |
| ∞ — Человек | Отладка, дорогой учитель | `human:browser` (subl + clipboard) |

### Слои инструментов

| Слой | Что содержит | Как пополняется |
|---|---|---|
| Layer 1 | 7 базовых примитивов | Захардкожено в коде |
| Layer 2 | Системные утилиты (pdftoppm, tesseract, ffmpeg, ...) | Сканирование PATH + pip list при старте |
| Layer 3 | Custom Python-инструменты | Создаётся Builder'ом через Steward с safety-check |

Steward — гибрид function + agent:
- **Function** (без LLM) — ищет по каталогу, координирует установку пакетов
- **Agent** (LLM) — включается если function не нашла, предлагает composite-решения

## Структура проекта

```
agentloop/
├── design/             # Спецификации форматов и примеры
├── src/
│   └── agentloop/
│       ├── providers/  # Provider Layer (4 провайдера)
│       ├── tools/      # Tool Catalog + Steward
│       ├── cost_tracker.py
│       └── models.yaml # Кураторский список моделей
└── tests/
```

## Текущий модуль: Provider Layer + Tool Catalog

### Использование провайдеров

```python
from agentloop.providers import get_provider, Message

# Локальная модель (бесплатно)
local = get_provider("local")  # → http://turbo:8080
resp = local.chat([Message(role="user", content="Привет")], model="gemma-4-26b")

# OpenRouter (с API key в env)
or_p = get_provider("openrouter")
resp = or_p.chat([Message(role="user", content="Hello")], model="google/gemini-3.1-flash")

# Human-as-API (для debug и дистилляции)
human = get_provider("human")
resp = human.chat(
    [Message(role="user", content="...")],
    model="browser",
    node_id="distill_teacher",
    reason="Заполни ответ от Gemini Pro",
)
```

### Использование Tool Catalog

```python
from agentloop.tools import ToolCatalog, Steward

catalog = ToolCatalog()
catalog.scan_system()  # сканирует PATH и pip list, заполняет Layer 2

# Поиск инструмента
result = catalog.search("извлечение текста из PDF")
for tool in result.found:
    print(f"{tool.name}: {tool.description}")

# Steward — с LLM-агентом для сложных запросов
steward = Steward(catalog=catalog, llm_provider=local)
result = steward.search("нормализовать LaTeX: \\frac → \\cfrac, проверка скобок")
# если function не нашёл — steward-agent предложит custom Python
```

### Провайдеры

- **local** — подключение к llama-server (например, `http://turbo:8080`)
- **openrouter** — OpenRouter API с курируемым списком моделей
- **zai** — Z.AI API (GLM-модели) с retry-логикой для 429
- **human** — открывает subl, копирует промпт в буфер обмена, ждёт ответ

## Особенности OS

Проект разрабатывается и тестируется на **Arch Linux**. При сканировании Layer 2
учитываются особенности:
- `pacman` как пакетный менеджер (а не `apt`)
- Имена пакетов могут отличаться от Debian/Ubuntu (например, `poppler` вместо `poppler-utils`)
- AUR поддерживается через `yay` (если установлен)
- `python-pip` для Python-пакетов

Для других дистрибутивов — в `known_tools.yaml` указаны альтернативные имена пакетов.

## Установка

```bash
git clone https://github.com/NikasAl/AgentLoop.git
cd AgentLoop
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Запуск тестов
pytest tests/ -v
```

## Переменные окружения

```bash
# API ключи (опционально для соответствующих провайдеров)
export OPENROUTER_API_KEY="sk-or-..."
export ZAI_API_KEY="..."

# URL локального llama-server (по умолчанию http://turbo:8080)
export LOCAL_LLM_URL="http://turbo:8080"

# Редактор для HumanProvider (по умолчанию subl)
export EDITOR="subl"
```

## Лицензия

MIT

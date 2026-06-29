# Design Examples: Adaptive Pipeline System

Демонстрация форматов на конкретных задачах. Цель — проверить, что форматы работают на **разных доменах**, не только на extraction.

## Сценарии

### Сценарий 1: Extraction (Сканави) — baseline проверка форматов

Research-режим за 2 итерации:
1. Гипотеза h1 (single-pass с Gemma 4) → baseline score 0.875
2. Дистилляция промпта через human-as-API → 0.962
3. Skill сохранён: `scanavi_extract_v1`

### Сценарий 2: Poetry (стихи) — проверка на субъективной задаче

Максимально отличается от extraction:
- Нет правильного ответа
- Множественные валидные выходы (10 стихов)
- Субъективные метрики (emotional_impact, originality)
- Итеративное улучшение, не исправление
- Human-as-API как judge

## Файлы

### Сценарий 1 (Extraction)

| Файл | Что демонстрирует | Когда возникает |
|---|---|---|
| `1_hypothesis.json` | Формат гипотез от Hypothesis Generator | Перед research-итерацией |
| `2_dag.json` | DAG v0.2 от Builder'а (с loop, file, gate, prerequisites) | После выбора гипотезы |
| `3_execution_log.json` | Лог выполнения + evaluator report | После прогона на sample |
| `4_skill.yaml` | Сохранённый навык в Skill Library | После успешной итерации |
| `5_human_as_api_example.md` | Subl-файл для human-as-API | При вызове teacher модели |

### Сценарий 2 (Poetry)

| Файл | Что демонстрирует | Когда возникает |
|---|---|---|
| `7_poem_hypothesis.json` | Гипотезы для написания стихов | Перед research-итерацией |
| `8_poem_dag.json` | DAG с collection outputs, llm_judge, parallel-refine | После выбора гипотезы |
| `9_poem_execution_log.json` | Лог с субъективными метриками | После прогона |

### Общие

| Файл | Что демонстрирует |
|---|---|
| `6_evaluator_config.yaml` | Примеры evaluator config для 4 разных доменов |
| `README.md` | Этот файл |

## Layer 1: Core Tools (всегда в контексте Builder'а)

Builder знает 7 примитивов + знание о Steward:

| Tool | Описание | Пример |
|---|---|---|
| `bash_run` | Выполнить shell-команду в sandbox | `pdftoppm`, `pdflatex`, `pipelines` |
| `python_run` | Выполнить Python-скрипт (core или custom) | `latex_validator_v1.py` |
| `llm_call` | Вызов LLM через Provider Layer | local:gemma-4-26b, openrouter:gemini-3.1-flash, human:browser |
| `wait_human` | Запросить человеческий ввод (subl + clipboard) | Дистилляция, judge, gate approval |
| `web_search` | Семантический веб-поиск | DuckDuckGo, Searx, кастомный |
| `web_fetch` | Fetch URL с парсингом (HTML→text, PDF→text) | `web_fetch(url, format="markdown")` |
| `file_op` | First-class файловые операции (read/write/append/list/move/copy) | `file_op(write, path, content)` |

**Принцип:** всё остальное — производное. Builder не получает `pdftotext` как отдельный инструмент, он использует `bash_run` с командой `pdftotext`. Если не знает команду — запрашивает Steward.

## Node Types в DAG v0.2

| Type | Описание |
|---|---|
| `bash` | Выполнение shell-команды в sandbox с timeout |
| `llm` | Вызов LLM через Provider Layer |
| `python` | Выполнение Python-скрипта (core или custom) |
| `file` | First-class файловая операция (read/write/append/list) |
| `loop` | Цикл над sub-graph: повторяет body пока exit_condition не true |
| `gate` | Промежуточная точка контроля (human_approval / quality_check / budget_check) |

## Output Kinds

| Kind | Описание |
|---|---|
| `files` | Список файлов по glob-паттерну |
| `json` | Структурированный JSON-объект по schema |
| `text` | Простой текст (обычно stdout для bash) |
| `file_ref` | Ссылка на конкретный файл (для file-узлов) |
| `media` | Медиа-файл с метаданными (mime, duration, resolution) |
| `collection` | Коллекция артефактов (множественные валидные выходы) |

## Condition Language v1

Простой язык для `condition` и `exit_condition` полей в узлах:

- Ссылки: `{node_id.output.field}` (dot notation)
- Фильтры: `|length`, `|is_empty`, `|contains:'substring'`
- Операторы: `==`, `!=`, `>`, `>=`, `<`, `<=`, `&&`, `||`
- Типы: `boolean`, `integer`, `string`, `null`

Примеры:
```
{validate_latex.output.has_invalid} == false
{extract_problems.output.problems|length} >= 8
{run_tests.output.all_passed} == true && {run_tests.output.failures|length} < 3
{emotional_impact_score} >= 0.7 || {human_override} == true
```

## Что проверять

### После прохождения по Extraction (сценарий 1)
- Понятны ли поля без дополнительных объяснений?
- Можно ли по `dag.json` написать executor?
- Достаточно ли `execution_log.json` для отладки?
- Можно ли по `skill.yaml` запустить pipeline на новой машине?

### После прохождения по Poetry (сценарий 2)
- Работают ли форматы на задаче без "правильного ответа"?
- Субъективные метрики (llm_judge через human:browser) выглядят ли естественно?
- Collection outputs (10 стихов) не ломают ли DAG?
- Loop над refine-циклом работает ли?

### Общие вопросы
- Где форматы избыточны?
- Где недостаточны?
- Какие узлы/типы пришлось бы добавить для других доменов (софт, видео, ИИ-модели)?

## Репозиторий

Код реализации будет в: https://github.com/NikasAl/AgentLoop

Эти design-файлы — спецификации для написания кода. При коммите в репозиторий они пойдут в `design/` директорию как референсы.

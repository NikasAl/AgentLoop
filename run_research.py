#!/usr/bin/env python3
"""
Запуск исследовательского агента AgentLoop.

Использование:
    act_env_general && python run_research.py "Сгенерировать 3 варианта приветствия"
    act_env_general && python run_research.py --help
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Подгрузка .env (если есть)
def _load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

_load_env()

from agentloop.providers import get_provider, Message
from agentloop.tools import ToolCatalog, Steward
from agentloop.cost_tracker import CostTracker
from agentloop.research import ResearchOrchestrator


def check_local_llm(base_url: str = "http://turbo:8080") -> bool:
    """Проверяет доступность локальной LLM."""
    import httpx
    try:
        r = httpx.get(f"{base_url}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        try:
            r = httpx.get(f"{base_url}/v1/models", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


def get_local_models(base_url: str) -> list[str]:
    """Возвращает список моделей с локального сервера."""
    import httpx
    try:
        r = httpx.get(f"{base_url}/v1/models", timeout=10)
        data = r.json()
        return [m.get("id", "?") for m in data.get("data", [])]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(
        description="AgentLoop Research Mode — запускает исследовательский цикл для задачи",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python run_research.py "Сгенерировать 3 варианта приветствия"
  python run_research.py "Извлечь задачи из PDF" --input '{"pdf_path": "/data/scanavi.pdf"}'
  python run_research.py "Написать стих" --hint "Используй рифму abab" --iterations 2
        """,
    )
    parser.add_argument("task", help="Описание задачи для исследовательского агента")
    parser.add_argument("--task-id", default=None, help="ID задачи (auto-generated по умолчанию)")
    parser.add_argument("--input", default=None, help="Входные данные JSON, например --input '{\"key\": \"value\"}'")
    parser.add_argument("--hint", default=None, help="Подсказка от пользователя")
    parser.add_argument("--iterations", type=int, default=3, help="Макс. кол-во итераций (default: 3)")
    parser.add_argument("--target-score", type=float, default=0.85, help="Целевой score для остановки (default: 0.85)")
    parser.add_argument("--work-dir", default=None, help="Рабочая директория (default: /tmp/agentloop_research/<task_id>)")
    parser.add_argument("--provider", default="local", choices=["local", "zai", "openrouter"], help="LLM провайдер (default: local)")
    parser.add_argument("--model", default=None, help="Модель для LLM (auto-detect по умолчанию)")
    parser.add_argument("--no-skill-save", action="store_true", help="Не сохранять навык в Skill Library")
    parser.add_argument("--dry-run", action="store_true", help="Только проверить подключение, не запускать")
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод")

    args = parser.parse_args()

    # ─── Проверка подключения ───────────────────────────────
    print("=" * 60)
    print("🤖 AgentLoop Research Mode")
    print("=" * 60)

    if args.provider == "local":
        base_url = os.getenv("LOCAL_LLM_URL", "http://turbo:8080")
        print(f"\n📡 Провайдер: local ({base_url})")
        print(f"   Проверяю подключение...", end=" ", flush=True)

        if not check_local_llm(base_url):
            print("❌ НЕДОСТУПЕН")
            print(f"   Локальный LLM сервер на {base_url} не отвечает.")
            print(f"   Убедитесь, что llama-server запущен.")
            print(f"   Проверьте LOCAL_LLM_URL в .env")
            sys.exit(1)

        print("✅ OK")

        # Показываем доступные модели
        models = get_local_models(base_url)
        if models:
            print(f"   Доступные модели: {', '.join(models)}")

        if args.dry_run:
            print("\n✅ Dry-run: подключение работает. Выход.")
            return

    # ─── Инициализация провайдера ───────────────────────────
    print(f"\n🔧 Инициализация...")

    provider_kwargs = {}
    if args.provider == "local":
        provider_kwargs["base_url"] = base_url
    elif args.provider == "zai":
        if not os.getenv("ZAI_API_KEY"):
            print("❌ ZAI_API_KEY не задан. Установите в .env")
            sys.exit(1)
    elif args.provider == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY"):
            print("❌ OPENROUTER_API_KEY не задан. Установите в .env")
            sys.exit(1)

    try:
        llm = get_provider(args.provider, **provider_kwargs)
    except Exception as e:
        print(f"❌ Ошибка инициализации провайдера: {e}")
        sys.exit(1)

    # Определяем модель
    model = args.model or os.getenv("RESEARCH_MODEL")
    if not model and args.provider == "local":
        models = get_local_models(base_url)
        if models:
            model = models[0]  # берём первую
        else:
            model = "gemma-4-26b"  # fallback

    if model:
        print(f"   Модель: {model}")
    print(f"   Провайдер: {args.provider}")

    # Quick test LLM call
    print(f"   Тестовый вызов LLM...", end=" ", flush=True)
    try:
        test_resp = llm.chat(
            [Message(role="user", content="Ответь одним словом: работает?")],
            model=model,
            temperature=0.3,
            max_tokens=32,
            timeout_sec=60,
        )
        print(f"✅ OK (\"{test_resp.content.strip()[:50]}\")")
    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        print("   Проверьте, что LLM сервер работает и модель доступна.")
        sys.exit(1)

    # ─── Подготовка Research Orchestrator ───────────────────
    catalog = ToolCatalog()

    task_id = args.task_id or f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    work_dir = args.work_dir or f"/tmp/agentloop_research/{task_id}"

    input_vars = {}
    if args.input:
        try:
            input_vars = json.loads(args.input)
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга --input: {e}")
            sys.exit(1)

    cost_tracker = CostTracker()

    print(f"\n📋 Задача: {args.task}")
    print(f"   Task ID: {task_id}")
    print(f"   Work dir: {work_dir}")
    print(f"   Max iterations: {args.iterations}")
    print(f"   Target score: {args.target_score}")
    if args.hint:
        print(f"   Hint: {args.hint}")
    if input_vars:
        print(f"   Input vars: {json.dumps(input_vars, ensure_ascii=False)}")
    print()

    orchestrator = ResearchOrchestrator(
        work_dir=work_dir,
        llm_provider=llm,
        catalog=catalog,
        cost_tracker=cost_tracker,
        hypothesis_model=model,
        builder_model=model,
        judge_model=model,
        max_iterations=args.iterations,
        target_score=args.target_score,
        auto_select_hypothesis=True,
        default_provider=args.provider,
        default_model=model,
    )

    # ─── Запуск ─────────────────────────────────────────────
    try:
        result = orchestrator.run(
            task_description=args.task,
            task_id=task_id,
            input_vars=input_vars,
            user_hint=args.hint,
        )
    except KeyboardInterrupt:
        print("\n\n⏹ Прервано пользователем")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Фатальная ошибка: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # ─── Результат ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("📊 РЕЗУЛЬТАТ")
    print(f"{'='*60}")
    print(f"   Успех: {'✅' if result.success else '❌'}")
    print(f"   Итераций выполнено: {result.iterations_run}")
    print(f"   Лучший score: {result.best_score:.3f}")
    print(f"   Лучшая гипотеза: {result.best_hypothesis_id}")
    print(f"   Общая стоимость: ${result.total_cost_usd:.4f}")
    print(f"   Общее время: {result.total_time_sec:.1f} сек")

    if result.skill_saved:
        print(f"   💾 Навык сохранён: {result.skill_id}")
        print(f"      → {result.skill_dir}")

    if result.error:
        print(f"   ⚠ Ошибка: {result.error}")

    # Детали по итерациям
    if args.verbose and result.history:
        print(f"\n📝 История итераций:")
        for entry in result.history:
            print(f"   - {entry.get('hypothesis_id', '?')}: "
                  f"score={entry.get('score', 0):.3f}, "
                  f"success={entry.get('execution_success', False)}, "
                  f"cost=${entry.get('cost_usd', 0):.4f}")
            if entry.get("feedback"):
                print(f"     feedback: {entry['feedback'][:150]}")

    # Стоимость
    if cost_tracker:
        summary = cost_tracker.summary()
        if summary.total_calls > 0:
            print(f"\n💰 Стоимость по вызовам:")
            print(f"   Всего вызовов: {summary.total_calls}")
            total_tokens = summary.total_tokens_in + summary.total_tokens_out
            print(f"   Всего токенов: {total_tokens}")
            print(f"   Стоимость: ${summary.total_cost_usd:.4f}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()

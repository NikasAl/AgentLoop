# ═════════════════════════════════════════════════════════
# HUMAN INPUT — Pipeline Node Response
# ═════════════════════════════════════════════════════════
#
# Run ID:       run_2026-06-29_15-30-00_h1_distilled
# Node:         extract_problems_teacher (LLM call)
# Role:         distillation teacher (создаём образцовый ответ для последующей
#               дистилляции промпта локальной Gemma 4)
# Reason:       Требуется эталон от сильной модели, но бюджет на API = 0.
#               Человек выступает как прокси к Gemini 3.1 Pro через браузер.
#
# Expected output: JSON с задачами в LaTeX
# Format:        {"problems": [{"number": "...", "latex": "...", "raw_text": "..."}]}
#
# Special markers (write on first line below, save, close):
#   SKIP   — пропустить узел, pipeline продолжится без него
#   ABORT  — остановить pipeline
#   RETRY  — повторить запрос (например, если промпт не скопировался в буфер)
#
# Действия:
#   1. Промпт уже в буфере обмена (Ctrl+V в чат с сильной моделью)
#   2. Также сохранён в /tmp/pipeline_h/teacher_prompt.md (если буфер затёрся)
#   3. Открой чат с Gemini Pro / Claude / GPT в браузере
#   4. Вставь промпт, отправь
#   5. Скопируй ответ модели
#   6. Вставь ответ НИЖЕ линии "PASTE RESPONSE BELOW"
#   7. Сохрани файл (Ctrl+S) и закрой (Ctrl+W)
#
# Таймаут: 30 минут
# Время ожидания: начинается сейчас...
#
# ─── PASTE RESPONSE BELOW THIS LINE ───────────────────────


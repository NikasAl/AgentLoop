"""
Human provider — человек как API через subl + clipboard.

Используется для:
- Debug-режима: пользователь отвечает сам
- Дистилляции: человек вставляет ответ от браузерной модели (Gemini Pro, Claude)
- Субъективных метрик: human-as-judge

Flow:
1. Печатаем в консоль пояснение (что за узел, что от пользователя хотят)
2. Копируем промпт в буфер обмена (xclip)
3. Открываем пустой temp-файл в subl
4. Ждём сохранения и закрытия (subl --wait блокирует)
5. Читаем содержимое как ответ
6. Логируем human_time_sec

Спецмаркеры (первая строка ответа):
- SKIP — пропустить узел
- ABORT — остановить pipeline
- RETRY — повторить запрос
"""

from __future__ import annotations

import os
import platform
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import Message, ModelInfo, ProviderError, Response


class HumanProvider:
    """
    Провайдер, где LLM = человек.

    Не имеет API key, не требует сети. Цена = 0, но human_time_sec логируется.
    """

    name = "human"

    def __init__(
        self,
        editor: str | None = None,
        clipboard_cmd: str | None = None,
        timeout_min: int = 30,
    ):
        self.editor = editor or os.getenv("EDITOR", "subl")
        self.clipboard_cmd = clipboard_cmd or self._detect_clipboard()
        self.timeout_min = timeout_min

    def _detect_clipboard(self) -> str:
        """Определяет команду для буфера обмена по платформе."""
        if platform.system() == "Linux":
            for cmd in ["xclip -selection clipboard", "wl-copy", "xsel --clipboard --input"]:
                parts = cmd.split()[0]
                try:
                    subprocess.run(["which", parts], check=True, capture_output=True)
                    return cmd
                except subprocess.CalledProcessError:
                    continue
        return "xclip -selection clipboard"

    def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.7,  # игнорируется
        max_tokens: int = 8192,
        json_mode: bool = False,
        timeout_sec: int = 1800,
        node_id: str | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> Response:
        """
        Запрашивает человеческий ввод.

        Args:
            messages: последние сообщения диалога (берётся последнее user-сообщение как prompt)
            model: "browser" / "self" / "gemini-pro-via-browser" — для логов
            node_id: ID узла в DAG (для контекста пользователю)
            reason: зачем нужен человек (например, "distillation teacher")
            timeout_sec: таймаут ожидания (default 30 min)

        Returns:
            Response с content=ответ человека, human_time_sec=сколько ждали
        """
        # Собираем промпт из messages
        prompt_text = self._format_prompt(messages)

        # Создаём temp-файл с шапкой
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_node = (node_id or "node").replace(" ", "_")
        temp_dir = Path(tempfile.gettempdir()) / "pipeline_h"
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_file = temp_dir / f"{timestamp}_{safe_node}.md"
        prompt_backup = temp_dir / f"{timestamp}_{safe_node}.prompt.md"

        # Записываем prompt в backup-файл (если буфер затрётся)
        prompt_backup.write_text(prompt_text, encoding="utf-8")

        # Копируем промпт в буфер обмена
        try:
            subprocess.run(
                self.clipboard_cmd.split(),
                input=prompt_text.encode(),
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"⚠ Clipboard failed: {e}. Prompt saved to {prompt_backup}")

        # Записываем шапку в temp-файл
        header = self._build_header(
            node_id=node_id or "unknown",
            model=model,
            reason=reason or "human input required",
            prompt_chars=len(prompt_text),
            prompt_backup=str(prompt_backup),
            timeout_min=self.timeout_min,
            expected_json=json_mode,
        )
        temp_file.write_text(header, encoding="utf-8")

        # Печатаем пояснение в консоль
        self._print_instructions(
            node_id=node_id or "unknown",
            model=model,
            reason=reason or "human input required",
            temp_file=str(temp_file),
            prompt_chars=len(prompt_text),
        )

        # Открываем editor в блокирующем режиме
        start = time.time()
        try:
            subprocess.run([self.editor, "--wait", str(temp_file)], check=True, timeout=timeout_sec)
        except subprocess.CalledProcessError as e:
            raise ProviderError(f"Editor failed: {e}", provider="human") from e
        except subprocess.TimeoutExpired:
            raise ProviderError(
                f"Human input timeout ({timeout_sec}s)", provider="human"
            ) from None

        human_time_sec = int(time.time() - start)

        # Читаем ответ (всё после "PASTE RESPONSE BELOW THIS LINE")
        content = temp_file.read_text(encoding="utf-8")
        response_text = self._extract_response(content)

        # Обработка спецмаркеров
        if response_text in ("SKIP", "ABORT", "RETRY"):
            raise ProviderError(
                f"Human returned {response_text}",
                provider="human",
            )

        # Примерная оценка токенов (для логов)
        approx_tokens = max(1, len(response_text) // 4)

        return Response(
            content=response_text,
            provider="human",
            model=f"human:{model}",
            input_tokens=len(prompt_text) // 4,
            output_tokens=approx_tokens,
            cost_usd=0.0,
            latency_ms=human_time_sec * 1000,
            human_time_sec=human_time_sec,
            raw={
                "temp_file": str(temp_file),
                "prompt_backup": str(prompt_backup),
                "editor": self.editor,
            },
        )

    def _format_prompt(self, messages: list[Message]) -> str:
        """Форматирует messages в единый текст для копирования в браузер."""
        parts = []
        for m in messages:
            role_label = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT"}.get(
                m.role, m.role.upper()
            )
            parts.append(f"### {role_label}\n{m.content}")
        return "\n\n".join(parts)

    def _build_header(
        self,
        node_id: str,
        model: str,
        reason: str,
        prompt_chars: int,
        prompt_backup: str,
        timeout_min: int,
        expected_json: bool,
    ) -> str:
        json_note = ""
        if expected_json:
            json_note = "\n# Expected format: JSON"

        return f"""# ═════════════════════════════════════════════════════════
# HUMAN INPUT — Pipeline Node Response
# ═════════════════════════════════════════════════════════
#
# Node:          {node_id}
# Role:          {reason}
# Model:         human:{model}
# Prompt size:   {prompt_chars} chars
# Prompt backup: {prompt_backup}{json_note}
#
# Special markers (write on first line below, save, close):
#   SKIP   — skip this node, continue pipeline
#   ABORT  — stop pipeline entirely
#   RETRY  — repeat the request (e.g., clipboard didn't work)
#
# Действия:
#   1. Prompt уже в буфере обмена (Ctrl+V в чат с моделью/редактор)
#   2. Также сохранён в {prompt_backup}
#   3. Вставь ответ НИЖЕ линии "PASTE RESPONSE BELOW"
#   4. Сохрани файл (Ctrl+S) и закрой (Ctrl+W)
#
# Таймаут: {timeout_min} минут
#
# ─── PASTE RESPONSE BELOW THIS LINE ───────────────────────

"""

    def _print_instructions(
        self,
        node_id: str,
        model: str,
        reason: str,
        temp_file: str,
        prompt_chars: int,
    ) -> None:
        print(
            f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏸  HUMAN INPUT REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Node:          {node_id}
Role:          {reason}
Model:         human:{model}
Prompt size:   {prompt_chars} chars (copied to clipboard)
Temp file:     {temp_file}

Действия:
  1. Открой чат с моделью (browser / IDE) или напиши ответ сам
  2. Вставь промпт (Ctrl+V)
  3. Скопируй ответ
  4. Вставь ответ в temp file (subl открыт)
  5. Сохрани и закрой файл (Ctrl+S, Ctrl+W)

Управление:
  Записать 'SKIP'   → пропустить узел
  Записать 'ABORT'  → остановить pipeline
  Записать 'RETRY'  → повторить запрос

Ожидание ответа...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        )

    def _extract_response(self, content: str) -> str:
        """Парсит temp-файл: всё после 'PASTE RESPONSE BELOW THIS LINE'."""
        parts = content.split("PASTE RESPONSE BELOW THIS LINE")
        if len(parts) < 2:
            return "SKIP"
        response = parts[1].strip()
        # Убираем возможные trailing пустые строки
        return response.strip()

    def list_models(self) -> list[ModelInfo]:
        """Виртуальные модели для human provider."""
        return [
            ModelInfo(
                name="browser",
                provider="human",
                full_id="human:browser",
                tier=999,
                capabilities=[Message.__mro__[0] and "text"],  # type: ignore
                price_input_usd_per_1m=0.0,
                price_output_usd_per_1m=0.0,
                max_context=100000,
                description="Human using browser-based LLM (Gemini/Claude/GPT)",
                rating=1.0,
                notes="Бесплатно, но требует человеческого времени",
            ),
            ModelInfo(
                name="self",
                provider="human",
                full_id="human:self",
                tier=999,
                price_input_usd_per_1m=0.0,
                price_output_usd_per_1m=0.0,
                max_context=100000,
                description="Human writing response directly",
                rating=1.0,
            ),
        ]

    def health_check(self) -> bool:
        """Проверка, что editor доступен."""
        try:
            subprocess.run(["which", self.editor.split()[0]], check=True, capture_output=True)
            return True
        except Exception:
            return False

"""
Cost Tracker — логирование всех LLM-вызовов в SQLite.

Использование:
    from agentloop.cost_tracker import CostTracker

    tracker = CostTracker("/path/to/usage.sqlite")
    tracker.log(
        task_id="scanavi_extract_research_001",
        provider="local",
        model="gemma-4-26b",
        input_tokens=850,
        output_tokens=1820,
        cost_usd=0.0,
        human_time_sec=0,
        node_id="extract_problems",
        run_id="run_2026-06-29_14-10-00_h1",
    )

    summary = tracker.summary(task_id="scanavi_extract_research_001")
    print(f"Spent: ${summary.total_cost_usd:.4f}")
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class CostSummary:
    """Агрегированная статистика по расходам."""

    total_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    total_human_time_sec: int = 0
    by_provider: dict[str, dict] = None  # type: ignore

    def __post_init__(self):
        if self.by_provider is None:
            self.by_provider = defaultdict(
                lambda: {
                    "calls": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost_usd": 0.0,
                    "human_time_sec": 0,
                }
            )


SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    task_id TEXT NOT NULL,
    run_id TEXT,
    node_id TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    latency_ms INTEGER DEFAULT 0,
    human_time_sec INTEGER DEFAULT 0,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_usage_task ON usage(task_id);
CREATE INDEX IF NOT EXISTS idx_usage_run ON usage(run_id);
CREATE INDEX IF NOT EXISTS idx_usage_provider ON usage(provider);
CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage(timestamp);
"""


class CostTracker:
    """Логирование LLM-вызовов в SQLite."""

    def __init__(self, db_path: str | Path = "usage.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA)

    def log(
        self,
        *,
        task_id: str,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: int = 0,
        human_time_sec: int = 0,
        run_id: str | None = None,
        node_id: str | None = None,
        notes: str | None = None,
    ) -> int:
        """Логирует один вызов. Возвращает ID записи."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO usage (
                    timestamp, task_id, run_id, node_id, provider, model,
                    input_tokens, output_tokens, cache_read_tokens,
                    cost_usd, latency_ms, human_time_sec, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    task_id,
                    run_id,
                    node_id,
                    provider,
                    model,
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cost_usd,
                    latency_ms,
                    human_time_sec,
                    notes,
                ),
            )
            return cur.lastrowid or 0

    def summary(
        self,
        *,
        task_id: Optional[str] = None,
        run_id: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> CostSummary:
        """Агрегированная статистика с фильтрами."""
        where = []
        params: list = []
        if task_id:
            where.append("task_id = ?")
            params.append(task_id)
        if run_id:
            where.append("run_id = ?")
            params.append(run_id)
        if provider:
            where.append("provider = ?")
            params.append(provider)

        where_clause = f"WHERE {' AND '.join(where)}" if where else ""

        with sqlite3.connect(self.db_path) as conn:
            # Total
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*),
                    COALESCE(SUM(input_tokens), 0),
                    COALESCE(SUM(output_tokens), 0),
                    COALESCE(SUM(cost_usd), 0),
                    COALESCE(SUM(human_time_sec), 0)
                FROM usage {where_clause}
                """,
                params,
            ).fetchone()

            summary = CostSummary(
                total_calls=row[0],
                total_tokens_in=row[1],
                total_tokens_out=row[2],
                total_cost_usd=row[3],
                total_human_time_sec=row[4],
            )

            # By provider
            for prow in conn.execute(
                f"""
                SELECT
                    provider,
                    COUNT(*),
                    COALESCE(SUM(input_tokens), 0),
                    COALESCE(SUM(output_tokens), 0),
                    COALESCE(SUM(cost_usd), 0),
                    COALESCE(SUM(human_time_sec), 0)
                FROM usage {where_clause}
                GROUP BY provider
                """,
                params,
            ).fetchall():
                summary.by_provider[prow[0]] = {
                    "calls": prow[1],
                    "tokens_in": prow[2],
                    "tokens_out": prow[3],
                    "cost_usd": prow[4],
                    "human_time_sec": prow[5],
                }

            return summary

    def recent(self, limit: int = 20, task_id: Optional[str] = None) -> list[dict]:
        """Последние N записей."""
        where = "WHERE task_id = ?" if task_id else ""
        params: list = [task_id] if task_id else []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT * FROM usage {where}
                ORDER BY timestamp DESC LIMIT ?
                """,
                params + [limit],
            ).fetchall()
            return [dict(r) for r in rows]

    def budget_check(
        self,
        task_id: str,
        budget_usd: float,
    ) -> tuple[float, float, bool]:
        """
        Проверка бюджета задачи.

        Returns:
            (spent_usd, budget_usd, exceeded)
        """
        s = self.summary(task_id=task_id)
        return (s.total_cost_usd, budget_usd, s.total_cost_usd >= budget_usd)

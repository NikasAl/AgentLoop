"""
SkillLibraryWriter / SkillLibraryReader — сохранение и загрузка навыков.

Skill — self-contained директория с:
- skill.yaml: метаданные + pipeline spec
- pipeline.json: DAG
- prompts/: промпты
- tools/custom/: кастомные Python-инструменты
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..executor.executor import ExecutionResult
from .evaluator import EvaluationResult
from .builder import BuildResult


@dataclass
class Skill:
    """Навык в Skill Library."""

    skill_id: str
    task_description: str
    task_fingerprint: str
    pipeline: dict[str, Any]
    prompts: dict[str, Any]
    research_history: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)
    required_tools: dict[str, Any] = field(default_factory=dict)
    status: str = "draft"  # draft | testing | production_ready | deprecated
    version: str = "1.0"
    created: str = ""
    notes: str = ""

    def to_yaml_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "created": self.created,
            "status": self.status,
            "task": {
                "description": self.task_description,
                "fingerprint": self.task_fingerprint,
            },
            "research": self.research_history,
            "pipeline": {"version": self.version},
            "prompts": self.prompts,
            "required_tools": self.required_tools,
            "performance": self.performance,
            "notes": self.notes,
        }


class SkillLibraryWriter:
    """
    Сохраняет навык в Skill Library.

    Структура:
        skills/
        └── {skill_id}/
            ├── skill.yaml
            ├── pipeline.json
            ├── prompts/
            └── tools/custom/
    """

    def __init__(self, skills_dir: Path | str):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        skill_id: str,
        task_description: str,
        build_result: BuildResult,
        execution_result: ExecutionResult,
        evaluation_result: EvaluationResult,
        custom_tools_dir: Path | None = None,
        notes: str = "",
        status: str = "testing",
    ) -> Path:
        """
        Сохраняет навык в директорию skills/{skill_id}/.

        Args:
            skill_id: уникальный ID (например, "scanavi_extract_v1")
            task_description: описание задачи
            build_result: результат Builder'а (DAG)
            execution_result: результат выполнения
            evaluation_result: результат оценки
            custom_tools_dir: директория с custom tools (если есть)
            notes: заметки
            status: начальный статус

        Returns:
            Путь к созданной директории навыка
        """
        skill_dir = self.skills_dir / skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Сохраняем DAG
        pipeline_path = skill_dir / "pipeline.json"
        pipeline_path.write_text(
            json.dumps(build_result.dag, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        # Сохраняем prompts
        prompts_dir = skill_dir / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        prompts = build_result.dag.get("prompts", {})
        for name, prompt_data in prompts.items():
            prompt_file = prompts_dir / f"{name}.md"
            if isinstance(prompt_data, dict):
                prompt_file.write_text(prompt_data.get("text", ""), encoding="utf-8")
            else:
                prompt_file.write_text(str(prompt_data), encoding="utf-8")

        # Копируем custom tools
        tools_dir = skill_dir / "tools" / "custom"
        tools_dir.mkdir(parents=True, exist_ok=True)
        custom_tools_created: list[str] = []
        if custom_tools_dir and custom_tools_dir.exists():
            for tool_file in custom_tools_dir.glob("*.py"):
                shutil.copy2(tool_file, tools_dir / tool_file.name)
                custom_tools_created.append(tool_file.stem)

        # Сохраняем skill.yaml
        skill = Skill(
            skill_id=skill_id,
            task_description=task_description,
            task_fingerprint=self._make_fingerprint(task_description),
            pipeline=build_result.dag,
            prompts={name: (p if isinstance(p, dict) else {"text": p}) for name, p in prompts.items()},
            research_history={
                "total_hypotheses_tested": 1,
                "iterations": 1,
                "research_cost_usd": execution_result.total_cost_usd,
                "research_time_sec": execution_result.total_duration_sec,
                "baseline_score": evaluation_result.composite_score,
                "final_score": evaluation_result.composite_score,
                "score_history": [
                    {
                        "iteration": 1,
                        "hypothesis_id": build_result.hypothesis_id,
                        "score": evaluation_result.composite_score,
                        "cost_usd": execution_result.total_cost_usd,
                    }
                ],
                "failure_patterns_observed": [
                    m.description for m in evaluation_result.metrics if m.value < 0.5
                ],
            },
            performance={
                "tokens_total": execution_result.total_tokens,
                "latency_sec": execution_result.total_duration_sec,
                "cost_usd": execution_result.total_cost_usd,
                "composite_score": evaluation_result.composite_score,
                "metrics": [
                    {"name": m.name, "value": m.value, "weight": m.weight}
                    for m in evaluation_result.metrics
                ],
                "last_validated": datetime.now(timezone.utc).isoformat(),
            },
            required_tools={
                "custom": [{"id": tid, "safety_checked": True} for tid in custom_tools_created],
                "system": [],  # TODO: извлечь из DAG
            },
            status=status,
            version="1.0",
            created=datetime.now(timezone.utc).isoformat(),
            notes=notes or f"Created after research iteration. Score: {evaluation_result.composite_score:.3f}",
        )

        skill_yaml_path = skill_dir / "skill.yaml"
        skill_yaml_path.write_text(
            yaml.safe_dump(skill.to_yaml_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        return skill_dir

    def _make_fingerprint(self, description: str) -> str:
        """Создает простой fingerprint из описания задачи."""
        import hashlib
        return hashlib.md5(description.encode("utf-8")).hexdigest()[:12]


class SkillLibraryReader:
    """Читает навыки из Skill Library."""

    def __init__(self, skills_dir: Path | str):
        self.skills_dir = Path(skills_dir)

    def list_skills(self) -> list[dict[str, Any]]:
        """Возвращает список всех навыков (метаданные)."""
        skills = []
        if not self.skills_dir.exists():
            return skills

        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            yaml_path = skill_dir / "skill.yaml"
            if not yaml_path.exists():
                continue
            try:
                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                skills.append({
                    "skill_id": data.get("skill_id", skill_dir.name),
                    "task_description": data.get("task", {}).get("description", ""),
                    "status": data.get("status", "unknown"),
                    "version": data.get("version", "?"),
                    "composite_score": data.get("performance", {}).get("composite_score", 0),
                    "cost_usd": data.get("performance", {}).get("cost_usd", 0),
                    "dir": str(skill_dir),
                })
            except Exception:
                continue

        return skills

    def load_skill(self, skill_id: str) -> Skill | None:
        """Загружает навык по ID."""
        skill_dir = self.skills_dir / skill_id
        yaml_path = skill_dir / "skill.yaml"
        if not yaml_path.exists():
            return None

        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            pipeline_path = skill_dir / "pipeline.json"
            pipeline = json.loads(pipeline_path.read_text(encoding="utf-8")) if pipeline_path.exists() else {}

            return Skill(
                skill_id=data.get("skill_id", skill_id),
                task_description=data.get("task", {}).get("description", ""),
                task_fingerprint=data.get("task", {}).get("fingerprint", ""),
                pipeline=pipeline,
                prompts=data.get("prompts", {}),
                research_history=data.get("research", {}),
                performance=data.get("performance", {}),
                required_tools=data.get("required_tools", {}),
                status=data.get("status", "unknown"),
                version=data.get("version", "1.0"),
                created=data.get("created", ""),
                notes=data.get("notes", ""),
            )
        except Exception:
            return None

    def find_similar(self, task_description: str) -> list[dict[str, Any]]:
        """Ищет похожие навыки по описанию задачи (простой keyword match)."""
        import re
        query_words = set(re.findall(r"\w+", task_description.lower()))
        results = []

        for skill_meta in self.list_skills():
            desc = skill_meta["task_description"].lower()
            desc_words = set(re.findall(r"\w+", desc))
            overlap = len(query_words & desc_words)
            if overlap > 0:
                skill_meta["similarity"] = overlap
                results.append(skill_meta)

        results.sort(key=lambda x: -x["similarity"])
        return results

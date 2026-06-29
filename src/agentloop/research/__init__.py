"""
Research Mode — цикл исследования: hypothesis → build → execute → evaluate → feedback.

Главные компоненты:
- HypothesisGenerator: LLM-агент, генерирует гипотезы по задаче
- PipelineBuilder: LLM-агент, строит DAG из гипотезы
- Evaluator: оценивает результат выполнения
- SkillLibraryWriter: сохраняет успешные pipelines
- ResearchOrchestrator: связывает всё в цикл
"""

from .hypothesis import Hypothesis, HypothesisGenerator, HypothesisSet
from .builder import BuildResult, PipelineBuilder
from .evaluator import Evaluator, EvaluationResult, Metric
from .skill_library import SkillLibraryWriter, SkillLibraryReader, Skill
from .orchestrator import ResearchOrchestrator, ResearchResult

__all__ = [
    "BuildResult",
    "Evaluator",
    "EvaluationResult",
    "Hypothesis",
    "HypothesisGenerator",
    "HypothesisSet",
    "Metric",
    "PipelineBuilder",
    "ResearchOrchestrator",
    "ResearchResult",
    "Skill",
    "SkillLibraryReader",
    "SkillLibraryWriter",
]

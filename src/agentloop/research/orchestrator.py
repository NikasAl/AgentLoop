"""
ResearchOrchestrator — связывает всё в цикл исследования.

Цикл:
1. HypothesisGenerator → 3 гипотезы
2. Пользователь выбирает одну (или auto: первая)
3. PipelineBuilder → DAG
4. PipelineExecutor → результат
5. Evaluator → score + feedback
6. Если score < target → feedback в HypothesisGenerator, повтор
7. Если score >= target → SkillLibraryWriter сохраняет навык
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..cost_tracker import CostTracker
from ..executor.executor import ExecutionResult, PipelineExecutor
from ..providers import Provider, get_provider
from ..tools import Steward, ToolCatalog
from .builder import BuildResult, PipelineBuilder
from .evaluator import EvaluationResult, Evaluator
from .hypothesis import Hypothesis, HypothesisGenerator, HypothesisSet
from .skill_library import SkillLibraryWriter


@dataclass
class ResearchResult:
    """Итог research-цикла."""

    task_id: str
    success: bool
    iterations_run: int
    best_hypothesis_id: str | None
    best_score: float
    best_execution_result: ExecutionResult | None
    best_evaluation_result: EvaluationResult | None
    best_build_result: BuildResult | None
    skill_saved: bool
    skill_id: str | None
    skill_dir: str | None
    history: list[dict[str, Any]] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_time_sec: float = 0.0
    error: str | None = None


class ResearchOrchestrator:
    """
    Главный orchestrator research-режима.

    Использование:
        orchestrator = ResearchOrchestrator(
            work_dir="/tmp/research",
            llm_provider=get_provider("local"),
            catalog=ToolCatalog(),
        )
        result = orchestrator.run(
            task_description="Извлечь задачи из PDF",
            task_id="scanavi_001",
            input_vars={"$INPUT": "/data/scanavi.pdf"},
        )
    """

    def __init__(
        self,
        work_dir: Path | str,
        llm_provider: Provider,
        catalog: ToolCatalog | None = None,
        steward: Steward | None = None,
        cost_tracker: CostTracker | None = None,
        skills_dir: Path | str | None = None,
        hypothesis_model: str = "gemma-4-26b",
        builder_model: str = "gemma-4-26b",
        judge_model: str = "gemma-4-26b",
        max_iterations: int = 3,
        target_score: float = 0.85,
        auto_select_hypothesis: bool = True,
        hypothesis_selector: Callable[[HypothesisSet], Hypothesis] | None = None,
        default_provider: str | None = None,
        default_model: str | None = None,
    ):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.llm = llm_provider
        self.catalog = catalog
        self.steward = steward
        self.cost_tracker = cost_tracker

        self.skills_dir = Path(skills_dir) if skills_dir else self.work_dir.parent / "skills"
        self.skill_writer = SkillLibraryWriter(self.skills_dir)

        self.hypothesis_generator = HypothesisGenerator(
            llm_provider=llm_provider,
            model=hypothesis_model,
            catalog=catalog,
        )
        self.builder = PipelineBuilder(
            llm_provider=llm_provider,
            model=builder_model,
            catalog=catalog,
            steward=steward,
        )
        self.evaluator = Evaluator(
            llm_provider=llm_provider,
            judge_model=judge_model,
        )

        self.max_iterations = max_iterations
        self.target_score = target_score
        self.auto_select_hypothesis = auto_select_hypothesis
        self.hypothesis_selector = hypothesis_selector
        self.default_provider = default_provider
        self.default_model = default_model

    def run(
        self,
        task_description: str,
        task_id: str = "task",
        input_vars: dict[str, Any] | None = None,
        user_hint: str | None = None,
        user_constraints: dict[str, Any] | None = None,
        sample_input: dict[str, Any] | None = None,
        skill_id: str | None = None,
    ) -> ResearchResult:
        """
        Запускает research-цикл.

        Args:
            task_description: что нужно сделать
            task_id: ID задачи
            input_vars: $INPUT, $PAGE_NUM и т.д. для pipeline
            user_hint: подсказка пользователем
            user_constraints: бюджет, лимиты
            sample_input: пример входных данных (для hypothesis gen)
            skill_id: ID навыка для сохранения (auto-generated если None)

        Returns:
            ResearchResult с лучшим результатом
        """
        start_time = time.time()
        input_vars = input_vars or {}
        user_constraints = user_constraints or {}
        sample_input = sample_input or {}

        history: list[dict[str, Any]] = []
        best_result: ResearchResult = ResearchResult(
            task_id=task_id,
            success=False,
            iterations_run=0,
            best_hypothesis_id=None,
            best_score=0.0,
            best_execution_result=None,
            best_evaluation_result=None,
            best_build_result=None,
            skill_saved=False,
            skill_id=None,
            skill_dir=None,
        )

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"🔬 RESEARCH ITERATION {iteration}/{self.max_iterations}")
            print(f"{'='*60}")

            # 1. Генерируем гипотезы
            print("\n📝 Generating hypotheses...")
            hypothesis_set = self.hypothesis_generator.generate(
                task_description=task_description,
                sample_input=sample_input,
                user_constraints=user_constraints,
                user_hint=user_hint,
                task_id=task_id,
                history=history,
            )
            print(f"   Generated {len(hypothesis_set.hypotheses)} hypotheses:")
            for h in hypothesis_set.hypotheses:
                print(f"   - {h.id}: {h.title}")

            # 2. Выбираем гипотезу
            if self.hypothesis_selector:
                selected = self.hypothesis_selector(hypothesis_set)
            elif self.auto_select_hypothesis:
                selected = hypothesis_set.hypotheses[0]
                print(f"\n✓ Auto-selected: {selected.id}")
            else:
                # В реальной системе — interactive choice
                selected = hypothesis_set.hypotheses[0]
                print(f"\n✓ Selected (default): {selected.id}")

            # 3. Строим DAG
            print(f"\n🔨 Building pipeline for {selected.id}...")
            build_result = self.builder.build(selected, task_id=task_id)

            # Переписываем модели в DAG на дефолтный провайдер (если задан)
            if build_result.success and self.default_provider:
                self._rewrite_dag_models(build_result.dag)

            if not build_result.success:
                print(f"   ✗ Build failed: {build_result.error}")
                history.append({
                    "hypothesis_id": selected.id,
                    "score": 0.0,
                    "feedback": f"Build failed: {build_result.error}",
                })
                continue

            print(f"   ✓ DAG built: {len(build_result.dag.get('nodes', []))} nodes")

            # 4. Выполняем
            print(f"\n▶ Executing pipeline...")
            run_dir = self.work_dir / f"run_iter{iteration}_{selected.id}"
            executor = PipelineExecutor(
                work_dir=run_dir,
                cost_tracker=self.cost_tracker,
                checkpoint_enabled=False,
            )
            execution_result = executor.execute(
                dag_dict=build_result.dag,
                task_id=task_id,
                hypothesis_id=selected.id,
                mode="research",
                input_vars=input_vars,
            )

            if execution_result.success:
                print(f"   ✓ Execution succeeded")
            else:
                print(f"   ✗ Execution failed: {list(execution_result.errors.keys())}")

            # 5. Оцениваем
            print(f"\n📊 Evaluating...")
            evaluation_result = self.evaluator.evaluate(
                execution_result=execution_result,
                hypothesis_id=selected.id,
            )
            print(f"   Composite score: {evaluation_result.composite_score:.3f}")
            for m in evaluation_result.metrics:
                print(f"   - {m.name}: {m.value:.3f} (weight={m.weight})")

            # Обновляем history
            history.append({
                "hypothesis_id": selected.id,
                "score": evaluation_result.composite_score,
                "feedback": "; ".join(evaluation_result.feedback.get("weaknesses", [])),
                "execution_success": execution_result.success,
                "cost_usd": execution_result.total_cost_usd,
            })

            # Обновляем best
            if evaluation_result.composite_score > best_result.best_score:
                best_result.best_score = evaluation_result.composite_score
                best_result.best_hypothesis_id = selected.id
                best_result.best_execution_result = execution_result
                best_result.best_evaluation_result = evaluation_result
                best_result.best_build_result = build_result

            best_result.total_cost_usd += execution_result.total_cost_usd

            # 6. Проверяем target
            if evaluation_result.composite_score >= self.target_score:
                print(f"\n✓ Target score {self.target_score} reached!")
                best_result.success = True
                break

            print(f"\n   Score {evaluation_result.composite_score:.3f} < target {self.target_score}")
            print(f"   Feedback: {evaluation_result.feedback.get('suggestions_for_next_iteration', [])[:2]}")

        best_result.iterations_run = min(iteration, self.max_iterations)
        best_result.history = history
        best_result.total_time_sec = time.time() - start_time

        # Сохраняем навык если есть успех
        if best_result.best_execution_result and best_result.best_build_result:
            skill_id = skill_id or f"skill_{task_id}_{int(time.time())}"
            print(f"\n💾 Saving skill: {skill_id}")
            try:
                custom_tools_dir = Path.home() / ".agentloop" / "custom_tools"
                skill_dir = self.skill_writer.save(
                    skill_id=skill_id,
                    task_description=task_description,
                    build_result=best_result.best_build_result,
                    execution_result=best_result.best_execution_result,
                    evaluation_result=best_result.best_evaluation_result,
                    custom_tools_dir=custom_tools_dir if custom_tools_dir.exists() else None,
                    notes=f"Created after {best_result.iterations_run} research iterations. Best score: {best_result.best_score:.3f}",
                    status="production_ready" if best_result.success else "testing",
                )
                best_result.skill_saved = True
                best_result.skill_id = skill_id
                best_result.skill_dir = str(skill_dir)
                print(f"   ✓ Saved to: {skill_dir}")
            except Exception as e:
                print(f"   ✗ Save failed: {e}")
                best_result.error = f"Skill save failed: {e}"

        return best_result

    def _rewrite_dag_models(self, dag: dict[str, Any]) -> None:
        """Переписывает провайдеры в model-полях всех LLM-узлов на default_provider.

        LLM может сгенерировать модель вида 'openrouter:gpt-4', но если у нас
        только local — переписываем на 'local:<default_model>'.
        """
        model_name = self.default_model or "gemma-4-26b"
        full_model = f"{self.default_provider}:{model_name}"

        for node in dag.get("nodes", []):
            if node.get("type") == "llm" and "model" in node:
                old_model = node["model"]
                if ":" in old_model:
                    old_provider = old_model.split(":", 1)[0]
                    if old_provider != self.default_provider:
                        print(f"   ⚠ Rewriting model in '{node['id']}': "
                              f"{old_model} → {full_model}")
                        node["model"] = full_model
                else:
                    node["model"] = full_model

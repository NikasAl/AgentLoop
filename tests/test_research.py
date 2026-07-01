"""Tests for Research Mode."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentloop.research import (
    BuildResult,
    Evaluator,
    EvaluationResult,
    Hypothesis,
    HypothesisGenerator,
    HypothesisSet,
    PipelineBuilder,
    ResearchOrchestrator,
    ResearchResult,
    Skill,
    SkillLibraryReader,
    SkillLibraryWriter,
)
from agentloop.executor.executor import ExecutionResult
from agentloop.providers.base import Response


# ─── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def mock_llm():
    """Мок LLM-провайдера."""
    llm = MagicMock()
    llm.name = "mock"
    llm.list_models.return_value = []
    response = Response(
        content="{}",
        provider="mock",
        model="mock",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.0,
        latency_ms=10,
    )
    llm.chat.return_value = response
    return llm


@pytest.fixture
def sample_execution_result():
    """Успешный execution result."""
    return ExecutionResult(
        run_id="test_run",
        success=True,
        completed_nodes=["n1", "n2"],
        final_output={"result": "test_data", "metadata": {"key": "value"}},
        total_duration_sec=5.0,
        total_cost_usd=0.0,
        total_tokens=500,
    )


@pytest.fixture
def sample_hypothesis():
    return Hypothesis(
        id="h1",
        title="Test hypothesis",
        rationale="Test rationale",
        approach=["step1", "step2"],
        model_assignments=[{"role": "main", "model": "local:gemma-4-26b", "reason": "test"}],
        custom_tools_needed=[],
        estimated={"tokens_per_page": 1000},
        risks=["test risk"],
    )


# ─── HypothesisGenerator ───────────────────────────────────


class TestHypothesisGenerator:
    def test_init(self, mock_llm):
        gen = HypothesisGenerator(llm_provider=mock_llm, model="test-model")
        assert gen.model == "test-model"
        assert gen.llm == mock_llm

    def test_generate_success(self, mock_llm):
        # Настраиваем мок на возврат 3 гипотез
        mock_llm.chat.return_value = Response(
            content=json.dumps({
                "hypotheses": [
                    {
                        "id": "h1",
                        "title": "Approach A",
                        "rationale": "First approach",
                        "approach": ["step1", "step2"],
                        "model_assignments": [{"role": "main", "model": "local:gemma-4-26b", "reason": "free"}],
                        "custom_tools_needed": [],
                        "estimated": {"tokens_per_page": 1000},
                        "risks": ["risk1"],
                    },
                    {
                        "id": "h2",
                        "title": "Approach B",
                        "rationale": "Second approach",
                        "approach": ["step1"],
                        "model_assignments": [],
                        "custom_tools_needed": [],
                        "estimated": {},
                        "risks": [],
                    },
                    {
                        "id": "h3",
                        "title": "Approach C",
                        "rationale": "Third approach",
                        "approach": [],
                        "model_assignments": [],
                        "custom_tools_needed": [],
                        "estimated": {},
                        "risks": [],
                    },
                ]
            }),
            provider="mock",
            model="mock",
            input_tokens=100,
            output_tokens=500,
        )

        gen = HypothesisGenerator(llm_provider=mock_llm, model="test-model")
        result = gen.generate(task_description="Test task", task_id="t1")

        assert isinstance(result, HypothesisSet)
        assert len(result.hypotheses) == 3
        assert result.hypotheses[0].id == "h1"
        assert result.hypotheses[1].title == "Approach B"

    def test_generate_fallback_on_llm_error(self, mock_llm):
        mock_llm.chat.side_effect = Exception("LLM failed")

        gen = HypothesisGenerator(llm_provider=mock_llm, model="test-model")
        result = gen.generate(task_description="Test")

        assert len(result.hypotheses) == 1
        assert result.hypotheses[0].id == "h_fallback"
        assert result.metadata.get("fallback") is True

    def test_generate_with_invalid_json(self, mock_llm):
        mock_llm.chat.return_value = Response(
            content="not a json at all",
            provider="mock",
            model="mock",
            input_tokens=10,
            output_tokens=5,
        )

        gen = HypothesisGenerator(llm_provider=mock_llm, model="test-model")
        result = gen.generate(task_description="Test")

        # Должен быть fallback
        assert len(result.hypotheses) == 1
        assert result.hypotheses[0].id == "h_fallback"

    def test_history_aware(self, mock_llm):
        mock_llm.chat.return_value = Response(
            content='{"hypotheses": [{"id": "h1", "title": "T", "rationale": "R", "approach": [], "model_assignments": [], "custom_tools_needed": [], "estimated": {}, "risks": []}]}',
            provider="mock",
            model="mock",
            input_tokens=10,
            output_tokens=5,
        )

        gen = HypothesisGenerator(llm_provider=mock_llm, model="test-model")
        history = [{"hypothesis_id": "h_prev", "score": 0.5, "feedback": "failed"}]
        result = gen.generate(task_description="Test", history=history)

        assert result.metadata["history_aware"] is True
        assert result.metadata["previous_attempts_considered"] == 1


# ─── PipelineBuilder ───────────────────────────────────────


class TestPipelineBuilder:
    def test_init(self, mock_llm):
        builder = PipelineBuilder(llm_provider=mock_llm, model="test")
        assert builder.model == "test"

    def test_build_success(self, mock_llm, sample_hypothesis):
        dag = {
            "nodes": [
                {"id": "n1", "type": "bash", "command": "echo hello", "timeout_sec": 5},
            ],
            "edges": [],
            "entry": "n1",
            "exit": "n1",
        }
        mock_llm.chat.return_value = Response(
            content=json.dumps(dag),
            provider="mock",
            model="mock",
            input_tokens=100,
            output_tokens=300,
        )

        builder = PipelineBuilder(llm_provider=mock_llm, model="test")
        result = builder.build(sample_hypothesis, task_id="t1")

        assert result.success
        assert len(result.dag["nodes"]) == 1
        assert result.dag["entry"] == "n1"

    def test_build_invalid_dag(self, mock_llm, sample_hypothesis):
        # DAG без nodes
        mock_llm.chat.return_value = Response(
            content='{"edges": []}',
            provider="mock",
            model="mock",
            input_tokens=10,
            output_tokens=5,
        )

        builder = PipelineBuilder(llm_provider=mock_llm, model="test")
        result = builder.build(sample_hypothesis)

        assert not result.success
        assert "validation" in (result.error or "").lower()

    def test_build_llm_failure(self, mock_llm, sample_hypothesis):
        mock_llm.chat.side_effect = Exception("LLM down")

        builder = PipelineBuilder(llm_provider=mock_llm, model="test")
        result = builder.build(sample_hypothesis)

        assert not result.success
        assert "LLM call failed" in (result.error or "")

    def test_validate_dag_missing_id(self, mock_llm, sample_hypothesis):
        dag = {
            "nodes": [{"type": "bash", "command": "echo"}],  # нет id
            "edges": [],
        }
        mock_llm.chat.return_value = Response(
            content=json.dumps(dag),
            provider="mock",
            model="mock",
            input_tokens=10,
            output_tokens=5,
        )

        builder = PipelineBuilder(llm_provider=mock_llm, model="test")
        result = builder.build(sample_hypothesis)
        assert not result.success
        assert "missing 'id'" in (result.error or "")

    def test_validate_dag_unknown_edge(self, mock_llm, sample_hypothesis):
        dag = {
            "nodes": [{"id": "n1", "type": "bash", "command": "echo"}],
            "edges": [{"from": "n1", "to": "nonexistent"}],
            "entry": "n1",
            "exit": "n1",
        }
        mock_llm.chat.return_value = Response(
            content=json.dumps(dag),
            provider="mock",
            model="mock",
            input_tokens=10,
            output_tokens=5,
        )

        builder = PipelineBuilder(llm_provider=mock_llm, model="test")
        result = builder.build(sample_hypothesis)
        assert not result.success
        assert "unknown node" in (result.error or "").lower()

    def test_validate_llm_node_requires_user_prompt_template(self, mock_llm, sample_hypothesis):
        """Builder должен отклонять LLM-узлы без user_prompt_template."""
        dag = {
            "nodes": [{
                "id": "n1",
                "type": "llm",
                "model": "local:gemma-4-26b",
                "system_prompt_ref": "extractor_v1",
                # НЕТ user_prompt_template — это баг!
                "json_mode": True,
            }],
            "edges": [],
            "entry": "n1",
            "exit": "n1",
        }
        mock_llm.chat.return_value = Response(
            content=json.dumps(dag),
            provider="mock",
            model="mock",
            input_tokens=10,
            output_tokens=5,
        )

        builder = PipelineBuilder(llm_provider=mock_llm, model="test")
        result = builder.build(sample_hypothesis)
        assert not result.success
        assert "user_prompt_template" in (result.error or "")

    def test_validate_llm_node_with_user_prompt_template(self, mock_llm, sample_hypothesis):
        """Builder должен принимать LLM-узлы с user_prompt_template."""
        dag = {
            "nodes": [{
                "id": "n1",
                "type": "llm",
                "model": "local:gemma-4-26b",
                "system_prompt_ref": "extractor_v1",
                "user_prompt_template": "Извлеки задачи: {$INPUT}",
                "json_mode": True,
            }],
            "edges": [],
            "entry": "n1",
            "exit": "n1",
            "prompts": {"extractor_v1": {"text": "Ты извлекатель", "version": "v1"}},
        }
        mock_llm.chat.return_value = Response(
            content=json.dumps(dag),
            provider="mock",
            model="mock",
            input_tokens=10,
            output_tokens=100,
        )

        builder = PipelineBuilder(llm_provider=mock_llm, model="test")
        result = builder.build(sample_hypothesis)
        assert result.success


# ─── Evaluator ─────────────────────────────────────────────


class TestEvaluator:
    def test_evaluate_success(self, sample_execution_result):
        evaluator = Evaluator()
        result = evaluator.evaluate(sample_execution_result, hypothesis_id="h1")

        assert result.success
        assert result.composite_score > 0.5
        assert len(result.metrics) >= 4
        assert any(m.name == "execution_success" and m.value == 1.0 for m in result.metrics)

    def test_evaluate_failure(self):
        failed_result = ExecutionResult(
            run_id="r",
            success=False,
            failed_nodes=["n1"],
            errors={"n1": "bash failed"},
        )
        evaluator = Evaluator()
        result = evaluator.evaluate(failed_result, hypothesis_id="h1")

        assert not result.success
        assert result.composite_score < 0.5
        # Должна быть feedback про ошибку
        assert any("n1" in w for w in result.feedback.get("weaknesses", []))

    def test_evaluate_empty_output(self):
        empty_result = ExecutionResult(
            run_id="r",
            success=True,
            final_output={},
        )
        evaluator = Evaluator()
        result = evaluator.evaluate(empty_result)

        assert result.success
        # Output completeness should be 0
        completeness_metric = next((m for m in result.metrics if m.name == "output_completeness"), None)
        assert completeness_metric is not None
        assert completeness_metric.value == 0.0
        # Content quality should be 0 too
        content_metric = next((m for m in result.metrics if m.name == "output_content_quality"), None)
        assert content_metric is not None
        assert content_metric.value == 0.0

    def test_evaluate_content_quality_empty_dict(self):
        """Главный тест: pipeline вернул {} — content_quality должен быть низким."""
        result_with_empty = ExecutionResult(
            run_id="r",
            success=True,
            final_output={},
        )
        evaluator = Evaluator()
        result = evaluator.evaluate(result_with_empty)

        content_metric = next((m for m in result.metrics if m.name == "output_content_quality"), None)
        assert content_metric is not None
        assert content_metric.value < 0.3  # ниже порога
        assert not content_metric.passed_threshold
        # Composite должен быть низким из-за провала content_quality
        assert result.composite_score <= 0.5
        # Threshold должен блокировать accept
        assert result.feedback.get("threshold_blocked") is True

    def test_evaluate_content_quality_meaningful(self):
        """Pipeline вернул содержательный output — content_quality высокий."""
        result_with_data = ExecutionResult(
            run_id="r",
            success=True,
            final_output={
                "professional": "Здравствуйте, уважаемые коллеги!",
                "casual": "Привет, народ!",
                "creative": "Йо, челленджеры и творцы!",
            },
        )
        evaluator = Evaluator()
        result = evaluator.evaluate(result_with_data)

        content_metric = next((m for m in result.metrics if m.name == "output_content_quality"), None)
        assert content_metric is not None
        assert content_metric.value > 0.5
        assert content_metric.passed_threshold

    def test_evaluate_content_quality_only_meta_fields(self):
        """Output содержит только служебные поля (cost, tokens) — content_quality низкий."""
        result_meta_only = ExecutionResult(
            run_id="r",
            success=True,
            final_output={
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.0,
                "latency_ms": 1000,
            },
        )
        evaluator = Evaluator()
        result = evaluator.evaluate(result_meta_only)

        content_metric = next((m for m in result.metrics if m.name == "output_content_quality"), None)
        assert content_metric is not None
        # Только служебные поля = нет meaningful данных
        assert content_metric.value < 0.3

    def test_evaluate_with_high_cost(self):
        expensive_result = ExecutionResult(
            run_id="r",
            success=True,
            final_output={"a": 1, "b": 2, "c": 3},
            total_cost_usd=0.60,  # > $0.50 → cost_efficiency = 0
        )
        evaluator = Evaluator()
        result = evaluator.evaluate(expensive_result)

        cost_metric = next((m for m in result.metrics if m.name == "cost_efficiency"), None)
        assert cost_metric is not None
        assert cost_metric.value == 0.0

    def test_feedback_generation(self, sample_execution_result):
        evaluator = Evaluator()
        result = evaluator.evaluate(sample_execution_result)

        assert "weaknesses" in result.feedback
        assert "suggestions_for_next_iteration" in result.feedback
        assert "next_action_recommended" in result.feedback


# ─── SkillLibrary ──────────────────────────────────────────


class TestSkillLibrary:
    @pytest.fixture
    def build_result(self, sample_hypothesis):
        return BuildResult(
            hypothesis_id="h1",
            dag={
                "nodes": [{"id": "n1", "type": "bash", "command": "echo hi"}],
                "edges": [],
                "entry": "n1",
                "exit": "n1",
                "prompts": {"extractor": {"text": "extract", "version": "v1"}},
            },
            success=True,
            metadata={"build_time_sec": 1},
        )

    @pytest.fixture
    def eval_result(self):
        return EvaluationResult(
            run_id="r1",
            hypothesis_id="h1",
            success=True,
            composite_score=0.85,
            metrics=[],
            feedback={"weaknesses": [], "suggestions": []},
        )

    def test_save_and_load(self, tmp_path, build_result, sample_execution_result, eval_result):
        writer = SkillLibraryWriter(tmp_path / "skills")
        skill_dir = writer.save(
            skill_id="test_skill_v1",
            task_description="Test task",
            build_result=build_result,
            execution_result=sample_execution_result,
            evaluation_result=eval_result,
        )

        assert skill_dir.exists()
        assert (skill_dir / "skill.yaml").exists()
        assert (skill_dir / "pipeline.json").exists()
        assert (skill_dir / "prompts" / "extractor.md").exists()

        # Reader
        reader = SkillLibraryReader(tmp_path / "skills")
        skills = reader.list_skills()
        assert len(skills) == 1
        assert skills[0]["skill_id"] == "test_skill_v1"

        loaded = reader.load_skill("test_skill_v1")
        assert loaded is not None
        assert loaded.task_description == "Test task"
        assert len(loaded.pipeline["nodes"]) == 1

    def test_find_similar(self, tmp_path, build_result, sample_execution_result, eval_result):
        writer = SkillLibraryWriter(tmp_path / "skills")
        writer.save(
            skill_id="pdf_extract_v1",
            task_description="Извлечение задач из PDF",
            build_result=build_result,
            execution_result=sample_execution_result,
            evaluation_result=eval_result,
        )
        writer.save(
            skill_id="poem_gen_v1",
            task_description="Генерация стихов на тему",
            build_result=build_result,
            execution_result=sample_execution_result,
            evaluation_result=eval_result,
        )

        reader = SkillLibraryReader(tmp_path / "skills")
        # Ищем похожее на PDF-задачу
        similar = reader.find_similar("извлечение текста из PDF документа")
        assert len(similar) >= 1
        # PDF-skill должен быть первым (больше совпадений)
        assert similar[0]["skill_id"] == "pdf_extract_v1"


# ─── ResearchOrchestrator ──────────────────────────────────


class TestResearchOrchestrator:
    def test_init(self, mock_llm, tmp_path):
        orch = ResearchOrchestrator(
            work_dir=tmp_path / "work",
            llm_provider=mock_llm,
            skills_dir=tmp_path / "skills",
        )
        assert orch.max_iterations == 3
        assert orch.target_score == 0.85

    def test_run_success_first_iteration(self, mock_llm, tmp_path):
        # Мокаем: hypothesis gen → 1 гипотеза, builder → простой DAG, executor → успех
        # Двойной mock: первый вызов = hypothesis gen, второй = builder
        hypothesis_response = Response(
            content=json.dumps({
                "hypotheses": [{
                    "id": "h1",
                    "title": "Test",
                    "rationale": "R",
                    "approach": ["step1"],
                    "model_assignments": [],
                    "custom_tools_needed": [],
                    "estimated": {},
                    "risks": [],
                }]
            }),
            provider="mock",
            model="mock",
            input_tokens=100,
            output_tokens=200,
        )
        dag = {
            "nodes": [{"id": "n1", "type": "bash", "command": "echo hello", "timeout_sec": 5}],
            "edges": [],
            "entry": "n1",
            "exit": "n1",
        }
        builder_response = Response(
            content=json.dumps(dag),
            provider="mock",
            model="mock",
            input_tokens=100,
            output_tokens=300,
        )
        mock_llm.chat.side_effect = [hypothesis_response, builder_response]

        orch = ResearchOrchestrator(
            work_dir=tmp_path / "work",
            llm_provider=mock_llm,
            skills_dir=tmp_path / "skills",
            max_iterations=1,
            target_score=0.5,  # низкий, чтобы прошёл с первого раза
        )
        result = orch.run(
            task_description="Test task",
            task_id="t1",
            input_vars={},
        )

        assert result.iterations_run == 1
        # Execution должен пройти (echo hello)
        # Score должен быть высоким (success, low cost, low latency)
        assert result.best_score > 0.5
        assert result.skill_saved

    def test_run_build_failure(self, mock_llm, tmp_path):
        # Hypothesis OK, но Builder возвращает невалидный DAG
        hypothesis_response = Response(
            content=json.dumps({
                "hypotheses": [{
                    "id": "h1", "title": "T", "rationale": "R",
                    "approach": [], "model_assignments": [],
                    "custom_tools_needed": [], "estimated": {}, "risks": [],
                }]
            }),
            provider="mock",
            model="mock",
            input_tokens=10,
            output_tokens=50,
        )
        # Невалидный DAG
        bad_response = Response(
            content='{"nodes": []}',  # пустой nodes
            provider="mock",
            model="mock",
            input_tokens=10,
            output_tokens=5,
        )
        mock_llm.chat.side_effect = [hypothesis_response, bad_response]

        orch = ResearchOrchestrator(
            work_dir=tmp_path / "work",
            llm_provider=mock_llm,
            skills_dir=tmp_path / "skills",
            max_iterations=1,
        )
        result = orch.run(task_description="Test", task_id="t1")

        # Build failed → нет skill
        assert result.iterations_run == 1
        assert result.best_score == 0.0
        # Skill всё равно может сохраниться с пустым execution_result? Нет — build_result.success=False
        # Проверим что best_execution_result None
        assert result.best_execution_result is None

    def test_run_multiple_iterations(self, mock_llm, tmp_path):
        # 2 итерации, обе с хорошим результатом, но target недостижимый
        def make_responses(n: int):
            responses = []
            for i in range(n):
                h_resp = Response(
                    content=json.dumps({
                        "hypotheses": [{
                            "id": f"h{i+1}",
                            "title": f"T{i+1}",
                            "rationale": "R",
                            "approach": [],
                            "model_assignments": [],
                            "custom_tools_needed": [],
                            "estimated": {},
                            "risks": [],
                        }]
                    }),
                    provider="mock",
                    model="mock",
                    input_tokens=10,
                    output_tokens=50,
                )
                dag = {
                    "nodes": [{"id": "n1", "type": "bash", "command": "echo hi", "timeout_sec": 5}],
                    "edges": [],
                    "entry": "n1",
                    "exit": "n1",
                }
                b_resp = Response(
                    content=json.dumps(dag),
                    provider="mock",
                    model="mock",
                    input_tokens=10,
                    output_tokens=100,
                )
                responses.extend([h_resp, b_resp])
            return responses

        mock_llm.chat.side_effect = make_responses(2)

        orch = ResearchOrchestrator(
            work_dir=tmp_path / "work",
            llm_provider=mock_llm,
            skills_dir=tmp_path / "skills",
            max_iterations=2,
            target_score=1.5,  # недостижимый → обе итерации (max composite = 1.0)
        )
        result = orch.run(task_description="Test", task_id="t1")

        assert result.iterations_run == 2
        assert len(result.history) == 2
        assert result.total_cost_usd >= 0

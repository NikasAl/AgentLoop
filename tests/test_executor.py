"""Tests for Pipeline Executor."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agentloop.executor import PipelineExecutor, PipelineState
from agentloop.executor.nodes import (
    BaseNode,
    BashNode,
    FileNode,
    GateNode,
    LLMNode,
    LoopNode,
    NodeFactory,
    NodeResult,
    PythonNode,
)
from agentloop.executor.state import VariableResolver


# ─── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def work_dir(tmp_path):
    wd = tmp_path / "pipeline_run"
    wd.mkdir(parents=True, exist_ok=True)
    return wd


@pytest.fixture
def state(work_dir):
    return PipelineState(work_dir=work_dir, task_id="test", hypothesis_id="h_test")


# ─── VariableResolver ──────────────────────────────────────


class TestVariableResolver:
    def test_resolve_dollar_var(self, state):
        state.set_var("INPUT", "/data/file.pdf")
        resolver = state.resolver()
        assert resolver.resolve("$INPUT") == "/data/file.pdf"

    def test_resolve_node_output(self, state):
        state.set_output("node1", {"result": 42, "items": [1, 2, 3]})
        resolver = state.resolver()
        assert resolver.resolve("{node1.output.result}") == 42
        assert resolver.resolve("{node1.output.items}") == [1, 2, 3]

    def test_resolve_with_length_filter(self, state):
        state.set_output("node1", {"items": [1, 2, 3]})
        resolver = state.resolver()
        assert resolver.resolve("{node1.output.items|length}") == 3

    def test_resolve_with_is_empty_filter(self, state):
        state.set_output("node1", {"items": []})
        resolver = state.resolver()
        assert resolver.resolve("{node1.output.items|is_empty}") is True

    def test_resolve_dict(self, state):
        state.set_var("NAME", "test")
        resolver = state.resolver()
        d = {"a": "$NAME", "b": "literal"}
        resolved = resolver.resolve(d)
        assert resolved == {"a": "test", "b": "literal"}

    def test_resolve_list(self, state):
        state.set_var("NAME", "test")
        resolver = state.resolver()
        lst = ["$NAME", "literal"]
        resolved = resolver.resolve(lst)
        assert resolved == ["test", "literal"]

    def test_resolve_condition_equals(self, state):
        state.set_output("n", {"flag": True, "count": 5})
        resolver = state.resolver()
        assert resolver.resolve_condition("{n.output.flag} == true")
        assert resolver.resolve_condition("{n.output.count} == 5")
        assert not resolver.resolve_condition("{n.output.count} == 10")

    def test_resolve_condition_and(self, state):
        state.set_output("n", {"flag": True, "count": 5})
        resolver = state.resolver()
        assert resolver.resolve_condition("{n.output.flag} == true && {n.output.count} > 3")
        assert not resolver.resolve_condition("{n.output.flag} == true && {n.output.count} > 10")

    def test_resolve_condition_or(self, state):
        state.set_output("n", {"count": 5})
        resolver = state.resolver()
        assert resolver.resolve_condition("{n.output.count} > 10 || {n.output.count} == 5")

    def test_resolve_condition_with_length_filter(self, state):
        state.set_output("n", {"items": [1, 2, 3]})
        resolver = state.resolver()
        assert resolver.resolve_condition("{n.output.items|length} > 2")
        assert resolver.resolve_condition("{n.output.items|length} == 3")


# ─── PipelineState ─────────────────────────────────────────


class TestPipelineState:
    def test_init_creates_work_dir(self, tmp_path):
        wd = tmp_path / "new_dir"
        state = PipelineState(work_dir=wd)
        assert wd.exists()

    def test_default_runtime_vars(self, state):
        assert "WORKDIR" in state.runtime_vars
        assert "RUN_ID" in state.runtime_vars
        assert "RUN_TIMESTAMP" in state.runtime_vars

    def test_set_get_output(self, state):
        state.set_output("n1", {"x": 1})
        assert state.get_output("n1") == {"x": 1}

    def test_output_saved_to_file(self, state):
        state.set_output("n1", {"x": 1})
        out_file = state.work_dir / "node_n1_output.json"
        assert out_file.exists()
        saved = json.loads(out_file.read_text())
        assert saved == {"x": 1}

    def test_set_input(self, state):
        state.set_input("/data/file.pdf")
        assert state.runtime_vars["INPUT"] == "/data/file.pdf"

    def test_snapshot(self, state):
        state.set_output("n1", {"x": 1})
        snap = state.snapshot()
        assert "n1" in snap["outputs"]
        assert "n1" in snap["completed_nodes"]


# ─── NodeFactory ───────────────────────────────────────────


class TestNodeFactory:
    def test_create_bash_node(self):
        d = {"id": "n1", "type": "bash", "command": "echo hello", "timeout_sec": 5}
        node = NodeFactory.from_dict(d)
        assert isinstance(node, BashNode)
        assert node.node_id == "n1"

    def test_create_llm_node(self):
        d = {"id": "n1", "type": "llm", "model": "local:gemma-4-26b", "user_prompt_template": "Hi"}
        node = NodeFactory.from_dict(d)
        assert isinstance(node, LLMNode)
        assert node.model == "local:gemma-4-26b"

    def test_create_python_node(self):
        d = {"id": "n1", "type": "python", "script_ref": "core:json_merge", "input": {"a": 1}}
        node = NodeFactory.from_dict(d)
        assert isinstance(node, PythonNode)

    def test_create_file_node(self):
        d = {"id": "n1", "type": "file", "operation": "write", "path": "/tmp/x.txt", "content": "hi"}
        node = NodeFactory.from_dict(d)
        assert isinstance(node, FileNode)

    def test_create_loop_node(self):
        d = {
            "id": "loop1",
            "type": "loop",
            "body": [{"id": "inner", "type": "bash", "command": "echo iter"}],
            "exit_condition": "true",
            "max_iterations": 2,
        }
        node = NodeFactory.from_dict(d)
        assert isinstance(node, LoopNode)
        assert len(node.body) == 1

    def test_create_gate_node(self):
        d = {"id": "g1", "type": "gate", "gate_kind": "human_approval", "prompt_template": "Approve?"}
        node = NodeFactory.from_dict(d)
        assert isinstance(node, GateNode)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown node type"):
            NodeFactory.from_dict({"id": "n", "type": "unknown"})


# ─── BashNode ──────────────────────────────────────────────


class TestBashNode:
    def test_echo_success(self, state):
        node = BashNode(node_id="n1", command="echo hello", timeout_sec=5)
        result = node.execute(state)
        assert result.success
        assert "hello" in result.output["stdout"]
        assert result.output["exit_code"] == 0

    def test_failing_command(self, state):
        node = BashNode(node_id="n1", command="exit 1", timeout_sec=5)
        result = node.execute(state)
        assert not result.success
        assert result.output["exit_code"] == 1

    def test_command_with_var(self, state):
        state.set_var("MSG", "test123")
        node = BashNode(node_id="n1", command="echo $MSG", timeout_sec=5)
        result = node.execute(state)
        assert result.success
        assert "test123" in result.output["stdout"]

    def test_command_with_node_ref(self, state):
        state.set_output("prev", {"path": "/tmp"})
        node = BashNode(node_id="n1", command="ls {prev.output.path}", timeout_sec=5)
        result = node.execute(state)
        assert result.success

    def test_timeout(self, state):
        node = BashNode(node_id="n1", command="sleep 10", timeout_sec=1)
        result = node.execute(state)
        assert not result.success
        assert "Timeout" in (result.error or "")

    def test_output_pattern_files(self, state, tmp_path):
        # Создаём пару файлов
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        node = BashNode(
            node_id="n1",
            command="ls",
            output_pattern=f"{tmp_path}/*.txt",
            output_list_as="files",
            timeout_sec=5,
        )
        result = node.execute(state)
        assert result.success
        assert len(result.output["files"]) == 2

    def test_condition_skip(self, state):
        state.set_output("prev", {"should_run": False})
        node = BashNode(
            node_id="n1",
            command="echo should_not_run",
            condition="{prev.output.should_run} == true",
            timeout_sec=5,
        )
        result = node.execute(state)
        assert result.skipped
        assert "should_not_run" not in state.outputs.get("n1", {}).get("stdout", "")

    def test_retry_on_failure(self, state):
        # Команда, которая всегда падает
        node = BashNode(
            node_id="n1",
            command="exit 1",
            timeout_sec=2,
            max_retries=2,
            fallback_on_failure="error",
        )
        result = node.execute(state)
        assert not result.success
        assert result.retries_used == 2

    def test_skip_and_log_fallback(self, state):
        node = BashNode(
            node_id="n1",
            command="exit 1",
            timeout_sec=2,
            fallback_on_failure="skip_and_log",
            default_output={"default": True},
        )
        result = node.execute(state)
        assert result.success  # soft success
        assert result.output == {"default": True}


# ─── FileNode ──────────────────────────────────────────────


class TestFileNode:
    def test_write_and_read(self, state, tmp_path):
        # Write
        path = tmp_path / "test.txt"
        write_node = FileNode(
            node_id="write",
            operation="write",
            path=str(path),
            content="hello world",
        )
        result = write_node.execute(state)
        assert result.success
        assert path.read_text() == "hello world"

        # Read
        read_node = FileNode(
            node_id="read",
            operation="read",
            path=str(path),
        )
        result = read_node.execute(state)
        assert result.success
        assert result.output["content"] == "hello world"

    def test_write_json(self, state, tmp_path):
        path = tmp_path / "out.json"
        node = FileNode(
            node_id="n1",
            operation="write",
            path=str(path),
            content='{"key": "value"}',
            format="json",
        )
        result = node.execute(state)
        assert result.success

    def test_append(self, state, tmp_path):
        path = tmp_path / "log.txt"
        # Дважды append
        for i in range(2):
            node = FileNode(
                node_id=f"append_{i}",
                operation="append",
                path=str(path),
                content=f"line{i}",
            )
            node.execute(state)
        content = path.read_text()
        assert "line0" in content
        assert "line1" in content

    def test_exists(self, state, tmp_path):
        path = tmp_path / "exists.txt"
        path.write_text("hi")

        node = FileNode(node_id="n1", operation="exists", path=str(path))
        result = node.execute(state)
        assert result.success
        assert result.output["exists"] is True

        # Несуществующий
        node2 = FileNode(node_id="n2", operation="exists", path=str(tmp_path / "no.txt"))
        result2 = node2.execute(state)
        assert result2.output["exists"] is False

    def test_list(self, state, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        node = FileNode(
            node_id="n1",
            operation="list",
            pattern=f"{tmp_path}/*.txt",
        )
        result = node.execute(state)
        assert result.success
        assert result.output["count"] == 2

    def test_write_batch(self, state, tmp_path):
        # Подготовим collection в state
        state.set_output("gen", {"poems": [
            {"variant_id": "v1", "text": "poem1"},
            {"variant_id": "v2", "text": "poem2"},
        ]})
        node = FileNode(
            node_id="n1",
            operation="write_batch",
            from_collection="{gen.output.poems}",
            path_template=f"{tmp_path}/poems/poem_{{variant_id}}.txt",
            content_field="text",
        )
        result = node.execute(state)
        assert result.success
        assert result.output["count"] == 2
        assert (tmp_path / "poems" / "poem_v1.txt").read_text() == "poem1"

    def test_content_from(self, state, tmp_path):
        state.set_output("prev", {"text": "from_prev"})
        path = tmp_path / "out.txt"
        node = FileNode(
            node_id="n1",
            operation="write",
            path=str(path),
            content_from="{prev.output.text}",
        )
        result = node.execute(state)
        assert result.success
        assert path.read_text() == "from_prev"


# ─── PythonNode ────────────────────────────────────────────


class TestPythonNode:
    def test_core_json_merge(self, state):
        node = PythonNode(
            node_id="n1",
            script_ref="core:json_merge",
            input_data={"a": {"x": 1}, "b": {"y": 2}},
        )
        result = node.execute(state)
        assert result.success
        assert result.output["a"] == {"x": 1}
        assert result.output["b"] == {"y": 2}

    def test_custom_script(self, state, tmp_path):
        # Создаём custom script
        custom_dir = Path.home() / ".agentloop" / "custom_tools"
        custom_dir.mkdir(parents=True, exist_ok=True)
        script = custom_dir / "test_adder_v1.py"
        script.write_text(
            "def main(input_data):\n"
            "    return {'sum': input_data['a'] + input_data['b']}\n"
        )
        try:
            node = PythonNode(
                node_id="n1",
                script_ref="custom:test_adder_v1",
                input_data={"a": 3, "b": 4},
            )
            result = node.execute(state)
            assert result.success
            assert result.output["sum"] == 7
        finally:
            if script.exists():
                script.unlink()

    def test_script_returns_error(self, state, tmp_path):
        # Создаём script, который возвращает {"error": "..."}
        custom_dir = Path.home() / ".agentloop" / "custom_tools"
        custom_dir.mkdir(parents=True, exist_ok=True)
        script = custom_dir / "test_failer_v1.py"
        script.write_text(
            "def main(input_data):\n"
            "    return {'error': 'something went wrong'}\n"
        )
        try:
            node = PythonNode(
                node_id="n1",
                script_ref="custom:test_failer_v1",
                input_data={},
            )
            result = node.execute(state)
            assert not result.success
            assert "something went wrong" in (result.error or "")
        finally:
            if script.exists():
                script.unlink()

    def test_missing_script(self, state):
        node = PythonNode(
            node_id="n1",
            script_ref="custom:nonexistent_script_v999",
            input_data={},
        )
        result = node.execute(state)
        assert not result.success


# ─── GateNode ──────────────────────────────────────────────


class TestGateNode:
    def test_auto_mode_approve(self, state):
        node = GateNode(
            node_id="g1",
            gate_kind="human_approval",
            prompt_template="Auto approve?",
            hitl_mode="auto",
        )
        result = node.execute(state)
        assert result.success
        assert result.output["decision"] == "approve"
        assert result.output["auto"] is True

    def test_manual_approve(self, state, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda *a: "approve")
        node = GateNode(
            node_id="g1",
            gate_kind="human_approval",
            prompt_template="Approve?",
            hitl_mode="manual",
        )
        result = node.execute(state)
        assert result.success
        assert result.output["decision"] == "approve"

    def test_manual_reject(self, state, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda *a: "reject")
        node = GateNode(
            node_id="g1",
            gate_kind="human_approval",
            prompt_template="Reject?",
            hitl_mode="manual",
            on_reject={"action": "abort"},
        )
        result = node.execute(state)
        assert not result.success
        assert result.output["decision"] == "reject"


# ─── LoopNode ──────────────────────────────────────────────


class TestLoopNode:
    def test_loop_exits_on_condition(self, state, tmp_path):
        # Sub-graph: writes counter to file, increments
        counter_file = tmp_path / "counter.txt"
        counter_file.write_text("0")

        body = [
            {
                "id": "increment",
                "type": "bash",
                "command": f"echo $(($(cat {counter_file}) + 1)) > {counter_file}",
                "timeout_sec": 5,
            },
            {
                "id": "check",
                "type": "bash",
                "command": f"cat {counter_file}",
                "timeout_sec": 5,
            },
        ]
        node = LoopNode(
            node_id="loop1",
            body=body,
            exit_condition="{check.output.stdout|contains:'3'}",
            max_iterations=5,
        )
        result = node.execute(state)
        assert result.success
        assert result.metadata["iterations"] == 3

    def test_loop_max_iterations(self, state):
        # Условие никогда не выполняется
        body = [
            {"id": "echo", "type": "bash", "command": "echo 0", "timeout_sec": 5}
        ]
        node = LoopNode(
            node_id="loop1",
            body=body,
            exit_condition="false",  # never true
            max_iterations=3,
            on_max_iterations="continue_with_warning",
        )
        result = node.execute(state)
        assert result.success
        assert result.metadata["iterations"] == 3
        assert "warning" in result.metadata


# ─── PipelineExecutor (end-to-end) ─────────────────────────


class TestPipelineExecutor:
    def test_simple_linear_pipeline(self, work_dir):
        dag = {
            "nodes": [
                {"id": "n1", "type": "bash", "command": "echo hello", "timeout_sec": 5},
                {"id": "n2", "type": "bash", "command": "echo world", "timeout_sec": 5},
            ],
            "edges": [{"from": "n1", "to": "n2"}],
            "entry": "n1",
            "exit": "n2",
        }
        executor = PipelineExecutor(work_dir=work_dir, checkpoint_enabled=False)
        result = executor.execute(dag_dict=dag, task_id="test")
        assert result.success
        assert "n1" in result.completed_nodes
        assert "n2" in result.completed_nodes
        assert "world" in result.final_output.get("stdout", "")

    def test_pipeline_with_var_passing(self, work_dir):
        dag = {
            "nodes": [
                {"id": "n1", "type": "bash", "command": "echo 'test_data'", "timeout_sec": 5},
                {
                    "id": "n2",
                    "type": "bash",
                    "command": "echo {n1.output.stdout}",
                    "timeout_sec": 5,
                },
            ],
            "edges": [{"from": "n1", "to": "n2"}],
            "entry": "n1",
            "exit": "n2",
        }
        executor = PipelineExecutor(work_dir=work_dir, checkpoint_enabled=False)
        result = executor.execute(dag_dict=dag)
        assert result.success
        assert "test_data" in result.final_output.get("stdout", "")

    def test_pipeline_with_input_vars(self, work_dir, tmp_path):
        msg_file = tmp_path / "msg.txt"
        msg_file.write_text("dynamic_message")
        dag = {
            "nodes": [
                {"id": "n1", "type": "bash", "command": "cat $MSG_FILE", "timeout_sec": 5},
            ],
            "edges": [],
            "entry": "n1",
            "exit": "n1",
        }
        executor = PipelineExecutor(work_dir=work_dir, checkpoint_enabled=False)
        result = executor.execute(
            dag_dict=dag,
            input_vars={"MSG_FILE": str(msg_file)},
        )
        assert result.success
        assert "dynamic_message" in result.final_output.get("stdout", "")

    def test_pipeline_failure(self, work_dir):
        dag = {
            "nodes": [
                {"id": "n1", "type": "bash", "command": "exit 1", "timeout_sec": 5},
                {"id": "n2", "type": "bash", "command": "echo unreachable", "timeout_sec": 5},
            ],
            "edges": [{"from": "n1", "to": "n2"}],
            "entry": "n1",
            "exit": "n2",
        }
        executor = PipelineExecutor(work_dir=work_dir, checkpoint_enabled=False)
        result = executor.execute(dag_dict=dag)
        assert not result.success
        assert "n1" in result.failed_nodes

    def test_pipeline_with_skip_and_log(self, work_dir):
        dag = {
            "nodes": [
                {
                    "id": "n1",
                    "type": "bash",
                    "command": "exit 1",
                    "timeout_sec": 5,
                    "fallback_on_failure": "skip_and_log",
                    "default_output": {"stdout": "fallback", "exit_code": 0},
                },
                {
                    "id": "n2",
                    "type": "bash",
                    "command": "echo {n1.output.stdout}",
                    "timeout_sec": 5,
                },
            ],
            "edges": [{"from": "n1", "to": "n2"}],
            "entry": "n1",
            "exit": "n2",
        }
        executor = PipelineExecutor(work_dir=work_dir, checkpoint_enabled=False)
        result = executor.execute(dag_dict=dag)
        assert result.success  # pipeline продолжается благодаря fallback
        assert "fallback" in result.final_output.get("stdout", "")

    def test_pipeline_with_gate_auto(self, work_dir):
        dag = {
            "nodes": [
                {"id": "n1", "type": "bash", "command": "echo data", "timeout_sec": 5},
                {
                    "id": "g1",
                    "type": "gate",
                    "gate_kind": "human_approval",
                    "prompt_template": "Approve?",
                    "hitl_mode": "auto",
                },
                {"id": "n2", "type": "bash", "command": "echo done", "timeout_sec": 5},
            ],
            "edges": [
                {"from": "n1", "to": "g1"},
                {"from": "g1", "to": "n2"},
            ],
            "entry": "n1",
            "exit": "n2",
        }
        executor = PipelineExecutor(work_dir=work_dir, checkpoint_enabled=False)
        result = executor.execute(dag_dict=dag)
        assert result.success
        assert "done" in result.final_output.get("stdout", "")

    def test_pipeline_with_python_node(self, work_dir):
        dag = {
            "nodes": [
                {
                    "id": "n1",
                    "type": "python",
                    "script_ref": "core:json_merge",
                    "input": {"a": {"x": 1}, "b": {"y": 2}},
                },
                {
                    "id": "n2",
                    "type": "bash",
                    "command": "echo {n1.output.a}",
                    "timeout_sec": 5,
                },
            ],
            "edges": [{"from": "n1", "to": "n2"}],
            "entry": "n1",
            "exit": "n2",
        }
        executor = PipelineExecutor(work_dir=work_dir, checkpoint_enabled=False)
        result = executor.execute(dag_dict=dag)
        assert result.success

    def test_pipeline_with_file_ops(self, work_dir, tmp_path):
        out_path = tmp_path / "result.txt"
        dag = {
            "nodes": [
                {
                    "id": "n1",
                    "type": "file",
                    "operation": "write",
                    "path": str(out_path),
                    "content": "test content",
                },
                {
                    "id": "n2",
                    "type": "file",
                    "operation": "read",
                    "path": str(out_path),
                },
            ],
            "edges": [{"from": "n1", "to": "n2"}],
            "entry": "n1",
            "exit": "n2",
        }
        executor = PipelineExecutor(work_dir=work_dir, checkpoint_enabled=False)
        result = executor.execute(dag_dict=dag)
        assert result.success
        assert result.final_output["content"] == "test content"

    def test_checkpoint_save_and_resume(self, work_dir):
        dag = {
            "nodes": [
                {"id": "n1", "type": "bash", "command": "echo step1", "timeout_sec": 5},
                {"id": "n2", "type": "bash", "command": "echo step2", "timeout_sec": 5},
            ],
            "edges": [{"from": "n1", "to": "n2"}],
            "entry": "n1",
            "exit": "n2",
        }

        # Первый запуск с checkpoint
        executor = PipelineExecutor(work_dir=work_dir, checkpoint_enabled=True)
        result = executor.execute(dag_dict=dag, task_id="test")
        assert result.success
        # Checkpoint должен быть удалён после успеха
        assert not (work_dir / "checkpoint.json").exists()

    def test_topological_sort(self, work_dir):
        executor = PipelineExecutor(work_dir=work_dir)
        dag = {
            "nodes": [
                {"id": "a"},
                {"id": "b"},
                {"id": "c"},
            ],
            "edges": [
                {"from": "a", "to": "b"},
                {"from": "b", "to": "c"},
            ],
        }
        order = executor._topological_sort(dag)
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

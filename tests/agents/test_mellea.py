import json
import sys
from pathlib import Path

import litellm
import pytest
import yaml

pytest.importorskip("mellea")

from mellea.backends.dummy import DummyBackend  # noqa: E402
from mellea.core import GenerateLog, ModelOutputThunk  # noqa: E402
from mellea.stdlib.components.genstub import GenerativeStub  # noqa: E402

from minisweagent.agents import get_agent  # noqa: E402
from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.agents.mellea import MelleaAgent  # noqa: E402
from minisweagent.environments.local import LocalEnvironment  # noqa: E402
from minisweagent.models.test_models import DeterministicModel  # noqa: E402
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager  # noqa: E402
from minisweagent.run.benchmarks.utils.common import (  # noqa: E402
    ProgressTrackingAgent,
    get_progress_tracking_agent_class,
)
from minisweagent.utils.serialize import recursive_merge  # noqa: E402


class FakeBackend(DummyBackend):
    """Returns predetermined raw strings and parses them like a real backend, recording each generation's action."""

    def __init__(self, responses: list[str], usage: dict | None = None):
        super().__init__(responses)
        self.usage = usage
        self.actions: list = []

    @property
    def histories(self) -> list[str]:
        """The `history` argument of every `next_action` generation."""
        return [a._arguments.value for a in self.actions if isinstance(a, GenerativeStub)]

    async def _generate_from_context(self, action, ctx, *, format=None, model_options=None, tool_calls=False):
        self.actions.append(action)
        mot = ModelOutputThunk(value=self.responses[self.idx])
        self.idx += 1
        mot._generate_log = GenerateLog()
        mot.generation.usage = self.usage
        mot.parsed_repr = action.parse(mot)
        return mot, ctx.add(action).add(mot)


def decision(action: str, command: str = "", thought: str = "thinking") -> str:
    return json.dumps({"result": {"thought": thought, "action": action, "command": command}})


@pytest.fixture
def mellea_config():
    return yaml.safe_load(Path("src/minisweagent/config/mellea.yaml").read_text())["agent"]


@pytest.fixture
def swebench_mellea_config():
    """swebench.yaml with the swebench_mellea.yaml overlay, as with `-c swebench.yaml -c swebench_mellea.yaml`."""
    configs = [
        yaml.safe_load(Path(f"src/minisweagent/config/benchmarks/{n}.yaml").read_text())
        for n in ("swebench", "swebench_mellea")
    ]
    return recursive_merge(*configs)


def make_agent(
    responses: list[str], config: dict, usage: dict | None = None, env: LocalEnvironment | None = None, **kwargs
) -> tuple[MelleaAgent, FakeBackend]:
    """`get_agent` deep-copies the config, so the backend must be taken from the agent."""
    agent = get_agent(
        DeterministicModel(outputs=[]),
        env or LocalEnvironment(),
        {**config, **kwargs, "backend": FakeBackend(responses, usage)},
    )
    return agent, agent.backend


def test_bash_decision_reaches_environment(mellea_config, tmp_path):
    agent, backend = make_agent(
        [decision("bash", f"cd {tmp_path} && echo hi > out.txt && cat out.txt"), decision("finish")], mellea_config
    )
    assert isinstance(agent, MelleaAgent)
    assert agent.run("Write hi to out.txt") == {"exit_status": "Submitted", "submission": ""}
    assert (tmp_path / "out.txt").read_text() == "hi\n"
    assert agent.n_calls == 2
    assert [m["role"] for m in agent.messages] == ["system", "user", "assistant", "user", "assistant", "exit"]
    assert agent.messages[2]["extra"]["actions"] == [{"command": f"cd {tmp_path} && echo hi > out.txt && cat out.txt"}]
    assert "<returncode>0</returncode>" in agent.messages[3]["content"]
    assert "Write hi to out.txt" in backend.histories[0]
    assert "hi\n" in backend.histories[1] and '"action":"bash"' in backend.histories[1]


def test_finish_does_not_execute_command(mellea_config, tmp_path):
    agent, _ = make_agent([decision("finish", f"touch {tmp_path}/marker", thought="all done")], mellea_config)
    assert agent.run("task")["exit_status"] == "Submitted"
    assert not (tmp_path / "marker").exists()
    assert agent.messages[-1]["content"] == "all done"
    assert agent.messages[-2]["extra"]["actions"] == []


@pytest.mark.parametrize(
    ("invalid_response", "expected_error"),
    [
        (json.dumps({"result": {"thought": "t", "action": "run", "command": "touch {marker}"}}), "literal_error"),
        ('{"result": {"thought": "t", "action": "bash", "command": "touch {marker}"', "json_invalid"),
        ("touch {marker}", "json_invalid"),
        (decision("bash", "   "), 'A decision with action "bash" must have a non-empty command.'),
    ],
)
def test_invalid_decision_does_not_reach_environment(mellea_config, tmp_path, invalid_response, expected_error):
    marker = tmp_path / "marker"
    agent, backend = make_agent([invalid_response.replace("{marker}", str(marker)), decision("finish")], mellea_config)
    assert agent.run("task")["exit_status"] == "Submitted"
    assert not marker.exists()
    assert agent.n_calls == 2
    format_errors = [m for m in agent.messages if m.get("extra", {}).get("interrupt_type") == "FormatError"]
    assert len(format_errors) == 1
    assert expected_error in format_errors[0]["content"]
    assert expected_error in backend.histories[1]
    stats = agent.serialize()["info"]["mellea_stats"]
    assert (stats["parse_errors"], stats["rejected_decisions"]) == ((0, 1) if "non-empty" in expected_error else (1, 0))


def test_repeated_format_errors_exit(mellea_config):
    agent, _ = make_agent(["garbage", "garbage", decision("finish")], mellea_config, max_consecutive_format_errors=2)
    assert agent.run("task")["exit_status"] == "RepeatedFormatError"
    assert agent.n_calls == 2


def test_step_limit(mellea_config):
    agent, _ = make_agent([decision("bash", "echo 1"), decision("bash", "echo 2")], mellea_config, step_limit=1)
    assert agent.run("task")["exit_status"] == "LimitsExceeded"
    assert agent.n_calls == 1
    assert agent.serialize()["info"]["config"]["agent_type"] == "minisweagent.agents.mellea.MelleaAgent"


def test_failed_requirement_is_repaired_before_execution(mellea_config, tmp_path):
    agent, backend = make_agent(
        [decision("bash", ""), decision("bash", f"touch {tmp_path}/marker"), decision("finish")],
        mellea_config,
        loop_budget=2,
    )
    assert agent.run("task")["exit_status"] == "Submitted"
    assert (tmp_path / "marker").exists()
    assert agent.n_calls == 2
    assert not any(m.get("extra", {}).get("interrupt_type") == "FormatError" for m in agent.messages)
    assert agent.messages[2]["extra"]["actions"] == [{"command": f"touch {tmp_path}/marker"}]
    assert "must have a non-empty command" in backend.actions[1].content


def test_exhausted_loop_budget_does_not_reach_environment(mellea_config):
    agent, _ = make_agent(
        [decision("bash", ""), decision("bash", " "), decision("finish")], mellea_config, loop_budget=2
    )
    assert agent.run("task")["exit_status"] == "Submitted"
    assert agent.n_calls == 2
    assert [m["role"] for m in agent.messages] == ["system", "user", "assistant", "user", "assistant", "exit"]
    assert "must have a non-empty command" in agent.messages[3]["content"]


def test_semantic_requirement_blocks_finish(mellea_config):
    requirement = "Only finish once the task has been implemented and verified."
    agent, backend = make_agent(
        [decision("finish"), "no", decision("finish"), "yes"], mellea_config, requirements=[requirement]
    )
    assert agent.run("task")["exit_status"] == "Submitted"
    assert agent.n_calls == 2
    assert requirement in agent.messages[3]["content"]
    assert requirement in backend.histories[1]
    assert [type(a).__name__ for a in backend.actions] == ["SyncGenerativeStub", "Requirement"] * 2


def test_diagnostics_record_attempts_failures_and_cost(mellea_config, tmp_path):
    requirement = "Only finish once the task has been verified."
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}
    agent, _ = make_agent(
        [decision("finish", thought="premature"), "no", decision("finish"), "yes"],
        mellea_config,
        usage=usage,
        requirements=[requirement],
        loop_budget=2,
        mellea_model_id="gpt-4o-mini",
        output_path=tmp_path / "traj.json",
    )
    assert agent.run("task")["exit_status"] == "Submitted"
    expected_cost = 4 * sum(litellm.cost_per_token(model="gpt-4o-mini", prompt_tokens=1000, completion_tokens=100))
    assert agent.messages[2]["extra"]["mellea"] == {
        "success": True,
        "attempts": [
            {
                "decision": {"thought": "premature", "action": "finish", "command": ""},
                "failed_requirements": [requirement],
            },
            {"decision": {"thought": "thinking", "action": "finish", "command": ""}, "failed_requirements": []},
        ],
        "lm_calls": 4,
        "prompt_tokens": 4000,
        "completion_tokens": 400,
    }
    assert agent.messages[2]["extra"]["cost"] == pytest.approx(expected_cost) == agent.cost
    info = json.loads((tmp_path / "traj.json").read_text())["info"]
    assert info["model_stats"]["instance_cost"] == pytest.approx(expected_cost)
    assert info["mellea_stats"] == {
        "decisions": 1,
        "attempts": 2,
        "lm_calls": 4,
        "rejected_decisions": 0,
        "parse_errors": 0,
        "requirement_failures": {requirement: 1},
    }


def test_unknown_model_cost(mellea_config):
    usage = {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
    agent, _ = make_agent([decision("finish")], mellea_config, usage=usage, mellea_model_id="no-such-model-xyz")
    with pytest.raises(RuntimeError, match="Cannot calculate cost for no-such-model-xyz"):
        agent.run("task")
    agent, _ = make_agent(
        [decision("finish")],
        mellea_config,
        usage=usage,
        mellea_model_id="no-such-model-xyz",
        cost_tracking="ignore_errors",
    )
    assert agent.run("task")["exit_status"] == "Submitted"
    assert agent.cost == 0.0


def test_smoke_edit_and_test_repository(mellea_config, tmp_path):
    """Scripted inspect -> edit -> test -> finish trajectory on a tiny repository (see test_fire.py for a real model)."""
    (tmp_path / "hello.py").write_text('def hello():\n    return "hello"\n')
    (tmp_path / "test_hello.py").write_text(
        'from hello import hello\n\n\ndef test_hello():\n    assert hello() == "hello world"\n'
    )
    pytest_cmd = f"cd {tmp_path} && {sys.executable} -m pytest -q -p no:cacheprovider test_hello.py"
    agent, _ = make_agent(
        [
            decision("bash", f"cd {tmp_path} && ls && cat hello.py"),
            decision("bash", f"cd {tmp_path} && printf 'def hello():\\n    return \"hello world\"\\n' > hello.py"),
            decision("bash", pytest_cmd),
            decision("finish", thought="hello() returns 'hello world' and the tests pass."),
        ],
        mellea_config,
        output_path=tmp_path / "traj.json",
    )
    assert agent.run('Change hello() to return "hello world" and run the tests.')["exit_status"] == "Submitted"
    assert (tmp_path / "hello.py").read_text() == 'def hello():\n    return "hello world"\n'
    assert "1 passed" in agent.messages[7]["content"]
    assert json.loads((tmp_path / "traj.json").read_text())["info"]["mellea_stats"]["decisions"] == 4


def test_finish_command_submits_through_environment(mellea_config, tmp_path):
    agent, backend = make_agent(
        [
            decision("finish", f"touch {tmp_path}/marker"),
            decision("bash", f"printf 'the patch\\n' > {tmp_path}/patch.txt"),
            decision("finish"),
        ],
        mellea_config,
        finish_command=f"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat {tmp_path}/patch.txt",
    )
    assert agent.run("task") == {"exit_status": "Submitted", "submission": "the patch\n"}
    assert not (tmp_path / "marker").exists()
    assert agent.n_calls == 3
    assert [m["role"] for m in agent.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "exit",
    ]
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in agent.messages[3]["content"]
    assert "No such file" in agent.messages[3]["content"] and "No such file" in backend.histories[1]


def test_swebench_overlay_keeps_baseline_templates_and_submits_patch(swebench_mellea_config, tmp_path):
    baseline = yaml.safe_load(Path("src/minisweagent/config/benchmarks/swebench.yaml").read_text())["agent"]
    config = swebench_mellea_config["agent"]
    assert (config["system_template"], config["instance_template"]) == (
        baseline["system_template"],
        baseline["instance_template"],
    )
    assert config["observation_template"] == swebench_mellea_config["model"]["observation_template"]
    agent, _ = make_agent(
        [
            decision("bash", "python3 -c 'print(\"x\" * 20000)'"),
            "garbage",
            decision("bash", "printf 'diff\\n' > patch.txt"),
            decision("finish"),
        ],
        config,
        env=LocalEnvironment(cwd=str(tmp_path)),
    )
    assert agent.run("Fix the bug")["submission"] == "diff\n"
    assert "10001 characters elided" in agent.messages[3]["content"]
    assert len(agent.messages[3]["content"]) < 12000
    assert "consult the first message about" in agent.messages[4]["content"]


@pytest.mark.parametrize(("spec", "agent_class"), [("default", DefaultAgent), ("mellea", MelleaAgent)])
def test_progress_tracking_agent_class(spec, agent_class):
    cls = get_progress_tracking_agent_class(spec)
    assert issubclass(cls, ProgressTrackingAgent) and issubclass(cls, agent_class)


def test_progress_tracking_mellea_agent_reports_steps(swebench_mellea_config, tmp_path):
    progress_manager = RunBatchProgressManager(1)
    progress_manager.on_instance_start("inst")
    config = {k: v for k, v in swebench_mellea_config["agent"].items() if k != "agent_class"} | {"step_limit": 2}
    agent = get_progress_tracking_agent_class("mellea")(
        DeterministicModel(outputs=[]),
        LocalEnvironment(cwd=str(tmp_path)),
        progress_manager=progress_manager,
        instance_id="inst",
        backend=FakeBackend([decision("bash", "echo hi"), decision("bash", "echo hi")]),
        **config,
    )
    assert agent.run("task")["exit_status"] == "LimitsExceeded"
    assert agent.n_calls == 2
    assert progress_manager._task_progress_bar.tasks[0].fields["status"].strip() == "Step   3 ($0.00)"

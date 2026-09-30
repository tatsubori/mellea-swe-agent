import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("mellea")

from mellea import MelleaSession  # noqa: E402
from mellea.backends.dummy import DummyBackend  # noqa: E402
from mellea.core import GenerateLog, ModelOutputThunk  # noqa: E402

from minisweagent.agents import get_agent  # noqa: E402
from minisweagent.agents.mellea import MelleaAgent  # noqa: E402
from minisweagent.environments.local import LocalEnvironment  # noqa: E402
from minisweagent.models.test_models import DeterministicModel  # noqa: E402


class FakeBackend(DummyBackend):
    """Returns predetermined raw strings and parses them like a real backend, recording each prompt's history."""

    def __init__(self, responses: list[str]):
        super().__init__(responses)
        self.histories: list[str] = []

    async def _generate_from_context(self, action, ctx, *, format=None, model_options=None, tool_calls=False):
        self.histories.append(action._arguments.value)
        mot = ModelOutputThunk(value=self.responses[self.idx])
        self.idx += 1
        mot._generate_log = GenerateLog()
        mot.parsed_repr = action.parse(mot)
        return mot, ctx.add(action).add(mot)


def decision(action: str, command: str = "", thought: str = "thinking") -> str:
    return json.dumps({"result": {"thought": thought, "action": action, "command": command}})


@pytest.fixture
def mellea_config():
    return yaml.safe_load(Path("src/minisweagent/config/mellea.yaml").read_text())["agent"]


def make_agent(responses: list[str], config: dict, **kwargs) -> tuple[MelleaAgent, FakeBackend]:
    """`get_agent` deep-copies the config, so the backend must be taken from the agent's session."""
    config = {**config, **kwargs, "session": MelleaSession(FakeBackend(responses))}
    agent = get_agent(DeterministicModel(outputs=[]), LocalEnvironment(), config)
    return agent, agent.session.backend


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
        (decision("bash", "   "), 'Action "bash" requires a non-empty command.'),
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


def test_repeated_format_errors_exit(mellea_config):
    agent, _ = make_agent(["garbage", "garbage", decision("finish")], mellea_config, max_consecutive_format_errors=2)
    assert agent.run("task")["exit_status"] == "RepeatedFormatError"
    assert agent.n_calls == 2


def test_step_limit(mellea_config):
    agent, _ = make_agent([decision("bash", "echo 1"), decision("bash", "echo 2")], mellea_config, step_limit=1)
    assert agent.run("task")["exit_status"] == "LimitsExceeded"
    assert agent.n_calls == 1
    assert agent.serialize()["info"]["config"]["agent_type"] == "minisweagent.agents.mellea.MelleaAgent"

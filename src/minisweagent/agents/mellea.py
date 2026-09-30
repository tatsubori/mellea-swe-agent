"""Agent whose next-action policy is a Mellea generative program instead of a parsed LM response.
See MELLEA_MINI_SWE_AGENT_PROTOTYPE.md for the motivation. Requires `pip install mini-swe-agent[mellea]`.
"""

import time
from typing import Literal

from jinja2 import StrictUndefined, Template
from mellea import MelleaSession, generative, start_session
from mellea.core import ComponentParseError
from pydantic import BaseModel

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models.utils.actions_text import format_observation_messages


class Decision(BaseModel):
    thought: str
    action: Literal["bash", "finish"]
    command: str = ""


@generative
def next_action(history: str) -> Decision:
    """Decide the single next action of a software engineering agent.

    `history` contains the instructions and the task, followed by all previous decisions and their observations.
    Either run exactly one bash command that makes progress on the task (action="bash"),
    or finish (action="finish", empty command) once the task has been solved and verified.
    """


def render_history(messages: list[dict]) -> str:
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)


class MelleaAgentConfig(AgentConfig):
    mellea_backend: str = "litellm"
    """Backend name passed to `mellea.start_session`."""
    mellea_model_id: str = ""
    """Model id for the Mellea backend. Defaults to the `model_name` of the mini-swe-agent model."""
    mellea_model_options: dict = {}
    """Model options passed to `mellea.start_session`."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    format_error_template: str = "Your previous decision was invalid:\n\n{{error}}\n\nPlease decide again."
    """Template used when a decision cannot be parsed or is structurally invalid."""


class MelleaAgent(DefaultAgent):
    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        session: MelleaSession | None = None,
        config_class: type = MelleaAgentConfig,
        **kwargs,
    ):
        """`model` is only used for message formatting and serialization, all decisions go through `session`."""
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.session = session or start_session(
            self.config.mellea_backend,
            self.config.mellea_model_id or model.config.model_name,
            model_options=self.config.mellea_model_options,
        )

    def _format_error(self, error: str) -> FormatError:
        content = Template(self.config.format_error_template, undefined=StrictUndefined).render(error=error)
        return FormatError({"role": "user", "content": content, "extra": {"interrupt_type": "FormatError"}})

    def query(self) -> dict:
        self.check_limits()
        self.n_calls += 1
        try:
            decision = next_action(self.session, history=render_history(self.messages))
        except ComponentParseError as e:
            raise self._format_error(str(e))
        actions = [{"command": decision.command}] if decision.action == "bash" else []
        message = self.model.format_message(
            role="assistant",
            content=decision.model_dump_json(),
            extra={"decision": decision.model_dump(), "actions": actions, "timestamp": time.time()},
        )
        self.add_messages(message)
        if decision.action == "finish":
            raise Submitted(
                {"role": "exit", "content": decision.thought, "extra": {"exit_status": "Submitted", "submission": ""}}
            )
        if not decision.command.strip():
            raise self._format_error('Action "bash" requires a non-empty command.')
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        outputs = [self.env.execute(action) for action in message["extra"]["actions"]]
        return self.add_messages(
            *format_observation_messages(
                outputs, observation_template=self.config.observation_template, template_vars=self.get_template_vars()
            )
        )

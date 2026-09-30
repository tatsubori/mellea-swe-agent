"""Agent whose next-action policy is a Mellea generative program instead of a parsed LM response.
See MELLEA_MINI_SWE_AGENT_PROTOTYPE.md for the motivation. Requires `pip install mini-swe-agent[mellea]`.
"""

import time
from typing import Literal

from jinja2 import StrictUndefined, Template
from mellea import ChatContext, Requirement, SamplingResult, ValidationResult, generative, start_backend
from mellea.core import Backend, ComponentParseError, Context
from mellea.stdlib.sampling.base import MultiTurnStrategy
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


def _has_command(ctx: Context) -> ValidationResult:
    decision = ctx.last_output().parsed_repr
    return ValidationResult(decision.action == "finish" or bool(decision.command.strip()))


HAS_COMMAND = Requirement('A decision with action "bash" must have a non-empty command.', _has_command)


class _RecordingStrategy(MultiTurnStrategy):
    """Repairs failed attempts with feedback and keeps the `SamplingResult`, which `@generative` does not return."""

    async def sample(self, *args, **kwargs) -> SamplingResult:
        self.result = await super().sample(*args, **kwargs)
        return self.result


class MelleaAgentConfig(AgentConfig):
    mellea_backend: str = "litellm"
    """Backend name passed to `mellea.start_backend`."""
    mellea_model_id: str = ""
    """Model id for the Mellea backend. Defaults to the `model_name` of the mini-swe-agent model."""
    mellea_model_options: dict = {}
    """Model options passed to `mellea.start_backend`."""
    requirements: list[str] = []
    """Semantic requirements that every decision is checked against by the LM (LLM-as-a-judge)."""
    loop_budget: int = 1
    """Maximum generation attempts per decision. Failed attempts are repaired with the failed requirements."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    format_error_template: str = "Your previous decision was invalid:\n\n{{error}}\n\nPlease decide again."
    """Template used when a decision cannot be parsed or violates requirements."""


class MelleaAgent(DefaultAgent):
    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        backend: Backend | None = None,
        config_class: type = MelleaAgentConfig,
        **kwargs,
    ):
        """`model` is only used for message formatting and serialization, all decisions go through `backend`."""
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.backend = (
            backend
            or start_backend(
                self.config.mellea_backend,
                self.config.mellea_model_id or model.config.model_name,
                model_options=self.config.mellea_model_options,
            )[1]
        )

    def _format_error(self, error: str) -> FormatError:
        content = Template(self.config.format_error_template, undefined=StrictUndefined).render(error=error)
        return FormatError({"role": "user", "content": content, "extra": {"interrupt_type": "FormatError"}})

    def query(self) -> dict:
        self.check_limits()
        self.n_calls += 1
        strategy = _RecordingStrategy(loop_budget=self.config.loop_budget)
        try:
            decision, _ = next_action(
                ChatContext(),
                self.backend,
                requirements=[HAS_COMMAND, *map(Requirement, self.config.requirements)],
                strategy=strategy,
                history=render_history(self.messages),
            )
        except ComponentParseError as e:
            raise self._format_error(str(e))
        actions = [{"command": decision.command}] if decision.action == "bash" else []
        message = self.model.format_message(
            role="assistant",
            content=decision.model_dump_json(),
            extra={"decision": decision.model_dump(), "actions": actions, "timestamp": time.time()},
        )
        self.add_messages(message)
        if not strategy.result.success:
            failed = [r.description for r, v in strategy.result.sample_validations[-1] if not v]
            raise self._format_error(
                "The decision violates these requirements:\n" + "\n".join(f"* {f}" for f in failed)
            )
        if decision.action == "finish":
            raise Submitted(
                {"role": "exit", "content": decision.thought, "extra": {"exit_status": "Submitted", "submission": ""}}
            )
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        outputs = [self.env.execute(action) for action in message["extra"]["actions"]]
        return self.add_messages(
            *format_observation_messages(
                outputs, observation_template=self.config.observation_template, template_vars=self.get_template_vars()
            )
        )

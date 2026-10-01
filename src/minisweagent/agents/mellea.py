"""Agent whose next-action policy is a Mellea generative program instead of a parsed LM response.
See MELLEA_MINI_SWE_AGENT_PROTOTYPE.md for the motivation. Requires `pip install mini-swe-agent[mellea]`.
"""

import os
import time
from collections import Counter
from typing import Literal

import litellm
from jinja2 import StrictUndefined, Template
from mellea import ChatContext, Requirement, SamplingResult, ValidationResult, generative, start_backend
from mellea.core import Backend, ComponentParseError, Context
from mellea.stdlib.components.genstub import FunctionResponse
from mellea.stdlib.sampling.base import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_text import format_observation_messages


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thought: str
    action: Literal["bash", "finish"]
    command: str


@generative
def next_action(history: str) -> Decision:
    """Decide the single next action of a software engineering agent.

    `history` contains the instructions and the task, followed by all previous decisions and their observations.
    Either run exactly one bash command that makes progress on the task (action="bash"),
    or finish (action="finish", empty command) once the task has been solved and verified.
    """


class NextActionResponse(FunctionResponse[Decision]):
    model_config = ConfigDict(extra="forbid")
    result: Decision


# Mellea sends its response model as a `strict` JSON schema, which e.g. Azure OpenAI rejects unless every object
# forbids additional properties and no `$ref` has sibling keywords. Mellea's generated wrapper violates both.
next_action._response_model = NextActionResponse


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
    """Model options passed to `mellea.start_backend`, on top of the mini-swe-agent model's `model_kwargs`."""
    requirements: list[str] = []
    """Semantic requirements that every decision is checked against by the LM (LLM-as-a-judge)."""
    loop_budget: int = 1
    """Maximum generation attempts per decision. Failed attempts are repaired with the failed requirements."""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Whether to raise if litellm cannot calculate the cost of the Mellea model."""
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
        self.model_id = self.config.mellea_model_id or model.config.model_name
        self.n_parse_errors = 0
        # Mellea's litellm backend always sets drop_params itself and rejects a second value.
        model_kwargs = {k: v for k, v in getattr(model.config, "model_kwargs", {}).items() if k != "drop_params"}
        self.backend = (
            backend
            or start_backend(
                self.config.mellea_backend,
                self.model_id,
                model_options=model_kwargs | self.config.mellea_model_options,
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
            self.n_parse_errors += 1
            raise self._format_error(str(e))
        diagnostics = self._diagnostics(strategy.result)
        cost = self._cost(diagnostics["prompt_tokens"], diagnostics["completion_tokens"])
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        actions = [{"command": decision.command}] if decision.action == "bash" else []
        message = self.model.format_message(
            role="assistant",
            content=decision.model_dump_json(),
            extra={
                "decision": decision.model_dump(),
                "actions": actions,
                "cost": cost,
                "mellea": diagnostics,
                "timestamp": time.time(),
            },
        )
        self.add_messages(message)
        if not diagnostics["success"]:
            failed = diagnostics["attempts"][-1]["failed_requirements"]
            raise self._format_error(
                "The decision violates these requirements:\n" + "\n".join(f"* {f}" for f in failed)
            )
        if decision.action == "finish":
            raise Submitted(
                {"role": "exit", "content": decision.thought, "extra": {"exit_status": "Submitted", "submission": ""}}
            )
        return message

    @staticmethod
    def _diagnostics(result: SamplingResult) -> dict:
        """Every attempt with its failed requirements, plus token usage of all generations and LM judgements."""
        thunks = [*result.sample_generations, *(v.thunk for vs in result.sample_validations for _, v in vs if v.thunk)]
        usages = [t.generation.usage for t in thunks if t.generation.usage]
        return {
            "success": result.success,
            "attempts": [
                {"decision": g.parsed_repr.model_dump(), "failed_requirements": [r.description for r, v in vs if not v]}
                for g, vs in zip(result.sample_generations, result.sample_validations)
            ],
            "lm_calls": len(thunks),
            "prompt_tokens": sum(u["prompt_tokens"] for u in usages),
            "completion_tokens": sum(u["completion_tokens"] for u in usages),
        }

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        if not prompt_tokens + completion_tokens:
            return 0.0
        try:
            return sum(
                litellm.cost_per_token(
                    model=self.model_id, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
                )
            )
        except Exception as e:
            if self.config.cost_tracking == "ignore_errors":
                return 0.0
            msg = f"Cannot calculate cost for {self.model_id}: {e}. Set agent.cost_tracking or MSWEA_COST_TRACKING to 'ignore_errors'."
            raise RuntimeError(msg) from e

    def serialize(self, *extra_dicts) -> dict:
        stats = [m["extra"]["mellea"] for m in self.messages if "mellea" in m.get("extra", {})]
        summary = {
            "decisions": len(stats),
            "attempts": sum(len(s["attempts"]) for s in stats),
            "lm_calls": sum(s["lm_calls"] for s in stats),
            "rejected_decisions": sum(not s["success"] for s in stats),
            "parse_errors": self.n_parse_errors,
            "requirement_failures": dict(
                Counter(f for s in stats for a in s["attempts"] for f in a["failed_requirements"])
            ),
        }
        return super().serialize({"info": {"mellea_stats": summary}}, *extra_dicts)

    def execute_actions(self, message: dict) -> list[dict]:
        outputs = [self.env.execute(action) for action in message["extra"]["actions"]]
        return self.add_messages(
            *format_observation_messages(
                outputs, observation_template=self.config.observation_template, template_vars=self.get_template_vars()
            )
        )

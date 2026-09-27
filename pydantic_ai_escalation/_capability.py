from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai.capabilities import AbstractCapability, ModelSelector
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.models import Model, ModelRequestContext, ModelSelectionContext
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from typing_extensions import NotRequired, TypedDict

__all__ = ('EscalateOnOutputRetry', 'EscalationBudgetWarning', 'Level', 'LevelSpec')


@dataclass(frozen=True)
class Level:
    """One step of the escalation ladder.

    The model gets `1 + output_retries` attempts. Every attempt after the first sees the
    validation errors from the attempts before it, so `output_retries` is the number of
    feedback retries on this model before the run moves to the next level.
    """

    model: Model | str
    """The model for this level: a `Model` instance or a model name such as `'openai:gpt-4o-mini'`."""

    output_retries: int = 0
    """Feedback retries on this model before escalating. `0` escalates after the first failed attempt."""

    def __post_init__(self) -> None:
        if self.output_retries < 0:
            raise UserError(f'`output_retries` must be 0 or greater, got {self.output_retries}')

    @property
    def attempts(self) -> int:
        """The number of requests this level can make: the first attempt plus its feedback retries."""
        return 1 + self.output_retries


class LevelSpec(TypedDict):
    """The serialized form of a [`Level`][pydantic_ai_escalation.Level] in an agent spec."""

    model: str
    output_retries: NotRequired[int]


_LEVEL_SPEC_KEYS = frozenset(LevelSpec.__annotations__)


class EscalationBudgetWarning(UserWarning):
    """The agent's output retry budget is too small to reach every level."""


@dataclass
class EscalateOnOutputRetry(AbstractCapability[AgentDepsT]):
    """Escalate to a stronger model when output validation keeps failing.

    The run starts on the first level's model. Each failed output validation (a `ModelRetry` from an
    output validator, or a schema validation error) counts as one output retry. When a level has used
    all of its attempts, the next request goes to the next level's model, which sees the earlier
    attempts and their errors in the message history. The last level keeps receiving requests until
    the agent's output retry budget runs out.

    Retries of function tools are not counted: a tool asking the model to call it again says nothing
    about whether the model can produce valid output.

    The agent's output retry budget must cover every level. Set it to
    [`required_output_retries`][pydantic_ai_escalation.EscalateOnOutputRetry.required_output_retries]:

    ```python {test="skip"}
    from pydantic_ai import Agent
    from pydantic_ai_escalation import EscalateOnOutputRetry, Level

    escalation = EscalateOnOutputRetry(
        levels=[
            Level('openai:gpt-4o-mini', output_retries=1),
            Level('openai:gpt-5.6-terra', output_retries=1),
            Level('openai:gpt-5.6-sol'),
        ]
    )
    agent = Agent(capabilities=[escalation], retries={'output': escalation.required_output_retries})
    ```
    """

    levels: Sequence[Level]
    """The models to try, cheapest first."""

    _output_tool_names: frozenset[str] = field(default=frozenset[str](), init=False, repr=False, compare=False)
    _selected_level: int = field(default=0, init=False, repr=False, compare=False)
    _reported_level: int = field(default=0, init=False, repr=False, compare=False)
    _budget_checked: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.levels:
            raise UserError('`EscalateOnOutputRetry` needs at least one level')
        self.levels = tuple(self.levels)

    @classmethod
    def from_spec(cls, levels: Sequence[LevelSpec]) -> EscalateOnOutputRetry[Any]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Build the capability from its agent spec form: a list of `{model, output_retries}` mappings."""
        for spec in levels:
            # A misspelled key (`retries: 1`) would otherwise be dropped silently and leave the default.
            if unknown := set(spec) - _LEVEL_SPEC_KEYS:
                raise UserError(
                    f'Unknown `EscalateOnOutputRetry` level field(s) {sorted(unknown)}; '
                    f'expected {sorted(_LEVEL_SPEC_KEYS)}'
                )
        return cls(levels=[Level(spec['model'], spec.get('output_retries', 0)) for spec in levels])

    @property
    def required_output_retries(self) -> int:
        """The output retry budget that lets the run reach every attempt on every level."""
        return sum(level.attempts for level in self.levels) - 1

    def level_for(self, output_retries: int) -> int:
        """Return the index of the level that serves the request made after `output_retries` failed validations."""
        remaining = output_retries
        for index, level in enumerate(self.levels):
            if remaining < level.attempts:
                return index
            remaining -= level.attempts
        return len(self.levels) - 1

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        # A fresh copy per run: the output tool names and the reported level are per-run state.
        return replace(self)

    def get_model(self) -> ModelSelector[AgentDepsT]:
        return self._select_model

    def _select_model(self, ctx: ModelSelectionContext[AgentDepsT]) -> Model | str:
        failures = _count_output_retries(ctx.messages, self._output_tool_names)
        self._selected_level = self.level_for(failures)
        return self.levels[self._selected_level].model

    async def prepare_output_tools(
        self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        # Output tool retries look like function tool retries in the history; the names tell them apart.
        self._output_tool_names = frozenset(tool_def.name for tool_def in tool_defs)
        if not self._budget_checked:
            self._budget_checked = True
            # In this hook `ctx.max_retries` is the output retry budget.
            if ctx.max_retries < self.required_output_retries:
                warnings.warn(
                    f'The output retry budget ({ctx.max_retries}) is smaller than the '
                    f'{self.required_output_retries} that `EscalateOnOutputRetry` needs to reach every level; '
                    f"set `retries={{'output': {self.required_output_retries}}}` on the agent or run.",
                    EscalationBudgetWarning,
                    stacklevel=2,
                )
        return tool_defs

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if self._selected_level != self._reported_level:
            level = self.levels[self._selected_level]
            span = ctx.tracer.start_span(
                'escalate_on_output_retry',
                attributes={
                    'escalation.from_level': self._reported_level,
                    'escalation.to_level': self._selected_level,
                    'escalation.model': _model_name(level.model),
                },
            )
            span.end()
            self._reported_level = self._selected_level
        return request_context


def _model_name(model: Model | str) -> str:
    return model if isinstance(model, str) else model.model_name


def _count_output_retries(messages: Sequence[ModelMessage], output_tool_names: frozenset[str]) -> int:
    """Count the failed output validations in the current run.

    Handles both the retry parts current Pydantic AI records (`RetryPromptPart`) and the ones
    pydantic/pydantic-ai#8094 introduces (`RetryFeedbackPart`, and `ToolReturnPart` with
    `outcome='retried'` for tool retries), matching on `part_kind` so it works before and after.
    """
    run_id = _current_run_id(messages)
    if run_id is None:
        # The first request of a run carries no run id yet, and nothing in this run has failed.
        return 0
    count = 0
    for message in messages:
        if not isinstance(message, ModelRequest) or message.run_id != run_id:
            continue
        for part in message.parts:
            # `str()` keeps the comparison open to part kinds newer Pydantic AI versions add.
            kind = str(part.part_kind)
            tool_name: str | None = getattr(part, 'tool_name', None)
            if kind == 'retry-prompt':
                if tool_name is None or tool_name in output_tool_names:
                    count += 1
            elif kind == 'retry-feedback':
                count += 1
            elif kind == 'tool-return':
                if getattr(part, 'outcome', None) == 'retried' and tool_name in output_tool_names:
                    count += 1
    return count


def _current_run_id(messages: Sequence[ModelMessage]) -> str | None:
    last = messages[-1] if messages else None
    return last.run_id if isinstance(last, ModelRequest) else None

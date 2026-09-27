from __future__ import annotations

import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings

from pydantic_ai_escalation import EscalateOnOutputRetry, EscalationBudgetWarning, Level
from pydantic_ai_escalation._capability import _count_output_retries  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@dataclass
class Call:
    model: str
    saw_feedback: bool


@dataclass
class Scripted:
    """Fake models that answer from a script and record which model served each request."""

    calls: list[Call] = field(default_factory=list[Call])

    def model(self, name: str, answers: Callable[[int, AgentInfo], ModelResponse]) -> FunctionModel:
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            saw_feedback = any(
                isinstance(part, RetryPromptPart)
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
            )
            own_calls = sum(call.model == name for call in self.calls)
            self.calls.append(Call(name, saw_feedback))
            return answers(own_calls, info)

        return FunctionModel(respond, model_name=name)

    @property
    def sequence(self) -> list[str]:
        return [call.model for call in self.calls]


def text(value: str) -> Callable[[int, AgentInfo], ModelResponse]:
    return lambda _n, _info: ModelResponse(parts=[TextPart(value)])


def text_after(bad: int, good: str) -> Callable[[int, AgentInfo], ModelResponse]:
    return lambda n, _info: ModelResponse(parts=[TextPart('wrong' if n < bad else good)])


def require_ok(_ctx: RunContext[object], output: str) -> str:
    if output != 'ok':
        raise ModelRetry(f'expected ok, got {output}')
    return output


class Invoice(BaseModel):
    total: int


def invoice(total: int) -> Callable[[int, AgentInfo], ModelResponse]:
    return lambda _n, info: ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'total': total})])


def require_total(_ctx: RunContext[object], output: Invoice) -> Invoice:
    if output.total != 100:
        raise ModelRetry(f'total should be 100, got {output.total}')
    return output


def escalate(*levels: Level) -> EscalateOnOutputRetry[object]:
    return EscalateOnOutputRetry[object](levels=levels)


class TestEscalateOnOutputRetry:
    async def test_first_level_success_does_not_escalate(self):
        script = Scripted()
        escalation = escalate(Level(script.model('mini', text('ok'))), Level(script.model('sol', text('ok'))))
        agent = Agent(capabilities=[escalation], retries={'output': escalation.required_output_retries})
        agent.output_validator(require_ok)

        result = await agent.run('go')

        assert result.output == 'ok'
        assert script.sequence == ['mini']

    async def test_feedback_retries_then_escalates(self):
        script = Scripted()
        escalation = escalate(
            Level(script.model('mini', text('wrong')), output_retries=1),
            Level(script.model('terra', text_after(1, 'ok')), output_retries=1),
            Level(script.model('sol', text('ok'))),
        )
        agent = Agent(capabilities=[escalation], retries={'output': escalation.required_output_retries})
        agent.output_validator(require_ok)

        result = await agent.run('go')

        assert result.output == 'ok'
        assert script.sequence == ['mini', 'mini', 'terra', 'terra']
        assert [call.saw_feedback for call in script.calls] == [False, True, True, True]

    async def test_reaches_last_level(self):
        script = Scripted()
        escalation = escalate(
            Level(script.model('mini', text('wrong')), output_retries=1),
            Level(script.model('terra', text('wrong'))),
            Level(script.model('sol', text('ok'))),
        )
        agent = Agent(capabilities=[escalation], retries={'output': escalation.required_output_retries})
        agent.output_validator(require_ok)

        result = await agent.run('go')

        assert result.output == 'ok'
        assert script.sequence == ['mini', 'mini', 'terra', 'sol']

    async def test_last_level_serves_until_budget_runs_out(self):
        script = Scripted()
        escalation = escalate(Level(script.model('mini', text('wrong'))), Level(script.model('sol', text('wrong'))))
        agent = Agent(capabilities=[escalation], retries={'output': 3})
        agent.output_validator(require_ok)

        with pytest.raises(UnexpectedModelBehavior, match='Exceeded maximum output retries'):
            await agent.run('go')

        assert script.sequence == ['mini', 'sol', 'sol', 'sol']

    async def test_tool_output_and_function_tool_retries(self):
        script = Scripted()

        def mini_answers(n: int, info: AgentInfo) -> ModelResponse:
            if n == 0:
                return ModelResponse(parts=[ToolCallPart('lookup', {'query': 'x'})])
            return invoice(1)(n, info)

        escalation = escalate(
            Level(script.model('mini', mini_answers), output_retries=1), Level(script.model('sol', invoice(100)))
        )
        agent = Agent(
            capabilities=[escalation],
            output_type=Invoice,
            retries={'output': escalation.required_output_retries, 'tools': 3},
        )
        agent.output_validator(require_total)

        @agent.tool_plain
        def lookup(query: str) -> str:  # pyright: ignore[reportUnusedFunction]
            raise ModelRetry(f'no results for {query}')

        result = await agent.run('go')

        assert result.output == Invoice(total=100)
        # The function tool retry after the first call does not count toward escalation;
        # the two failed invoices do.
        assert script.sequence == ['mini', 'mini', 'mini', 'sol']

    async def test_earlier_runs_in_history_do_not_count(self):
        script = Scripted()
        escalation = escalate(
            Level(script.model('mini', text_after(1, 'ok')), output_retries=1), Level(script.model('sol', text('ok')))
        )
        agent = Agent(capabilities=[escalation], retries={'output': escalation.required_output_retries})
        agent.output_validator(require_ok)

        first = await agent.run('go')
        second = await agent.run('again', message_history=first.all_messages())

        assert second.output == 'ok'
        assert script.sequence == ['mini', 'mini', 'mini']

    async def test_runs_do_not_share_state(self):
        script = Scripted()
        escalation = escalate(Level(script.model('mini', text_after(1, 'ok'))), Level(script.model('sol', text('ok'))))
        agent = Agent(capabilities=[escalation], retries={'output': escalation.required_output_retries})
        agent.output_validator(require_ok)

        await agent.run('first')
        await agent.run('second')

        assert script.sequence == ['mini', 'sol', 'mini']

    async def test_budget_warning_on_tool_output(self):
        script = Scripted()
        escalation = escalate(
            Level(script.model('mini', invoice(1)), output_retries=2), Level(script.model('sol', invoice(100)))
        )
        agent = Agent(capabilities=[escalation], output_type=Invoice, retries={'output': 1})
        agent.output_validator(require_total)

        with (
            pytest.warns(EscalationBudgetWarning, match=r"retries=\{'output': 3\}"),
            pytest.raises(UnexpectedModelBehavior),
        ):
            await agent.run('go')

    async def test_escalation_span(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        script = Scripted()
        escalation = escalate(
            Level(script.model('mini', text('wrong')), output_retries=1),
            Level('test'),
            Level(script.model('sol', text('ok'))),
        )
        agent = Agent(
            capabilities=[escalation, Instrumentation(InstrumentationSettings(tracer_provider=provider))],
            retries={'output': escalation.required_output_retries},
        )

        @agent.output_validator
        def validate(ctx: RunContext[object], output: str) -> str:  # pyright: ignore[reportUnusedFunction]
            if output != 'ok':
                raise ModelRetry('try again')
            return output

        await agent.run('go')

        escalations = [span for span in exporter.get_finished_spans() if span.name == 'escalate_on_output_retry']
        assert [dict(span.attributes or {}) for span in escalations] == [
            {'escalation.from_level': 0, 'escalation.to_level': 1, 'escalation.model': 'test'},
            {'escalation.from_level': 1, 'escalation.to_level': 2, 'escalation.model': 'sol'},
        ]

    async def test_agent_spec(self, tmp_path: Path):
        spec = tmp_path / 'agent.yaml'
        spec.write_text(
            textwrap.dedent(
                """\
                retries:
                  output: 2
                capabilities:
                  - EscalateOnOutputRetry:
                      levels:
                        - {model: test, output_retries: 1}
                        - {model: test}
                """
            )
        )

        agent = Agent.from_file(spec, custom_capability_types=[EscalateOnOutputRetry])
        result = await agent.run('go')

        assert result.output == 'success (no tool calls)'

    async def test_agent_spec_rejects_unknown_level_fields(self, tmp_path: Path):
        spec = tmp_path / 'agent.yaml'
        spec.write_text(
            textwrap.dedent(
                """\
                capabilities:
                  - EscalateOnOutputRetry:
                      levels:
                        - {model: test, retries: 1}
                """
            )
        )

        with pytest.raises(ValueError, match=r"Unknown `EscalateOnOutputRetry` level field\(s\) \['retries'\]"):
            Agent.from_file(spec, custom_capability_types=[EscalateOnOutputRetry])

    async def test_missing_output_escalates(self):
        script = Scripted()
        escalation = escalate(
            Level(script.model('mini', text('I forgot to call the output tool'))),
            Level(script.model('sol', invoice(100))),
        )
        agent = Agent(
            capabilities=[escalation], output_type=Invoice, retries={'output': escalation.required_output_retries}
        )

        result = await agent.run('go')

        assert result.output == Invoice(total=100)
        assert script.sequence == ['mini', 'sol']


class TestLevels:
    def test_required_output_retries(self):
        escalation = escalate(Level('test', output_retries=2), Level('test'), Level('test'))

        assert escalation.required_output_retries == 4
        assert [escalation.level_for(n) for n in range(7)] == [0, 0, 0, 1, 2, 2, 2]

    def test_from_spec_defaults(self):
        escalation = EscalateOnOutputRetry.from_spec([{'model': 'a', 'output_retries': 2}, {'model': 'b'}])

        assert escalation.levels == (Level('a', output_retries=2), Level('b'))

    def test_needs_a_level(self):
        with pytest.raises(UserError, match='at least one level'):
            escalate()

    def test_negative_retries(self):
        with pytest.raises(UserError, match='0 or greater'):
            Level('test', output_retries=-1)


@dataclass
class _FutureRetryFeedbackPart:
    """Stand-in for the `RetryFeedbackPart` that pydantic/pydantic-ai#8094 adds."""

    part_kind: str = 'retry-feedback'


@dataclass
class _FutureToolReturnPart:
    """Stand-in for a `ToolReturnPart` with the `outcome='retried'` that pydantic/pydantic-ai#8094 adds."""

    tool_name: str
    outcome: str
    part_kind: str = 'tool-return'


class TestRetryPartsAfterDeprecation:
    """The history format pydantic/pydantic-ai#8094 introduces, which the current release can't produce."""

    def test_counts_retry_feedback_and_retried_output_tool_returns(self):
        request = ModelRequest(
            parts=[
                UserPromptPart('go'),
                _FutureRetryFeedbackPart(),  # pyright: ignore[reportArgumentType]
                _FutureToolReturnPart('final_result', 'retried'),  # pyright: ignore[reportArgumentType]
                _FutureToolReturnPart('lookup', 'retried'),  # pyright: ignore[reportArgumentType]
                ToolReturnPart('final_result', 'ok', tool_call_id='3'),
            ],
            run_id='run',
        )

        assert _count_output_retries([request], frozenset({'final_result'})) == 2

    def test_empty_history(self):
        assert _count_output_retries([], frozenset()) == 0

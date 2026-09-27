"""Escalation in runs that stream model responses.

`run_stream_events()` and `run()` with an `event_stream_handler` stream every model response and
still retry output validation, so escalation works as in a plain `run()`. `run_stream()` stops at
the first output and doesn't support output retries at all, so there is nothing to escalate.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import AgentStreamEvent, ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from pydantic_ai_escalation import EscalateOnOutputRetry, Level

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class Invoice(BaseModel):
    total: int


def streamed_text(served: list[str], name: str, value: str) -> FunctionModel:
    async def stream(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        served.append(name)
        yield value

    return FunctionModel(stream_function=stream, model_name=name)


def streamed_invoice(served: list[str], name: str, total: int) -> FunctionModel:
    async def stream(_messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        served.append(name)
        yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args=f'{{"total": {total}}}')}

    return FunctionModel(stream_function=stream, model_name=name)


def require_ok(ctx: RunContext[object], output: str) -> str:
    if not ctx.partial_output and output != 'ok':
        raise ModelRetry(f'expected ok, got {output}')
    return output


def require_total(ctx: RunContext[object], output: Invoice) -> Invoice:
    if not ctx.partial_output and output.total != 100:
        raise ModelRetry(f'total should be 100, got {output.total}')
    return output


def text_agent(served: list[str]) -> Agent[object, str]:
    escalation = EscalateOnOutputRetry[object](
        levels=[
            Level(streamed_text(served, 'mini', 'wrong'), output_retries=1),
            Level(streamed_text(served, 'sol', 'ok')),
        ]
    )
    agent = Agent(capabilities=[escalation], retries={'output': escalation.required_output_retries})
    agent.output_validator(require_ok)
    return agent


class TestStreaming:
    async def test_run_stream_events_text(self):
        served: list[str] = []
        agent = text_agent(served)

        outputs: list[str] = []
        async with agent.run_stream_events('go') as events:
            async for event in events:
                if event.event_kind == 'agent_run_result':
                    outputs.append(event.result.output)

        assert outputs == ['ok']
        assert served == ['mini', 'mini', 'sol']

    async def test_run_stream_events_tool_output(self):
        served: list[str] = []
        escalation = EscalateOnOutputRetry[object](
            levels=[
                Level(streamed_invoice(served, 'mini', 1), output_retries=1),
                Level(streamed_invoice(served, 'sol', 100)),
            ]
        )
        agent = Agent(
            capabilities=[escalation], output_type=Invoice, retries={'output': escalation.required_output_retries}
        )
        agent.output_validator(require_total)

        outputs: list[Invoice] = []
        async with agent.run_stream_events('go') as events:
            async for event in events:
                if event.event_kind == 'agent_run_result':
                    outputs.append(event.result.output)

        assert outputs == [Invoice(total=100)]
        assert served == ['mini', 'mini', 'sol']

    async def test_run_with_event_stream_handler(self):
        served: list[str] = []
        agent = text_agent(served)

        async def handle(_ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _ in stream:
                pass

        result = await agent.run('go', event_stream_handler=handle)

        assert result.output == 'ok'
        assert served == ['mini', 'mini', 'sol']

    async def test_run_stream_does_not_retry_output(self):
        served: list[str] = []
        agent = text_agent(served)

        with pytest.raises(UnexpectedModelBehavior, match='retries are not supported in `run_stream\\(\\)`'):
            async with agent.run_stream('go') as result:
                await result.get_output()

        assert served == ['mini']

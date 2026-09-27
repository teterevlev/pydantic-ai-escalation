# pydantic-ai-escalation

Escalate a [Pydantic AI](https://ai.pydantic.dev/) agent to a stronger model when output validation keeps failing.

Most requests in an extraction or classification pipeline are handled well by a cheap model. Output validators catch the rest: the totals don't add up, the entity isn't in the database, the date is in the future. `EscalateOnOutputRetry` gives the cheap model a chance to fix its answer from the validation error, and only when that isn't enough moves the run to a stronger model, which sees what went wrong before it.

```
mini ──✗──▶ mini + error ──✗──▶ terra + errors ──✓
```

## Install

```bash
pip install pydantic-ai-escalation
```

Requires `pydantic-ai-slim` 2.51 or later and Python 3.10+.

## Usage

```python
from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai_escalation import EscalateOnOutputRetry, Level


class Invoice(BaseModel):
    lines: list[float]
    total: float


escalation = EscalateOnOutputRetry(
    levels=[
        Level('openai:gpt-4o-mini', output_retries=1),
        Level('openai:gpt-5.6-terra', output_retries=1),
        Level('openai:gpt-5.6-sol'),
    ]
)

agent = Agent(
    output_type=Invoice,
    capabilities=[escalation],
    retries={'output': escalation.required_output_retries},
)


@agent.output_validator
def totals_match(ctx: RunContext[None], invoice: Invoice) -> Invoice:
    if abs(sum(invoice.lines) - invoice.total) > 0.01:
        raise ModelRetry(f'The lines add up to {sum(invoice.lines)}, but the total is {invoice.total}.')
    return invoice
```

With these levels a run makes at most five requests:

| Request | Model | Sees |
| --- | --- | --- |
| 1 | `gpt-4o-mini` | the prompt |
| 2 | `gpt-4o-mini` | the prompt, its answer, the validation error |
| 3 | `gpt-5.6-terra` | everything above |
| 4 | `gpt-5.6-terra` | everything above, plus its own attempt and error |
| 5 | `gpt-5.6-sol` | everything above |

The run stops at the first answer that passes validation.

### In an agent spec

The capability can be declared in a [YAML or JSON agent spec](https://ai.pydantic.dev/agent-spec/):

```yaml
# agent.yaml
retries:
  output: 4
capabilities:
  - EscalateOnOutputRetry:
      levels:
        - {model: openai:gpt-4o-mini, output_retries: 1}
        - {model: openai:gpt-5.6-terra, output_retries: 1}
        - {model: openai:gpt-5.6-sol}
```

```python
from pydantic_ai import Agent
from pydantic_ai_escalation import EscalateOnOutputRetry

agent = Agent.from_file('agent.yaml', custom_capability_types=[EscalateOnOutputRetry])
```

## How it works

- **A level** is a model plus `output_retries`, the number of feedback retries it gets before the run moves on. `output_retries=0` (the default) escalates after the first failed attempt.
- **What counts as a failure:** every failed output validation in the current run: a `ModelRetry` from an output validator, an output that doesn't match the schema, or a response with no output at all. Retries of function tools don't count, and neither do failures from earlier runs in the message history.
- **The next level sees everything.** Escalation doesn't reset the conversation: the stronger model receives the earlier attempts and their errors, so it knows what didn't work.
- **The last level keeps going** until the agent's output retry budget runs out, then the run fails the way it would without the capability.

### The output retry budget

Pydantic AI stops a run once it has used up its output retry budget (`retries={'output': ...}`, which defaults to 1). A capability can't raise that budget, so set it yourself: `required_output_retries` is the budget that reaches every attempt on every level. If the budget is too small, the upper levels are never reached. On the tool output path (a structured `output_type`, the default) the capability detects this and emits an `EscalationBudgetWarning`; with plain text output it can't see the budget, so check it yourself.

## Telemetry

When [instrumentation](https://ai.pydantic.dev/logfire/) is on, each escalation emits one zero-duration `escalate_on_output_retry` span with `escalation.from_level`, `escalation.to_level` and `escalation.model`. Requests that stay on the same level emit nothing extra: core's own spans already cover them. The span carries no prompt or output content.

## Limitations

- Tested with the regular `run` loop. Streaming runs and durable execution use the same model selection path but haven't been tested yet.
- The upcoming change to how Pydantic AI records retries ([pydantic/pydantic-ai#8094](https://github.com/pydantic/pydantic-ai/pull/8094)) is handled, but only tested against a simulation of the new format.

## Why it's built this way

See [DESIGN.md](DESIGN.md) for the decisions behind the API: why this is a capability rather than a `FallbackModel`, what exactly counts as a failure, and why the budget is yours to set.

## License

MIT

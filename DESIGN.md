# Design notes

Why `EscalateOnOutputRetry` works the way it does. Each section states a decision, the alternatives that were considered, and why they lost. Everything here was checked against `pydantic-ai-slim` 2.51.0.

## The problem

A pipeline that runs a cheap model with strict output validators fails in a predictable way: most inputs pass, a few fail validation, and some of those would pass if the model saw the error, while others need a stronger model. The policy we want is:

1. Try the cheap model.
2. If validation fails, let the same model retry with the error message.
3. If it still fails, move to a stronger model, which should also see what went wrong.

Pydantic AI has two mechanisms that each cover part of this, and they don't combine.

**`ModelRetry` from an output validator** sends the error back and asks again, always on the same model. It covers step 2 only.

**`FallbackModel`** moves to the next model on API errors, or, with a response handler in `fallback_on`, on a response the handler rejects. It covers step 3, with two gaps:

- A rejected response is discarded. The next model receives the original request and never learns why the previous answer was rejected, so step 3's "should also see what went wrong" is lost.
- Every retry triggered by `ModelRetry` is a new request, and `FallbackModel` starts each request from its first model. Combined with output validators, the run keeps retrying the cheapest model until the output retry budget is spent and never reaches the stronger ones.

Both behaviours were confirmed with scripted `FunctionModel` runs before this package was written. Neither is a bug: `FallbackModel` is built for availability (the provider is down, try another), not for quality.

## A capability, not a model wrapper

**Decision:** implement the policy as a capability whose `get_model()` returns a model selector.

**Alternatives:**

- *A `FallbackModel` subclass or wrapper model.* A model sees one request at a time and has no notion of the run's validation history, so it can't tell "the first attempt" from "the third attempt after two validation errors". It would also sit below output validation, which runs after the model returns.
- *Swapping `request_context.model` in `before_model_request`.* This hook has the information we need, but by the time it runs, the request parameters (tool schemas, output mode) have already been prepared for the originally selected model's profile. Swapping to a model from a different provider at that point risks sending parameters that model doesn't support.
- *A model selector.* Pydantic AI calls it before each request step, before anything is prepared for a specific model, and hands it the message history. This is the documented extension point for choosing a model per step.

The selector runs before the step's hooks, so it can't read the run's internal retry counter; it works out the count from the history instead (next section). The rest of the capability is small: `prepare_output_tools` records output tool names and checks the budget, and `before_model_request` emits the telemetry span.

## What counts as a failure

**Decision:** count failed output validations in the current run, and nothing else.

A failed output validation is recorded in the history as a retry part. The selector counts retry parts that:

- belong to the current run, identified by `run_id`, so a conversation continued with `message_history` starts every run on the first level;
- are about the output, not about a function tool.

**Why function tool retries don't count.** A tool raising `ModelRetry` (a lookup found nothing, an argument was malformed) says nothing about whether the model can produce valid output. Counting them would escalate agents that use tools heavily for reasons unrelated to output quality, and would burn through the levels before the model had even attempted an answer.

**How output retries are told apart.** With plain text, native or prompted output, an output retry part has no tool name. With tool output (the default for a structured `output_type`) it carries the output tool's name, which looks exactly like a function tool retry. The capability records the output tool names in `prepare_output_tools` and uses them to classify retry parts. Those names are recorded at each step and used by the selector at the next one, which is fine because no output retry can exist before the first response.

**Why count from the history instead of keeping a counter.** Pydantic AI's guidance for capabilities is that per-run state must be derivable from the run context, because durable execution re-creates capabilities in worker processes. The history is the source of truth that survives that; a private counter wouldn't. It also means the count can't drift from what actually happened in the run. The capability does keep two small pieces of per-run state in a fresh copy made by `for_run`: the output tool names and the last level it reported to telemetry. Neither is derived from the history, so if a durable worker re-creates the capability mid-run, the first request after that would not know the output tool names and could go to a lower level than it should, and an escalation span could be emitted twice. Both correct themselves on the next step. This is untested, which is why durable execution is listed under "Not done yet".

**Forward compatibility.** [pydantic/pydantic-ai#8094](https://github.com/pydantic/pydantic-ai/pull/8094), approved but not yet released at the time of writing, replaces `RetryPromptPart` with `RetryFeedbackPart` for non-tool retries and `ToolReturnPart(outcome='retried')` for tool retries. The counter matches on `part_kind` rather than on classes, so it handles both formats without importing a class that doesn't exist yet. Without this, the release of #8094 would silently turn escalation off: no `RetryPromptPart` would ever be found, the count would stay at zero, and every request would go to the first level.

## Levels

**Decision:** an ordered list of `Level(model, output_retries)`.

- *Per-level retries* rather than one global setting, because the right number differs: a feedback retry on a cheap model costs little, while extra attempts on the strongest model are what the whole design tries to avoid.
- *`output_retries=0` by default.* It is the simplest rule to explain ("one attempt per level unless you ask for more") and never spends more than the user configured. The examples use `1` on the cheaper levels because that's where a feedback retry pays off.
- *The next level sees the whole history.* The stronger model gets every failed attempt and its error. This is usually what you want: knowing what didn't work is useful context. The cost is tokens, now billed at the stronger model's price, and the history grows with every retry ([pydantic/pydantic-ai#7875](https://github.com/pydantic/pydantic-ai/issues/7875), [#4908](https://github.com/pydantic/pydantic-ai/issues/4908)). Trimming on escalation is a reasonable future option, but it should be opt-in, so version 0.1 doesn't do it.
- *The last level keeps serving* once all levels are used, until the budget runs out. The alternative, failing as soon as the last level's attempts are used, would make the capability's own limit fight with Pydantic AI's budget; having one limit is easier to reason about.
- *Plain dataclasses, not dicts.* `Level` is a frozen, typed dataclass, so mistakes such as a negative `output_retries` or a misspelled field fail at construction rather than halfway through a run.

## The output retry budget

**Decision:** the user sets `retries={'output': ...}`; the capability provides `required_output_retries` and warns when it can see that the budget is too small.

A capability has no public way to raise the agent's output retry budget, and the agent exposes no public accessor for it. Pydantic AI stops the run when the budget is spent, so a budget that is too small makes the upper levels unreachable, which defeats the purpose.

- `required_output_retries` computes the budget that reaches every attempt on every level, so the user never has to add it up by hand.
- On the tool output path, `prepare_output_tools` receives the output budget as `ctx.max_retries`, so the capability checks it once per run and emits an `EscalationBudgetWarning`. It is a warning rather than an error because a smaller budget can be a deliberate cost cap.
- On the plain text path no hook exposes the budget, so there is nothing to check. The README says so explicitly rather than implying a safety net that doesn't exist.

## Telemetry

**Decision:** one zero-duration span per escalation, nothing per request.

The capability's own decision is the escalation; the requests themselves are already covered by core's spans. The span name is static (`escalate_on_output_retry`) with the variable parts as attributes (`escalation.from_level`, `escalation.to_level`, `escalation.model`), following the Pydantic AI Harness conventions. Model names are configuration, not user content, so nothing needs to be gated behind `trace_include_content`. The span also gives an answer to the most useful operational question: how often the cheap model is enough.

## Agent spec support

`from_spec` accepts a list of `{model, output_retries}` mappings typed as a `TypedDict`, so the JSON schema Pydantic AI generates for agent specs describes the fields, and model names stay plain strings in YAML. The spec loader does not reject unknown keys inside nested mappings, so `from_spec` checks them itself: a misspelled `retries: 1` would otherwise be dropped silently and leave the level at `output_retries=0`, which is the kind of configuration mistake that only shows up as a surprisingly high bill.

## Naming

- **`EscalateOnOutputRetry`** follows the Pydantic AI Harness naming rule: a capability that acts on the run and whose whole contract fits in one verb phrase gets an imperative name (like `ClearToolResults` or `WarnOnCacheBusts`). The name answers "what will this do to my agent?" and says what triggers it.
- **`Level`** and **`output_retries`** use the user's vocabulary: Pydantic AI already calls the budget `retries={'output': ...}`.
- **`pydantic-ai-escalation`** follows the `pydantic-ai-<name>` convention for third-party capability packages.

## Package layout and quality bar

The layout mirrors a Pydantic AI Harness capability (`_capability.py` with the implementation, public names re-exported from `__init__.py`), so that moving it into the harness later would be close to a copy. The code is checked with Pyright in strict mode and Ruff (120 columns, single quotes), and tests keep 100% branch coverage. Tests go through `Agent` with scripted `FunctionModel`s and make no provider calls. The one exception to testing through the public surface is the forward-compatibility test for #8094's retry format, which the released Pydantic AI can't produce yet.

## Streaming

`run_stream_events()` and `run()` with an `event_stream_handler` stream every model response and still retry output validation, so escalation behaves exactly as in a plain `run()`; the tests cover both, with text and tool output. `run_stream()` is different: Pydantic AI treats the first output as final and raises `UnexpectedModelBehavior` if it fails validation, because the output has already been streamed to the caller. There is no second request, so there is nothing for the capability to escalate. It can't detect which method started the run, so this is documented rather than warned about.

## Not done yet

- Durable execution uses the same selection path but isn't covered by tests.
- Optional trimming of earlier failed attempts on escalation.
- Letting a level use the agent's own model instead of naming one explicitly.

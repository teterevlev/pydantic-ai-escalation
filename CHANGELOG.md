# Changelog

## Unreleased

- Test escalation in streaming runs (`run_stream_events()`, `run()` with an `event_stream_handler`) and document that `run_stream()` can't escalate because it doesn't retry output validation.

## 0.1.0

- First release: `EscalateOnOutputRetry` capability with per-level feedback retries, agent spec support, an `EscalationBudgetWarning` for a too-small output retry budget on the tool output path, and an `escalate_on_output_retry` telemetry span.

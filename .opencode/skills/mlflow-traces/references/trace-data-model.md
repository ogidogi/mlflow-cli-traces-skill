# MLflow trace data model (cheat sheet)

Loaded on demand. This is the shape the CLI reads. It targets the **MLflow 3.x**
tracing API (3.13+); the field names below were verified against MLflow 3.14.

## Trace = TraceInfo + TraceData

A `Trace` object has two parts:

- `trace.info` — a `TraceInfo` (metadata, timing, status)
- `trace.data` — a `TraceData` holding `trace.data.spans` (a list of `Span`)

### TraceInfo fields

| Meaning | Field | Type |
|---|---|---|
| Trace ID | `trace_id` | `str` (e.g. `tr-<hex>`) |
| Status | `state` | `TraceState` enum → `OK` / `ERROR` / `IN_PROGRESS` / `STATE_UNSPECIFIED` |
| Total latency | `execution_duration` | `int` milliseconds |
| Start time | `request_time` | `int` epoch milliseconds |
| Input preview (JSON, truncated) | `request_preview` | `str` |
| Output preview (JSON, truncated) | `response_preview` | `str` or `None` (often `None` on error) |
| Backend metadata (dict) | `trace_metadata` | `dict` |
| User/searchable tags (dict) | `tags` | `dict` |

Useful metadata keys: `mlflow.trace.run_id` (links the trace to an MLflow run),
`mlflow.source.name`. Useful tag: `mlflow.traceName`.

## Span (OpenTelemetry-compliant)

Each `Span` in `trace.data.spans`:

| Meaning | Field(s) |
|---|---|
| Span ID | `span_id` |
| Parent span ID (root = `None`) | `parent_id` |
| Name | `name` |
| Start / end (nanoseconds) | `start_time_ns` / `end_time_ns` |
| Status | `status.status_code` (`SpanStatusCode` enum: `OK` / `ERROR` / `UNSET`) + `status.description` |
| Inputs / outputs | `inputs` / `outputs` (dict; `None` if unset) |
| Attributes (dict) | `attributes` — e.g. `mlflow.spanType` = `LLM`/`RETRIEVER`/`CHAIN`/… |
| Events | `events` — list of `SpanEvent` (`.name`, `.attributes`) |

**Per-span duration:** `(end_time_ns - start_time_ns) / 1_000_000` ms.
**Self time:** span total minus the sum of its direct children's totals.

## Exceptions (root-causing)

When an exception propagates out of a traced span, MLflow sets
`span.status = ERROR` and appends a `SpanEvent` named `exception` whose
`attributes` carry:

- `exception.type` — e.g. `RateLimitError`, `ValueError`
- `exception.message`
- `exception.stacktrace`

A parent span's ERROR is often *propagated* from a failing child — the child
with the `exception` event is the real culprit. The CLI's `errors` command
groups by the exception-bearing span for exactly this reason.

## Fetch & search (Python)

```python
from mlflow import MlflowClient
client = MlflowClient()                       # reads MLFLOW_TRACKING_URI / Databricks auth

trace = client.get_trace("<trace_id>")        # one trace

errors = client.search_traces(                # many, filtered
    experiment_ids=["<id>"],
    filter_string="trace.status = 'ERROR'",
    order_by=["timestamp_ms DESC"],
    max_results=200,
)
```

`filter_string` is a SQL-like DSL over status, tags, and metadata, e.g.
`trace.status = 'ERROR' AND tag.environment = 'prod'`.

> Note: in MLflow 3.14 the `experiment_ids` argument is deprecated in favour of
> `locations`, but it still works (the CLI silences that warning). It remains
> the portable choice for 3.13+.

## Sources

- https://mlflow.org/docs/latest/genai/tracing/search-traces/
- https://mlflow.org/docs/latest/tracing/tracing-schema
- https://mlflow.org/docs/latest/python_api/mlflow.client.html

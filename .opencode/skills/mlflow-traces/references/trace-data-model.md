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

Metadata values are JSON **strings**, including the aggregates that
`TraceInfo.token_usage` / `TraceInfo.cost` decode for you:

| Meaning | Metadata key | Decoded shape |
|---|---|---|
| Token usage for the trace | `mlflow.trace.tokenUsage` | `{input_tokens, output_tokens, total_tokens}` |
| Cost for the trace (USD) | `mlflow.trace.cost` | `{input_cost, output_cost, total_cost}` |

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

LLM span attributes (present only when the provider integration reports them;
`Span.span_type` / `.model_name` / `.llm_cost` are the typed accessors):

| Meaning | Attribute key |
|---|---|
| Span type | `mlflow.spanType` |
| Model / provider | `mlflow.llm.model` / `mlflow.llm.provider` |
| Token usage | `mlflow.chat.tokenUsage` — `{input_tokens, output_tokens, total_tokens, cache_read_input_tokens, cache_creation_input_tokens}` |
| Cost (USD) | `mlflow.llm.cost` — `{input_cost, output_cost, total_cost}` |

## JSON export shape (`Trace.to_json()`)

A serialized trace is **not** the live entity's field names — it is the OTel
proto spelling. This is what a dump saved from the UI or from
`trace.to_json()` looks like, and what `--from-file` has to read:

| Live entity | JSON export |
|---|---|
| `info.execution_duration` (int ms) | `info.execution_duration_ms` |
| `info.request_time` (int epoch ms) | `info.request_time` (**ISO-8601 string**) |
| `span.parent_id` | `span.parent_span_id` |
| `span.start_time_ns` / `end_time_ns` | `span.start_time_unix_nano` / `end_time_unix_nano` |
| `span.status.status_code` = `ERROR` | `span.status.code` = `"STATUS_CODE_ERROR"`, `.message` |
| `event.timestamp` | `event.time_unix_nano` |
| `span.inputs` / `span.outputs` | *absent* — read `attributes["mlflow.spanInputs"/"mlflow.spanOutputs"]` |
| `span.attributes` values (decoded) | every value is a **JSON-encoded string** |

Span/trace ids inside the export are base64 OTel ids, not the `tr-<hex>` form;
they are still internally consistent, so parent links resolve.

## Exceptions (root-causing)

When an exception propagates out of a traced span, MLflow sets
`span.status = ERROR` and appends a `SpanEvent` named `exception` whose
`attributes` carry:

- `exception.type` — e.g. `RateLimitError`, `ValueError`
- `exception.message`
- `exception.stacktrace`

An exception that propagates is re-recorded on **every** ancestor span, not
just the one that raised — a 3-level trace yields the same `exception` event
three times, with `status.description` set to `"Type: message"` on each. The
CLI attributes it to the *deepest* span carrying it (the origin) and reports
the ancestors as a propagation trail, so one failure is counted once.

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
errors.token                                  # page token; None when exhausted
```

`search_traces` returns a `PagedList`: the server may cap the page below
`max_results`, so follow `.token` via `page_token=` to collect the rest.
`include_spans=True` (the default) is what makes span-level grouping possible.

`filter_string` is a SQL-like DSL over status, tags, and metadata, e.g.
`trace.status = 'ERROR' AND tag.environment = 'prod'`. A time window pushes
down as `trace.timestamp > <epoch_ms>`, which beats filtering client-side —
otherwise `max_results` is spent on traces outside the window.

> Note: in MLflow 3.14 the `experiment_ids` argument is deprecated in favour of
> `locations`, but it still works (the CLI silences that warning). It remains
> the portable choice for 3.13+.

## Sources

- https://mlflow.org/docs/latest/genai/tracing/search-traces/
- https://mlflow.org/docs/latest/tracing/tracing-schema
- https://mlflow.org/docs/latest/python_api/mlflow.client.html

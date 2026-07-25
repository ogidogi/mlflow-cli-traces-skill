---
name: mlflow-traces
description: >-
  Inspect, profile, and root-cause MLflow traces from the CLI: span waterfalls,
  latency hotspots, token/cost accounting, and exception root-causing. Use when
  given an MLflow trace ID, asked why an LLM/RAG/agent run was slow or failed,
  asked to triage errors across an MLflow experiment, or handed a saved trace
  JSON dump. Targets MLflow 3.x tracing, self-hosted or Databricks.
license: MIT
metadata:
  requires: uv (recommended) or python3 with mlflow-skinny
---

# MLflow Traces

Turn an MLflow trace into answers: the span waterfall, where the time went,
what threw, and what to fix.

| Situation | Command |
|---|---|
| Have a trace ID; want detail / inputs / outputs / span tree | `get` |
| A trace is slow — find the hotspot, or compare before/after a prompt or model change | `profile` |
| A trace failed — root-cause it and propose a fix | `diagnose` |
| Triage failures across a whole experiment | `errors` |
| Audit token usage and cost | `get` (per span + trace totals) |

## Running

```
uv run <skill-dir>/scripts/mlflow_traces.py <command> [args]
```

**Run from the project directory, not the skill directory** — invoke the script
by its full path. The script auto-discovers a `.env` by walking up from the
*current working directory*, so staying in the project is what lets it find
that project's credentials. (Examples below abbreviate the path as
`scripts/mlflow_traces.py`.)

Flags go **after** the subcommand. `--json` works on every command; the default
text output is already compact and meant to be read directly, so prefer it
unless you need to post-process.

**No install step.** Under `uv` the client is fetched on demand (PEP 723
declares `mlflow-skinny>=3.13`). Plain `python3` works if `mlflow` is installed.

## Commands

### get — full detail + span-tree waterfall

```
uv run scripts/mlflow_traces.py get <trace_id> [<trace_id> ...]
uv run scripts/mlflow_traces.py get <trace_id> --errors-only   # only failing spans, with their inputs
uv run scripts/mlflow_traces.py get <trace_id> --max-io 2000   # widen I/O truncation (default 500 chars)
```

Several ids are fetched in parallel.

```
Trace tr-abc123def456  [✗ ERROR]
  duration : 1.25s
  started  : 2024-07-24T12:00:00+00:00
  tokens   : in 120 / out 30 / total 150
  cost     : $0.003000
  request  : {"question": "What is the refund policy?"}
  spans (4):
✗ predict  1.25s (self 20.0ms)  ████████████████████  [CHAIN]
    ! downstream span 'llm_generate' failed
  ✓ retrieve_docs  50.0ms (self 50.0ms)    [RETRIEVER]
  ✓ rerank  840.0ms (self 840.0ms)  █████████████  [RERANKER]
  ✗ llm_generate  340.0ms (self 340.0ms)  █████  [LLM  gpt-4  in 120 / out 30 / total 150  $0.0030]
      ! RateLimitError: Rate limit exceeded for model gpt-4 (requested 31200)
```

### profile — latency breakdown

```
uv run scripts/mlflow_traces.py profile <trace_id> [<trace_id> ...] [--top 8]
```

Ranked by **self** time. Pass several ids to also get an aggregate table
(n / min / avg / p95 / max per span name) — that is the before/after comparison.

```
Trace tr-abc123def456  total 1.25s  [ERROR]
    840.0ms   67.2%  rerank
    340.0ms   27.2%  llm_generate
     50.0ms    4.0%  retrieve_docs
```

### diagnose — root-cause report for one trace

```
uv run scripts/mlflow_traces.py diagnose <trace_id>
```

Names the span the failure *originated* in, with its exception and inputs, then
the latency hotspots:

```
DIAGNOSE tr-abc123def456  [ERROR]  total 1.25s

FAILING SPANS (2):
  • llm_generate  [ERROR]  340.0ms
      exception.type   : RateLimitError
      exception.message: Rate limit exceeded for model gpt-4 (requested 31200)
      exception.stacktrace (tail):
          File "app/rag.py", line 88, in llm_generate
            response = client.chat.completions.create(model=model, messages=messages)
      inputs : {"model": "gpt-4", "prompt_tokens": 31200, ...}

  propagated through: predict

LATENCY HOTSPOTS (self time):
    840.0ms  ( 67%)  rerank
```

### errors — aggregate failures across an experiment

```
uv run scripts/mlflow_traces.py errors --experiment-id <id> [--since 24h] [--max-results 200]
uv run scripts/mlflow_traces.py errors --experiment-name my-app --filter "tag.environment = 'prod'"
```

Searches `trace.status = 'ERROR'` and groups by exception type + originating
span, collapsing messages that differ only in numbers or quoted values.
`--experiment-id`/`--experiment-name` are repeatable and fall back to
`MLFLOW_EXPERIMENT_ID`/`MLFLOW_EXPERIMENT_NAME`. `--since` takes `24h`, `7d`,
or an ISO datetime and is pushed into the server-side filter, so
`--max-results` (default 200) bounds traces *inside* the window; paging
continues up to that cap.

```
2 error group(s):

[1] ×17  RateLimitError  in span 'llm_generate'
      Rate limit exceeded for model gpt-4: 30000 tokens per minute (requested 31200)
      examples: tr-abc123def456, tr-def789abc012, tr-99aa11bb22cc (+14 more)
```

## Reading the output

| Element | Meaning |
|---|---|
| `✓` `✗` `·` `…` | span status: OK, ERROR, UNSET, in progress |
| `1.25s (self 20.0ms)` | total wall time, then time **excluding children** — self time is the real local cost, so the largest self time is the hotspot |
| `████` | that span's share of total trace duration |
| `[LLM  gpt-4  in 120 / out 30 / total 150  $0.0030]` | span type, model, tokens, cost — shown only when the provider reported them |
| `! Type: message` | an exception recorded on that span |
| `propagated through: predict` | ancestors that merely re-recorded the same exception; the span named above it is the origin |
| `×17` … `examples:` | occurrences in that error group, and trace IDs to feed to `diagnose` |
| lines starting `#` on stderr | informational notes (env loaded, result cap hit), not data |

Failures print an `error: ...` line on stderr and exit 1 (success exits 0);
backend exceptions add a `(set MLFLOW_TRACES_DEBUG=1 ...)` hint — set that
variable to get the full traceback instead.

## Workflows

**Fix a failing trace.** `diagnose <trace_id>` → read the exception type,
message, and stacktrace tail plus the originating span's inputs → map the top
stacktrace frame to the source file → propose a concrete code fix.

**Triage an experiment.** `errors --experiment-id <id> --since 24h` → take the
highest-count group → `diagnose` one of its example trace IDs.

**Profile a slow trace.** `profile <trace_id>` → the largest **self** time is
the hotspot. Run over several ids for the aggregate table to catch regressions.

**Audit cost.** `get <trace_id>` shows trace-level `tokens`/`cost` and per-LLM-span
model, tokens, and cost. Absent keys mean the provider integration didn't
report them, not that cost was zero.

## Setup

Connection comes entirely from the environment — nothing is hardcoded:

| Purpose | Environment variables, in order |
|---|---|
| Tracking URI | `MLFLOW_TRACKING_URI` → `MLFLOW_TRACKING_ARN` |
| Default experiment | `MLFLOW_EXPERIMENT_ID` → `MLFLOW_EXPERIMENT_NAME` |

- **Self-hosted:** `MLFLOW_TRACKING_URI=http://host:5000` (or a file/DB URI)
- **Databricks:** `MLFLOW_TRACKING_URI=databricks` (or `databricks://<profile>`)
  plus `DATABRICKS_HOST`/`DATABRICKS_TOKEN`, or a `~/.databrickscfg` profile
- **Per-call override:** `--tracking-uri <uri>` beats both the env and the `.env`

**`.env` auto-loading.** Walks up from the current directory, preferring
`backend/.env` over `.env` in each directory, and never overrides variables
already set in the real environment. Use `--env-file <path>` to point elsewhere,
`--no-env-file` to use the ambient environment only. Skipped entirely with
`--from-file`.

**ARN resolution.** If `MLFLOW_TRACKING_ARN` is a Secrets Manager ARN
(`arn:aws:secretsmanager:...`), the secret is fetched via boto3's default
credential chain; its JSON body must contain `MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`. A directly-set
`MLFLOW_TRACKING_URI` always wins and skips ARN resolution.

MLflow's own env vars still apply — e.g. `MLFLOW_TRACKING_INSECURE_TLS=true`
for a self-signed certificate.

## Offline / saved traces

`--from-file <path>` reads traces from a JSON dump instead of a live backend;
no credentials needed and the trace ID becomes optional:

```
uv run scripts/mlflow_traces.py get --from-file trace.json
uv run scripts/mlflow_traces.py diagnose --from-file trace.json
```

Accepts a single trace object, a bare list, or `{"traces": [...]}`, in either:

- **MLflow's own export** — `mlflow.get_trace(id).to_json()`, or a trace
  downloaded from the MLflow UI (OTel field names, span I/O inside
  `mlflow.span*` attributes)
- **This script's `--json` output**, which round-trips back in

Multi-trace dumps load in full, so `profile` gives its aggregate table and
`errors` groups across them offline. Passing ids alongside `--from-file` selects
those traces out of the dump.

## Troubleshooting

| Message | Fix |
|---|---|
| `no MLflow tracking URI found` | Set `MLFLOW_TRACKING_URI`, pass `--tracking-uri`, or run where the `.env` is discoverable |
| `no experiment specified` | Pass `--experiment-id`/`--experiment-name`, or set `MLFLOW_EXPERIMENT_ID`/`_NAME` |
| `experiment not found by name` | Name is case-sensitive and scoped to the tracking server; try `--experiment-id` |
| `mlflow is not available` | Run via `uv run`, or `pip install mlflow-skinny` |
| Auth/connection errors | Re-check the token or profile; `MLFLOW_TRACES_DEBUG=1` shows the full traceback |

## Reference & tests

- `references/trace-data-model.md` — TraceInfo/Span field map, token & cost
  attribute keys, the `Trace.to_json()` export shape, exception-event schema,
  and the underlying `MlflowClient` calls. Read it when interpreting raw
  attributes or extending the script.
- `tests/` — `uv run --with pytest python -m pytest tests -q` (offline; one
  integration test additionally exercises a real MLflow backend when installed).

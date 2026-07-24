---
name: mlflow-traces
description: >-
  Inspect, profile, and root-cause MLflow traces from the CLI. Use when you
  have one or more MLflow trace IDs (from logs, an error report, or the MLflow
  UI) and need span-level detail, a latency breakdown, error aggregation across
  traces, or a diagnosis to fix a failing trace.
license: MIT
metadata:
  requires: uv (recommended) or python3 with mlflow-skinny
---

# MLflow Traces

Turn an MLflow trace ID into answers: the full span waterfall, where the time
went, what threw, and what to fix.

## When to use

- You have a trace ID and want its detail / inputs / outputs / span tree → `get`
- A trace is slow and you want the latency hotspot → `profile`
- You want to triage errors across a whole experiment → `errors`
- A trace failed and you want to root-cause and fix it → `diagnose`

## Setup

The script needs an MLflow backend, taken entirely from the environment — it
never hardcodes one:

- **Self-hosted:** `export MLFLOW_TRACKING_URI=http://host:5000` (or a file/DB URI)
- **Databricks:** `export MLFLOW_TRACKING_URI=databricks` (or `databricks://<profile>`),
  with `DATABRICKS_HOST`/`DATABRICKS_TOKEN` or a `~/.databrickscfg` profile
- Optional default experiment for `errors`: `MLFLOW_EXPERIMENT_ID` or `MLFLOW_EXPERIMENT_NAME`

**`.env` auto-loading.** For any live command, the script walks up from the
current directory looking for `backend/.env` (then `.env`) and loads it, without
overriding variables already set in the environment. Point it elsewhere with
`--env-file <path>`, or disable with `--no-env-file`.

**Tracking-URI variable names** (the standard name is preferred):

| Purpose | Names checked, in order |
|---|---|
| Tracking URI | `MLFLOW_TRACKING_URI` → `MLFLOW_TRACKING_ARN` |
| Default experiment | `MLFLOW_EXPERIMENT_ID`, then `MLFLOW_EXPERIMENT_NAME` |

An ARN value is used verbatim as the tracking URI (no AWS-specific handling);
whatever plugin/credentials that URI needs must already be in the environment.

No install step: run through **uv** and the client is fetched on demand (the
script declares `mlflow-skinny>=3.13` via PEP 723). If `mlflow` is already
installed, plain `python3` works too. Targets the MLflow 3.x tracing API.

All commands are run from this skill directory:

```
uv run scripts/mlflow_traces.py <command> [args]
```

## Commands

Add `--json` to any command for machine-readable output you can parse. Flags go
**after** the subcommand.

**get** — full detail + span-tree waterfall (self time, status, latency bar):
```
uv run scripts/mlflow_traces.py get <trace_id> [<trace_id> ...]
uv run scripts/mlflow_traces.py get <trace_id> --errors-only        # only failing spans + their exceptions & inputs
uv run scripts/mlflow_traces.py get <trace_id> --max-io 2000        # widen input/output truncation
```

**profile** — per-span total & self time, % of trace, slowest-N. Pass several
IDs to get an aggregate table (n / min / avg / p95 / max per span name) for
spotting regressions:
```
uv run scripts/mlflow_traces.py profile <trace_id> [<trace_id> ...] [--top 8]
```

**errors** — search `trace.status = 'ERROR'` across an experiment and group by
exception type + failing span:
```
uv run scripts/mlflow_traces.py errors --experiment-id <id> [--since 24h] [--max-results 200]
uv run scripts/mlflow_traces.py errors --experiment-name my-app --filter "tag.environment = 'prod'"
```

**diagnose** — compact root-cause report for one trace: failing spans, each
`exception.{type,message,stacktrace}`, the failing span's inputs, and the
latency hotspots:
```
uv run scripts/mlflow_traces.py diagnose <trace_id>
```

## Workflows

**Fix a failing trace.** Run `diagnose <trace_id>`. Read the exception type,
message, and stacktrace tail, plus the failing span's inputs. Map the top
stacktrace frame to a file in the repo, form a root-cause hypothesis, then
propose a concrete code fix. The script surfaces the evidence; you do the
reasoning and the edit.

**Profile a slow trace.** Run `profile <trace_id>`. The span with the largest
**self** time is the hotspot (self time excludes children, so it's the real
local cost). Run `profile` over a few IDs to compare and catch regressions.

**Triage an experiment.** Run `errors --experiment-id <id> --since 24h`. Pick
the highest-count group, take one of its example trace IDs, and `diagnose` it.

## Offline / reuse a saved trace

`--from-file <path>` renders a trace from a JSON dump instead of a live backend
— handy for a trace you already exported, or the bundled sample:

```
uv run scripts/mlflow_traces.py get t --from-file fixtures/sample_trace.json
```

The `--json` output of `get` round-trips back in through `--from-file`.

## Reference & tests

- `references/trace-data-model.md` — Trace/Span field map (incl. v2↔v3 renames),
  exception-event schema, and the underlying `MlflowClient` calls.
- `tests/` — offline pytest over the pure helpers: `uv run --with pytest python -m pytest tests -q`.

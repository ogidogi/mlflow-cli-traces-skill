# mlflow-cli-traces-skill

An [opencode](https://opencode.ai) skill to work with **MLflow traces** from the
CLI. Given one or more trace IDs, it fetches full detail, profiles latency,
aggregates errors across traces, and produces a structured diagnosis so the
agent can root-cause and fix a failing trace.

## What's in here

```
.opencode/skills/mlflow-traces/
├── SKILL.md                     # manifest + agent instructions
├── scripts/mlflow_traces.py     # the CLI (single file, stdlib + mlflow-skinny via PEP 723)
├── references/trace-data-model.md
├── fixtures/sample_trace.json   # offline sample for demos/tests
└── tests/test_mlflow_traces.py
```

## Install

The skill is discovered by opencode as a **project skill** at
`.opencode/skills/mlflow-traces/`. Run `opencode` from a repo that contains this
folder and the `mlflow-traces` skill is available.

To use it in another project, copy the skill folder into that project:

```sh
cp -r .opencode/skills/mlflow-traces /path/to/other-project/.opencode/skills/
```

opencode also reads `~/.config/opencode/skills/` and `~/.claude/skills/`, so you
can symlink the folder there to make it global / cross-tool:

```sh
ln -s "$PWD/.opencode/skills/mlflow-traces" ~/.config/opencode/skills/mlflow-traces
```

## Requirements

- [uv](https://docs.astral.sh/uv/) (recommended) — the script declares
  `mlflow-skinny>=3.13` via a PEP 723 header, so `uv run` installs it on demand.
  No global setup.
- Or `python3` with `mlflow`/`mlflow-skinny` (3.13+) already installed.

Targets the MLflow 3.x tracing API (3.13/3.14); older trace protocols are not
supported.

Point it at a backend via the standard environment — nothing is hardcoded:

```sh
export MLFLOW_TRACKING_URI=http://localhost:5000     # self-hosted
# or
export MLFLOW_TRACKING_URI=databricks                # Databricks (with DATABRICKS_* / ~/.databrickscfg)
# or set MLFLOW_TRACKING_ARN — used verbatim as the tracking URI when URI is unset
```

For any live command the script also auto-loads a `.env`, searching `backend/.env`
then `.env` upward from the working directory (without overriding vars already in
the environment). Override the location with `--env-file <path>`, or skip it with
`--no-env-file`. The default experiment for `errors` comes from
`MLFLOW_EXPERIMENT_ID` / `MLFLOW_EXPERIMENT_NAME`.

## Usage

From `.opencode/skills/mlflow-traces/`:

```sh
uv run scripts/mlflow_traces.py get <trace_id>                    # detail + span waterfall
uv run scripts/mlflow_traces.py profile <trace_id> [<id> ...]     # latency breakdown / regression
uv run scripts/mlflow_traces.py errors --experiment-id <id>       # error aggregation
uv run scripts/mlflow_traces.py diagnose <trace_id>               # root-cause report
```

Add `--json` to any command for machine-readable output. Try it offline against
the bundled sample:

```sh
uv run scripts/mlflow_traces.py diagnose t --from-file fixtures/sample_trace.json
```

## Tests

```sh
uv run --with pytest python -m pytest tests -q
```

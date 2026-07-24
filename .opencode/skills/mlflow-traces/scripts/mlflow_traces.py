#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["mlflow-skinny>=3.13"]
# ///
"""Operate on MLflow traces from the CLI.

Targets the MLflow 3.x tracing API (3.13+). Given one or more trace IDs, fetch
full detail, profile latency, aggregate errors across traces, and emit a
structured diagnose report for root-causing.

Backend is whatever the environment already points at: MLFLOW_TRACKING_URI
(self-hosted server or `databricks[://profile]`) plus the usual Databricks
auth (DATABRICKS_HOST/DATABRICKS_TOKEN or ~/.databrickscfg). Nothing is
hardcoded. `mlflow` is imported lazily, so --from-file works with no backend.

Run with uv (zero-install):  uv run mlflow_traces.py <cmd> ...
Or, if mlflow is installed:   python mlflow_traces.py <cmd> ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Normalized data model
#
# Live MLflow Trace objects and JSON-loaded dicts are both converted into
# these plain structures so every downstream helper is trivially testable and
# version-independent. The JSON shape emitted by --json round-trips back in
# through --from-file.
# --------------------------------------------------------------------------


@dataclass
class NEvent:
    name: str
    timestamp_ns: Optional[int]
    attributes: Dict[str, Any]


@dataclass
class NSpan:
    span_id: str
    parent_id: Optional[str]
    name: str
    start_ns: Optional[int]
    end_ns: Optional[int]
    status: str
    status_message: str
    inputs: Any
    outputs: Any
    attributes: Dict[str, Any]
    events: List[NEvent] = field(default_factory=list)

    @property
    def total_ms(self) -> Optional[float]:
        if self.start_ns is None or self.end_ns is None:
            return None
        return max(0.0, (self.end_ns - self.start_ns) / 1_000_000)


@dataclass
class NTrace:
    trace_id: str
    state: str
    duration_ms: Optional[float]
    request_time_ms: Optional[int]
    request_preview: str
    response_preview: str
    tags: Dict[str, Any]
    metadata: Dict[str, Any]
    spans: List[NSpan]


# --------------------------------------------------------------------------
# Field access that tolerates both attribute objects and dicts
# --------------------------------------------------------------------------


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Return the first present, non-None value among ``names``.

    Works whether ``obj`` is an attribute-style object (a live MLflow 3.x
    entity) or a plain dict (our own --json / fixture shape). The two names per
    field are the live attribute name and our JSON key, never legacy aliases.
    """
    for name in names:
        if obj is None:
            break
        if isinstance(obj, dict):
            if obj.get(name) is not None:
                return obj[name]
        else:
            val = getattr(obj, name, None)
            if val is not None:
                return val
    return default


def _short_status(raw: Any) -> str:
    """Collapse OK/ERROR/UNSET/state enums to a short upper token."""
    if raw is None:
        return "UNKNOWN"
    val = getattr(raw, "value", raw)
    s = str(val)
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return s.upper() or "UNKNOWN"


def _as_obj(value: Any) -> Any:
    """Parse a JSON string into an object; leave non-strings untouched."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


def _to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    # MLflow entity dict-likes sometimes expose .to_dict()
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
            if isinstance(result, dict):
                return result
        except Exception:  # noqa: BLE001 - best effort
            pass
    return {}


# --------------------------------------------------------------------------
# Normalization: live MLflow objects / dict exports  ->  N* structures
# --------------------------------------------------------------------------


def normalize_event(raw: Any) -> NEvent:
    return NEvent(
        name=str(_get(raw, "name", default="")),
        timestamp_ns=_get(raw, "timestamp_ns", "timestamp"),
        attributes=_to_dict(_get(raw, "attributes", default={})),
    )


def normalize_span(raw: Any) -> NSpan:
    # Live span: ``status`` is a SpanStatus(status_code=<SpanStatusCode>, description=...).
    # Our --json: ``status`` is a plain string and ``status_message`` is a sibling.
    status = _get(raw, "status")
    if status is None or isinstance(status, str):
        status_code = status
        status_message = _get(raw, "status_message", default="")
    else:
        status_code = _get(status, "status_code")
        status_message = _get(status, "description") or _get(raw, "status_message", default="")

    events = [normalize_event(e) for e in (_get(raw, "events", default=[]) or [])]

    return NSpan(
        span_id=str(_get(raw, "span_id", default="")),
        parent_id=_get(raw, "parent_id"),
        name=str(_get(raw, "name", default="<unnamed>")),
        start_ns=_get(raw, "start_time_ns", "start_ns"),
        end_ns=_get(raw, "end_time_ns", "end_ns"),
        status=_short_status(status_code),
        status_message=str(status_message or ""),
        inputs=_as_obj(_get(raw, "inputs")),
        outputs=_as_obj(_get(raw, "outputs")),
        attributes=_to_dict(_get(raw, "attributes", default={})),
        events=events,
    )


def _extract_spans(trace: Any) -> List[Any]:
    # Live object: trace.data.spans ; flat dict: trace["spans"] ;
    # export dict: trace["data"]["spans"].
    data = _get(trace, "data")
    spans = _get(data, "spans") if data is not None else None
    if spans is None:
        spans = _get(trace, "spans")
    return list(spans or [])


def normalize_trace(trace: Any) -> NTrace:
    """Convert a live MLflow 3.x Trace or our flat --json/fixture dict.

    A live Trace / export dict keeps its metadata under ``info``; our flat dict
    carries the same fields at the top level.
    """
    info = _get(trace, "info")
    src = info if info is not None else trace

    duration = _get(src, "execution_duration", "duration_ms")
    spans = [normalize_span(s) for s in _extract_spans(trace)]

    return NTrace(
        trace_id=str(_get(src, "trace_id", default="")),
        state=_short_status(_get(src, "state")),
        duration_ms=float(duration) if duration is not None else None,
        request_time_ms=_get(src, "request_time", "request_time_ms"),
        request_preview=str(_get(src, "request_preview", default="") or ""),
        response_preview=str(_get(src, "response_preview", default="") or ""),
        tags=_to_dict(_get(src, "tags", default={})),
        metadata=_to_dict(_get(src, "trace_metadata", "metadata", default={})),
        spans=spans,
    )


# --------------------------------------------------------------------------
# Pure analysis helpers (fully covered by tests)
# --------------------------------------------------------------------------


def build_span_forest(spans: List[NSpan]) -> Tuple[List[NSpan], Dict[str, List[NSpan]]]:
    """Return (root spans, parent_id -> children). Roots have no known parent."""
    by_id = {s.span_id: s for s in spans}
    children: Dict[str, List[NSpan]] = {}
    roots: List[NSpan] = []
    for s in spans:
        if s.parent_id and s.parent_id in by_id:
            children.setdefault(s.parent_id, []).append(s)
        else:
            roots.append(s)

    def _start(sp: NSpan) -> int:
        return sp.start_ns if sp.start_ns is not None else 0

    roots.sort(key=_start)
    for kids in children.values():
        kids.sort(key=_start)
    return roots, children


def span_self_ms(span: NSpan, children: Dict[str, List[NSpan]]) -> Optional[float]:
    """Total minus the sum of direct children's total (clamped at 0)."""
    total = span.total_ms
    if total is None:
        return None
    child_total = 0.0
    for child in children.get(span.span_id, []):
        child_total += child.total_ms or 0.0
    return max(0.0, total - child_total)


def span_exceptions(span: NSpan) -> List[Dict[str, str]]:
    """Exception records recorded on a span as OpenTelemetry events."""
    out: List[Dict[str, str]] = []
    for ev in span.events:
        attrs = ev.attributes or {}
        is_exc = ev.name == "exception" or any(k.startswith("exception.") for k in attrs)
        if not is_exc:
            continue
        out.append(
            {
                "type": str(attrs.get("exception.type", "") or ""),
                "message": str(attrs.get("exception.message", "") or ""),
                "stacktrace": str(attrs.get("exception.stacktrace", "") or ""),
            }
        )
    return out


def error_spans(trace: NTrace) -> List[NSpan]:
    return [s for s in trace.spans if s.status == "ERROR" or span_exceptions(s)]


def slowest_spans(trace: NTrace, top_n: int = 5) -> List[Tuple[NSpan, Optional[float]]]:
    _, children = build_span_forest(trace.spans)
    ranked = [(s, span_self_ms(s, children)) for s in trace.spans]
    ranked.sort(key=lambda pair: (pair[1] is None, -(pair[1] or 0.0)))
    return ranked[:top_n]


def _normalize_message(msg: str, limit: int = 100) -> str:
    """Collapse volatile bits (numbers, hex, quoted values) so similar
    exception messages group together."""
    msg = msg.strip().splitlines()[0] if msg.strip() else ""
    msg = re.sub(r"0x[0-9a-fA-F]+", "0x#", msg)
    msg = re.sub(r"\d+", "#", msg)
    msg = re.sub(r"'[^']*'", "'…'", msg)
    msg = re.sub(r'"[^"]*"', '"…"', msg)
    msg = re.sub(r"\s+", " ", msg)
    return msg[:limit]


def group_errors(traces: List[NTrace]) -> List[Dict[str, Any]]:
    """Group erroring traces by (exception type, failing span, message shape).

    Prefer spans that carry an actual exception event; only fall back to
    error-status spans (e.g. a parent whose ERROR is purely propagated from a
    failing child) when the trace has no exception events at all.
    """
    groups: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for tr in traces:
        buckets = []
        spans_with_exc = [s for s in tr.spans if span_exceptions(s)]
        if spans_with_exc:
            for span in spans_with_exc:
                for exc in span_exceptions(span):
                    buckets.append((exc["type"] or "<no-type>", span.name, exc["message"]))
        else:
            for span in error_spans(tr):
                buckets.append(("<no-exception-event>", span.name, span.status_message))
        if not buckets:
            # Trace-level error without any error span.
            buckets.append(("<trace-level>", "<trace>", tr.response_preview))

        for exc_type, span_name, raw_msg in buckets:
            key = (exc_type, span_name, _normalize_message(raw_msg))
            g = groups.setdefault(
                key,
                {
                    "exception_type": exc_type,
                    "span": span_name,
                    "message": _normalize_message(raw_msg),
                    "example_message": raw_msg.strip().splitlines()[0] if raw_msg.strip() else "",
                    "count": 0,
                    "trace_ids": [],
                },
            )
            g["count"] += 1
            if tr.trace_id not in g["trace_ids"]:
                g["trace_ids"].append(tr.trace_id)

    return sorted(groups.values(), key=lambda g: g["count"], reverse=True)


# --------------------------------------------------------------------------
# Serialization (stable --json shape; also what --from-file reads)
# --------------------------------------------------------------------------


def span_to_dict(span: NSpan) -> Dict[str, Any]:
    return {
        "span_id": span.span_id,
        "parent_id": span.parent_id,
        "name": span.name,
        "start_ns": span.start_ns,
        "end_ns": span.end_ns,
        "duration_ms": span.total_ms,
        "status": span.status,
        "status_message": span.status_message,
        "inputs": span.inputs,
        "outputs": span.outputs,
        "attributes": span.attributes,
        "events": [
            {"name": e.name, "timestamp_ns": e.timestamp_ns, "attributes": e.attributes}
            for e in span.events
        ],
    }


def trace_to_dict(trace: NTrace) -> Dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "state": trace.state,
        "duration_ms": trace.duration_ms,
        "request_time_ms": trace.request_time_ms,
        "request_preview": trace.request_preview,
        "response_preview": trace.response_preview,
        "tags": trace.tags,
        "metadata": trace.metadata,
        "spans": [span_to_dict(s) for s in trace.spans],
    }


# --------------------------------------------------------------------------
# Text rendering
# --------------------------------------------------------------------------

_STATUS_MARK = {"OK": "✓", "ERROR": "✗", "UNSET": "·", "IN_PROGRESS": "…"}


def _stringify(value: Any, max_len: Optional[int] = None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        s = value
    else:
        try:
            s = json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            s = str(value)
    if max_len is not None and len(s) > max_len:
        s = s[:max_len] + f"… (+{len(s) - max_len} chars)"
    return s


def _fmt_ms(ms: Optional[float]) -> str:
    if ms is None:
        return "   n/a"
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    return f"{ms:.1f}ms"


def render_span_tree(trace: NTrace, errors_only: bool = False, max_io: int = 500) -> str:
    roots, children = build_span_forest(trace.spans)
    total = trace.duration_ms or max((s.total_ms or 0.0) for s in trace.spans) if trace.spans else 0.0
    lines: List[str] = []

    def visit(span: NSpan, depth: int) -> None:
        exc = span_exceptions(span)
        show = (not errors_only) or span.status == "ERROR" or exc
        if show:
            mark = _STATUS_MARK.get(span.status, "?")
            total_ms = span.total_ms or 0.0
            bar_w = int((total_ms / total) * 20) if total else 0
            bar = "█" * bar_w
            indent = "  " * depth
            self_ms = span_self_ms(span, children)
            lines.append(
                f"{indent}{mark} {span.name}  "
                f"{_fmt_ms(span.total_ms)} (self {_fmt_ms(self_ms)})  {bar}"
            )
            if span.status == "ERROR" and span.status_message:
                lines.append(f"{indent}    ! {span.status_message}")
            for e in exc:
                lines.append(f"{indent}    ! {e['type']}: {e['message']}")
            if errors_only and (span.status == "ERROR" or exc):
                if span.inputs is not None:
                    lines.append(f"{indent}    inputs: {_stringify(span.inputs, max_io)}")
        for child in children.get(span.span_id, []):
            visit(child, depth + 1)

    for root in roots:
        visit(root, 0)
    return "\n".join(lines) if lines else "  (no spans)"


def render_trace_detail(trace: NTrace, errors_only: bool = False, max_io: int = 500) -> str:
    mark = _STATUS_MARK.get(trace.state, "?")
    out = [
        f"Trace {trace.trace_id}  [{mark} {trace.state}]",
        f"  duration : {_fmt_ms(trace.duration_ms)}",
    ]
    if trace.request_time_ms:
        out.append(f"  started  : {_fmt_epoch_ms(trace.request_time_ms)}")
    if trace.request_preview:
        out.append(f"  request  : {_stringify(trace.request_preview, max_io)}")
    if trace.response_preview:
        out.append(f"  response : {_stringify(trace.response_preview, max_io)}")
    if trace.tags:
        out.append(f"  tags     : {_stringify(trace.tags, max_io)}")
    if trace.metadata:
        out.append(f"  metadata : {_stringify(trace.metadata, max_io)}")
    out.append(f"  spans ({len(trace.spans)}):")
    out.append(render_span_tree(trace, errors_only=errors_only, max_io=max_io))
    return "\n".join(out)


def render_profile(traces: List[NTrace], top_n: int = 8) -> str:
    out: List[str] = []
    for trace in traces:
        out.append(f"Trace {trace.trace_id}  total {_fmt_ms(trace.duration_ms)}  [{trace.state}]")
        for span, self_ms in slowest_spans(trace, top_n):
            pct = ""
            if self_ms is not None and trace.duration_ms:
                pct = f"{(self_ms / trace.duration_ms) * 100:5.1f}%"
            out.append(
                f"  {self_ms if self_ms is None else round(self_ms, 1):>8}  "
                f"{pct:>6}  {span.name}"
            )
        out.append("")

    if len(traces) > 1:
        out.append(_render_profile_aggregate(traces))
    return "\n".join(out).rstrip()


def _render_profile_aggregate(traces: List[NTrace]) -> str:
    by_name: Dict[str, List[float]] = {}
    for trace in traces:
        _, children = build_span_forest(trace.spans)
        for span in trace.spans:
            self_ms = span_self_ms(span, children)
            if self_ms is not None:
                by_name.setdefault(span.name, []).append(self_ms)

    lines = [f"Aggregate self-time by span name across {len(traces)} traces:", ""]
    lines.append(f"  {'span':<32} {'n':>3} {'min':>8} {'avg':>8} {'p95':>8} {'max':>8}")
    rows = []
    for name, vals in by_name.items():
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        p95 = vals_sorted[min(n - 1, int(round(0.95 * (n - 1))))]
        rows.append(
            (
                sum(vals) / n,
                f"  {name[:32]:<32} {n:>3} "
                f"{min(vals):>8.1f} {sum(vals) / n:>8.1f} {p95:>8.1f} {max(vals):>8.1f}",
            )
        )
    rows.sort(key=lambda r: r[0], reverse=True)
    lines.extend(r[1] for r in rows)
    return "\n".join(lines)


def render_errors(groups: List[Dict[str, Any]]) -> str:
    if not groups:
        return "No error traces found."
    out = [f"{len(groups)} error group(s):", ""]
    for i, g in enumerate(groups, 1):
        examples = ", ".join(g["trace_ids"][:3])
        more = f" (+{len(g['trace_ids']) - 3} more)" if len(g["trace_ids"]) > 3 else ""
        out.append(f"[{i}] ×{g['count']}  {g['exception_type']}  in span '{g['span']}'")
        if g["example_message"]:
            out.append(f"      {g['example_message'][:200]}")
        out.append(f"      examples: {examples}{more}")
        out.append("")
    return "\n".join(out).rstrip()


def render_diagnose(trace: NTrace) -> str:
    """Structured, compact report meant to be read by the agent to root-cause."""
    out = [f"DIAGNOSE {trace.trace_id}  [{trace.state}]  total {_fmt_ms(trace.duration_ms)}", ""]

    errs = error_spans(trace)
    if errs:
        out.append(f"FAILING SPANS ({len(errs)}):")
        for span in errs:
            out.append(f"  • {span.name}  [{span.status}]  {_fmt_ms(span.total_ms)}")
            if span.status_message:
                out.append(f"      status: {span.status_message}")
            for exc in span_exceptions(span):
                out.append(f"      exception.type   : {exc['type']}")
                out.append(f"      exception.message: {exc['message']}")
                if exc["stacktrace"]:
                    tail = "\n".join(exc["stacktrace"].strip().splitlines()[-12:])
                    out.append("      exception.stacktrace (tail):")
                    out.extend("        " + ln for ln in tail.splitlines())
            if span.inputs is not None:
                out.append(f"      inputs : {_stringify(span.inputs, 800)}")
            if span.outputs is not None:
                out.append(f"      outputs: {_stringify(span.outputs, 400)}")
            out.append("")
    else:
        out.append("No error spans. This trace did not fail; review latency below.")
        out.append("")

    out.append("LATENCY HOTSPOTS (self time):")
    for span, self_ms in slowest_spans(trace, 5):
        pct = f"{(self_ms / trace.duration_ms) * 100:.0f}%" if self_ms and trace.duration_ms else "n/a"
        out.append(f"  {_fmt_ms(self_ms):>9}  ({pct:>4})  {span.name}")
    out.append("")
    out.append(
        "NEXT: identify the root cause from the exception + failing-span inputs "
        "above, then propose a concrete code fix."
    )
    return "\n".join(out)


def _fmt_epoch_ms(ms: Any) -> str:
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat()
    except (ValueError, TypeError, OSError):
        return str(ms)


# --------------------------------------------------------------------------
# Environment / .env loading and config resolution
#
# The backend is chosen entirely from env vars. Alongside the standard MLflow
# names we accept one local variant: an ARN used directly as the tracking URI.
# --------------------------------------------------------------------------

# Tracking URI: standard name first, then the ARN fallback. An ARN, when
# present, is used verbatim as the tracking URI (no AWS-specific handling).
_TRACKING_URI_KEYS = ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_ARN")
_EXPERIMENT_NAME_KEYS = ("MLFLOW_EXPERIMENT_NAME",)


def parse_env_file(path: str) -> Dict[str, str]:
    """Parse a minimal ``KEY=VALUE`` .env file (stdlib only).

    Ignores blank lines and ``#`` comments, tolerates a leading ``export ``,
    and strips one layer of matching single/double quotes.
    """
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            if key:
                out[key] = val
    return out


def find_env_file(start: Optional[str] = None) -> Optional[str]:
    """Walk up from ``start`` (default: CWD) looking for ``backend/.env`` then
    ``.env`` in each directory. Return the first match, or None."""
    directory = os.path.abspath(start or os.getcwd())
    while True:
        for candidate in (os.path.join(directory, "backend", ".env"), os.path.join(directory, ".env")):
            if os.path.isfile(candidate):
                return candidate
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


def load_env_file(explicit_path: Optional[str] = None) -> Optional[str]:
    """Load a .env into ``os.environ`` without overriding existing vars.

    Uses ``explicit_path`` if given, else auto-discovers via ``find_env_file``.
    Returns the path loaded, or None.
    """
    path = explicit_path or find_env_file()
    if not path or not os.path.isfile(path):
        if explicit_path:
            sys.exit(f"error: --env-file not found: {explicit_path}")
        return None
    for key, val in parse_env_file(path).items():
        os.environ.setdefault(key, val)  # real environment wins
    print(f"# mlflow-traces: loaded env from {path}", file=sys.stderr)
    return path


def resolve_tracking_uri(cli_value: Optional[str]) -> Optional[str]:
    """CLI flag first, then the standard/variant tracking env vars in order."""
    if cli_value:
        return cli_value
    for key in _TRACKING_URI_KEYS:
        val = os.environ.get(key)
        if val:
            return val
    return None


def env_experiment_name() -> Optional[str]:
    for key in _EXPERIMENT_NAME_KEYS:
        val = os.environ.get(key)
        if val:
            return val
    return None


# --------------------------------------------------------------------------
# MLflow access layer (imported lazily; not needed for --from-file)
# --------------------------------------------------------------------------


def _import_mlflow():
    try:
        import mlflow  # noqa: F401
        from mlflow import MlflowClient

        return mlflow, MlflowClient
    except ImportError:
        sys.exit(
            "error: mlflow is not available. Run this script with uv "
            "(`uv run mlflow_traces.py ...`) which auto-installs mlflow-skinny, "
            "or `pip install mlflow-skinny`."
        )


def get_client(tracking_uri: Optional[str]):
    _, MlflowClient = _import_mlflow()
    uri = resolve_tracking_uri(tracking_uri)
    if not uri:
        sys.exit(
            "error: no MLflow tracking URI found. Set MLFLOW_TRACKING_URI (or "
            "MLFLOW_TRACKING_ARN), pass --tracking-uri, or run where backend/.env "
            "is discoverable (or use --env-file)."
        )
    os.environ.setdefault("MLFLOW_TRACKING_URI", uri)  # keep fluent API in sync
    return MlflowClient(tracking_uri=uri)


def fetch_trace(client, trace_id: str) -> NTrace:
    trace = client.get_trace(trace_id)
    return normalize_trace(trace)


def resolve_experiment_ids(
    client, experiment_ids: List[str], experiment_names: List[str]
) -> List[str]:
    ids = list(experiment_ids or [])
    for name in experiment_names or []:
        exp = client.get_experiment_by_name(name)
        if exp is None:
            sys.exit(f"error: experiment not found by name: {name}")
        ids.append(exp.experiment_id)
    if not ids:
        env_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
        env_name = env_experiment_name()
        if env_id:
            ids.append(env_id)
        elif env_name:
            exp = client.get_experiment_by_name(env_name)
            if exp is None:
                sys.exit(f"error: experiment not found by name: {env_name}")
            ids.append(exp.experiment_id)
    if not ids:
        sys.exit(
            "error: no experiment specified. Pass --experiment-id/--experiment-name, "
            "or set MLFLOW_EXPERIMENT_ID / MLFLOW_EXPERIMENT_NAME."
        )
    return ids


def search_error_traces(
    client, experiment_ids: List[str], extra_filter: Optional[str], max_results: int
) -> List[NTrace]:
    filter_string = "trace.status = 'ERROR'"
    if extra_filter:
        filter_string = f"{filter_string} AND {extra_filter}"

    # `experiment_ids` works on 3.13 and 3.14 (deprecated in favour of
    # `locations` in 3.14); silence that FutureWarning to keep output clean.
    # order_by is best-effort: retry without it if the field is unsupported.
    last: Optional[Exception] = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        for order_by in (["timestamp_ms DESC"], None):
            try:
                kwargs: Dict[str, Any] = {
                    "experiment_ids": experiment_ids,
                    "filter_string": filter_string,
                    "max_results": max_results,
                }
                if order_by is not None:
                    kwargs["order_by"] = order_by
                results = client.search_traces(**kwargs)
                return [normalize_trace(t) for t in results]
            except Exception as exc:  # retry without order_by, then surface
                last = exc
    raise last  # type: ignore[misc]


def load_trace_from_file(path: str) -> NTrace:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        data = data[0]
    return normalize_trace(data)


def _cutoff_ms(since: Optional[str]) -> Optional[int]:
    """Parse --since as ISO datetime or a relative span like 24h / 7d / 30m."""
    if not since:
        return None
    m = re.fullmatch(r"(\d+)([smhdw])", since.strip())
    if m:
        qty = int(m.group(1))
        unit = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}[m.group(2)]
        dt = datetime.now(tz=timezone.utc) - timedelta(**{unit: qty})
        return int(dt.timestamp() * 1000)
    try:
        dt = datetime.fromisoformat(since)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError:
        sys.exit(f"error: could not parse --since '{since}' (use ISO datetime or e.g. 24h, 7d)")


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------


def _load_traces(args, trace_ids: List[str]) -> List[NTrace]:
    if args.from_file:
        return [load_trace_from_file(args.from_file)]
    client = get_client(args.tracking_uri)
    return [fetch_trace(client, tid) for tid in trace_ids]


def cmd_get(args) -> int:
    traces = _load_traces(args, args.trace_ids)
    if args.json:
        print(json.dumps([trace_to_dict(t) for t in traces], indent=2, default=str))
    else:
        print("\n\n".join(render_trace_detail(t, args.errors_only, args.max_io) for t in traces))
    return 0


def cmd_profile(args) -> int:
    traces = _load_traces(args, args.trace_ids)
    if args.json:
        out = []
        for t in traces:
            _, children = build_span_forest(t.spans)
            out.append(
                {
                    "trace_id": t.trace_id,
                    "duration_ms": t.duration_ms,
                    "spans": [
                        {
                            "name": s.name,
                            "total_ms": s.total_ms,
                            "self_ms": span_self_ms(s, children),
                            "status": s.status,
                        }
                        for s in t.spans
                    ],
                }
            )
        print(json.dumps(out, indent=2, default=str))
    else:
        print(render_profile(traces, args.top))
    return 0


def cmd_diagnose(args) -> int:
    traces = _load_traces(args, [args.trace_id])
    trace = traces[0]
    if args.json:
        payload = trace_to_dict(trace)
        payload["error_groups"] = group_errors([trace])
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(render_diagnose(trace))
    return 0


def cmd_errors(args) -> int:
    if args.from_file:
        traces = [load_trace_from_file(args.from_file)]
    else:
        client = get_client(args.tracking_uri)
        exp_ids = resolve_experiment_ids(client, args.experiment_id, args.experiment_name)
        traces = search_error_traces(client, exp_ids, args.filter, args.max_results)

    cutoff = _cutoff_ms(args.since)
    if cutoff is not None:
        traces = [t for t in traces if (t.request_time_ms or 0) >= cutoff]

    groups = group_errors(traces)
    if args.json:
        print(json.dumps({"total_traces": len(traces), "groups": groups}, indent=2, default=str))
    else:
        print(render_errors(groups))
    return 0


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlflow_traces.py",
        description="Fetch, profile, and root-cause MLflow traces from the CLI.",
    )

    # Common options live on a parent parser so they may appear after the
    # subcommand (e.g. `get <id> --json`), matching normal CLI ergonomics.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tracking-uri", help="Override MLFLOW_TRACKING_URI for this call.")
    common.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    common.add_argument(
        "--from-file",
        metavar="PATH",
        help="Read a trace from a JSON dump instead of a live backend (offline).",
    )
    common.add_argument(
        "--env-file",
        metavar="PATH",
        help="Load this .env instead of auto-discovering backend/.env upward from CWD.",
    )
    common.add_argument(
        "--no-env-file",
        action="store_true",
        help="Do not load any .env file; use the ambient environment only.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_get = sub.add_parser(
        "get", parents=[common], help="Full detail + span-tree waterfall for trace(s)."
    )
    p_get.add_argument("trace_ids", nargs="+")
    p_get.add_argument("--errors-only", action="store_true", help="Show only error spans.")
    p_get.add_argument("--max-io", type=int, default=500, help="Truncate inputs/outputs (chars).")
    p_get.set_defaults(func=cmd_get)

    p_prof = sub.add_parser(
        "profile", parents=[common], help="Latency breakdown; aggregate across traces."
    )
    p_prof.add_argument("trace_ids", nargs="+")
    p_prof.add_argument("--top", type=int, default=8, help="Show N slowest spans per trace.")
    p_prof.set_defaults(func=cmd_profile)

    p_diag = sub.add_parser(
        "diagnose", parents=[common], help="Structured root-cause report for one trace."
    )
    p_diag.add_argument("trace_id")
    p_diag.set_defaults(func=cmd_diagnose)

    p_err = sub.add_parser(
        "errors", parents=[common], help="Aggregate ERROR traces across an experiment."
    )
    p_err.add_argument("--experiment-id", action="append", default=[], help="Repeatable.")
    p_err.add_argument("--experiment-name", action="append", default=[], help="Repeatable.")
    p_err.add_argument("--filter", help="Extra MLflow filter DSL, AND-ed with status=ERROR.")
    p_err.add_argument("--since", help="Only traces newer than this (ISO datetime or e.g. 24h, 7d).")
    p_err.add_argument("--max-results", type=int, default=200)
    p_err.set_defaults(func=cmd_errors)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # A live backend needs config; auto-load a .env unless we're offline or opted out.
    if not args.from_file and not args.no_env_file:
        load_env_file(args.env_file)
    try:
        return args.func(args)
    except Exception as exc:  # backend/connection/auth errors → one clean line
        if os.environ.get("MLFLOW_TRACES_DEBUG"):
            raise
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("       (set MLFLOW_TRACES_DEBUG=1 for the full traceback)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

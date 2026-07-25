#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["mlflow-skinny>=3.13", "boto3>=1.34"]
# ///
"""Operate on MLflow traces from the CLI.

Targets the MLflow 3.x tracing API (3.13+). Given one or more trace IDs, fetch
full detail, profile latency, aggregate errors across traces, and emit a
structured diagnose report for root-causing.

Backend is whatever the environment already points at: MLFLOW_TRACKING_URI
(self-hosted server or `databricks[://profile]`) plus the usual Databricks
auth (DATABRICKS_HOST/DATABRICKS_TOKEN or ~/.databrickscfg). If
MLFLOW_TRACKING_ARN is a Secrets Manager ARN, it's resolved via boto3 (using
whatever AWS credentials are already in the environment) into
MLFLOW_TRACKING_URI/_USERNAME/_PASSWORD. Nothing is hardcoded. `mlflow` is
imported lazily, so --from-file works with no backend — and it reads both
MLflow's own `Trace.to_json()` export and this script's --json output.

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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Normalized data model
#
# Live MLflow Trace objects and JSON-loaded dicts are both converted into
# these plain structures so every downstream helper is trivially testable and
# version-independent. Three input shapes are supported (see normalize_span):
# a live MLflow 3.x entity, MLflow's own ``Trace.to_json()`` export (OTel
# field names), and the flat shape this script emits with --json — all three
# read back in through --from-file.
# --------------------------------------------------------------------------

# Attribute/metadata keys MLflow actually writes (mlflow.tracing.constant,
# verified against mlflow 3.14). Span I/O lives in attributes on the JSON
# export, which is why they are pulled out here rather than assumed present
# as top-level fields.
_ATTR_INPUTS = "mlflow.spanInputs"
_ATTR_OUTPUTS = "mlflow.spanOutputs"
_ATTR_SPAN_TYPE = "mlflow.spanType"
_ATTR_TOKENS = "mlflow.chat.tokenUsage"
_ATTR_COST = "mlflow.llm.cost"
_ATTR_MODEL = "mlflow.llm.model"
_META_TOKENS = "mlflow.trace.tokenUsage"
_META_COST = "mlflow.trace.cost"


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
    span_type: str = ""
    model: str = ""
    tokens: Dict[str, Any] = field(default_factory=dict)
    cost_usd: Optional[float] = None

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
    tokens: Dict[str, Any] = field(default_factory=dict)
    cost_usd: Optional[float] = None


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
    if "." in s:  # str(SpanStatusCode.ERROR) == "SpanStatusCode.ERROR"
        s = s.rsplit(".", 1)[-1]
    s = s.upper()
    # MLflow's JSON export uses the OTel proto spelling.
    for prefix in ("STATUS_CODE_", "TRACE_STATE_", "STATE_"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    return s or "UNKNOWN"


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


def _decode_attributes(raw: Any) -> Dict[str, Any]:
    """Attributes as a plain dict, with JSON-encoded values decoded.

    A live span hands back decoded values; MLflow's JSON export stores every
    attribute value as a JSON string (``"mlflow.spanType": "\\"LLM\\""``).
    Decoding here makes both shapes identical for everything downstream.
    """
    return {key: _as_obj(val) for key, val in _to_dict(raw).items()}


def _first_float(*values: Any) -> Optional[float]:
    """First value coercible to float; dicts contribute their ``total_*`` key.

    MLflow reports cost as ``{"input_cost": .., "output_cost": .., "total_cost": ..}``.
    """
    for val in values:
        if isinstance(val, dict):
            val = val.get("total_cost", val.get("total"))
        if val is None or isinstance(val, bool):
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def normalize_event(raw: Any) -> NEvent:
    return NEvent(
        name=str(_get(raw, "name", default="")),
        timestamp_ns=_get(raw, "timestamp_ns", "timestamp", "time_unix_nano"),
        attributes=_to_dict(_get(raw, "attributes", default={})),
    )


def normalize_span(raw: Any) -> NSpan:
    # ``status`` arrives in three shapes:
    #   live span   -> SpanStatus(status_code=<SpanStatusCode>, description=...)
    #   MLflow JSON -> {"code": "STATUS_CODE_ERROR", "message": ...}
    #   our --json  -> a plain string, with ``status_message`` as a sibling
    status = _get(raw, "status")
    if status is None or isinstance(status, str):
        status_code = status
        status_message = _get(raw, "status_message", default="")
    else:
        status_code = _get(status, "status_code", "code")
        status_message = _get(status, "description", "message") or _get(
            raw, "status_message", default=""
        )

    events = [normalize_event(e) for e in (_get(raw, "events", default=[]) or [])]
    attributes = _decode_attributes(_get(raw, "attributes", default={}))

    # Live spans expose inputs/outputs as properties; the JSON export only has
    # them under the corresponding mlflow.* attributes.
    inputs = _as_obj(_get(raw, "inputs"))
    if inputs is None:
        inputs = attributes.get(_ATTR_INPUTS)
    outputs = _as_obj(_get(raw, "outputs"))
    if outputs is None:
        outputs = attributes.get(_ATTR_OUTPUTS)

    tokens = _get(raw, "tokens") or attributes.get(_ATTR_TOKENS)

    return NSpan(
        span_id=str(_get(raw, "span_id", default="")),
        parent_id=_get(raw, "parent_id", "parent_span_id"),
        name=str(_get(raw, "name", default="<unnamed>")),
        start_ns=_get(raw, "start_time_ns", "start_ns", "start_time_unix_nano"),
        end_ns=_get(raw, "end_time_ns", "end_ns", "end_time_unix_nano"),
        status=_short_status(status_code),
        status_message=str(status_message or ""),
        inputs=inputs,
        outputs=outputs,
        attributes=attributes,
        events=events,
        span_type=str(_get(raw, "span_type", default="") or attributes.get(_ATTR_SPAN_TYPE) or ""),
        model=str(
            _get(raw, "model_name", "model", default="") or attributes.get(_ATTR_MODEL) or ""
        ),
        tokens=tokens if isinstance(tokens, dict) else {},
        cost_usd=_first_float(
            _get(raw, "llm_cost", "cost_usd"), attributes.get(_ATTR_COST)
        ),
    )


def _extract_spans(trace: Any) -> List[Any]:
    # Live object: trace.data.spans ; flat dict: trace["spans"] ;
    # export dict: trace["data"]["spans"].
    data = _get(trace, "data")
    spans = _get(data, "spans") if data is not None else None
    if spans is None:
        spans = _get(trace, "spans")
    return list(spans or [])


def _epoch_ms(value: Any) -> Optional[int]:
    """Coerce a timestamp to epoch milliseconds.

    Live entities use int ms; MLflow's JSON export writes ISO-8601 instead
    (``"2026-07-25T01:59:15.245Z"``).
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
        try:  # 3.10's fromisoformat can't read a trailing "Z"
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    return None


def normalize_trace(trace: Any) -> NTrace:
    """Convert a live MLflow 3.x Trace, an MLflow JSON export, or our flat dict.

    A live Trace / export dict keeps its metadata under ``info``; our flat dict
    carries the same fields at the top level.
    """
    info = _get(trace, "info")
    src = info if info is not None else trace

    duration = _get(src, "execution_duration", "duration_ms", "execution_duration_ms")
    spans = [normalize_span(s) for s in _extract_spans(trace)]
    metadata = _to_dict(_get(src, "trace_metadata", "metadata", default={}))

    # Aggregate token usage / cost live in trace metadata as JSON strings; the
    # live entity also exposes them as .token_usage / .cost.
    tokens = (
        _get(src, "token_usage") or _as_obj(metadata.get(_META_TOKENS)) or _get(trace, "tokens")
    )
    cost = _first_float(_get(src, "cost", "cost_usd"), _as_obj(metadata.get(_META_COST)))

    return NTrace(
        trace_id=str(_get(src, "trace_id", default="")),
        state=_short_status(_get(src, "state")),
        duration_ms=_first_float(duration),
        request_time_ms=_epoch_ms(_get(src, "request_time", "request_time_ms")),
        request_preview=str(_get(src, "request_preview", default="") or ""),
        response_preview=str(_get(src, "response_preview", default="") or ""),
        tags=_to_dict(_get(src, "tags", default={})),
        metadata=metadata,
        spans=spans,
        tokens=tokens if isinstance(tokens, dict) else {},
        cost_usd=cost,
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


def is_error_trace(trace: NTrace) -> bool:
    """Did this trace fail? (state, or any span carrying an error/exception)"""
    return trace.state == "ERROR" or bool(error_spans(trace))


def span_depths(spans: List[NSpan]) -> Dict[str, int]:
    """Depth of each span from its root, following parent links."""
    by_id = {s.span_id: s for s in spans}
    depths: Dict[str, int] = {}

    def depth_of(span: NSpan) -> int:
        cached = depths.get(span.span_id)
        if cached is not None:
            return cached
        depths[span.span_id] = 0  # placeholder: also breaks parent-link cycles
        parent = by_id.get(span.parent_id) if span.parent_id else None
        value = depth_of(parent) + 1 if parent is not None else 0
        depths[span.span_id] = value
        return value

    for s in spans:
        depth_of(s)
    return depths


def originating_exceptions(trace: NTrace) -> List[Tuple[NSpan, Dict[str, str]]]:
    """Exceptions paired with the deepest span that recorded them.

    An exception propagating out of a nested span is re-recorded on every
    ancestor, so a single failure otherwise shows up once per level. Keeping
    only the deepest occurrence of each (type, message) points at the span the
    failure actually came from.
    """
    depths = span_depths(trace.spans)
    deepest: Dict[Tuple[str, str], Tuple[int, NSpan, Dict[str, str]]] = {}
    for span in trace.spans:
        for exc in span_exceptions(span):
            key = (exc["type"], exc["message"])
            depth = depths.get(span.span_id, 0)
            current = deepest.get(key)
            if current is None or depth > current[0]:
                deepest[key] = (depth, span, exc)
    return [(span, exc) for _, span, exc in deepest.values()]


def slowest_spans(
    trace: NTrace,
    top_n: int = 5,
    children: Optional[Dict[str, List[NSpan]]] = None,
) -> List[Tuple[NSpan, Optional[float]]]:
    """Spans ranked by self time. Pass ``children`` to reuse an existing forest."""
    if children is None:
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
        origins = originating_exceptions(tr)
        if origins:
            for span, exc in origins:
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
        "span_type": span.span_type,
        "model": span.model,
        "tokens": span.tokens,
        "cost_usd": span.cost_usd,
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
        "tokens": trace.tokens,
        "cost_usd": trace.cost_usd,
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


def _fmt_tokens(tokens: Dict[str, Any]) -> str:
    """``{"input_tokens": 10, "output_tokens": 5, ...}`` -> ``in 10 / out 5 / total 15``."""
    parts = []
    for key, label in (
        ("input_tokens", "in"),
        ("output_tokens", "out"),
        ("total_tokens", "total"),
        ("cache_read_input_tokens", "cache-read"),
        ("cache_creation_input_tokens", "cache-write"),
    ):
        val = tokens.get(key)
        if val:
            parts.append(f"{label} {val}")
    return " / ".join(parts)


def _span_llm_note(span: NSpan) -> str:
    """Compact ``[LLM gpt-4o  in 10 / out 5  $0.0012]`` suffix, empty when absent."""
    bits = [b for b in (span.span_type or "", span.model) if b and b != "UNKNOWN"]
    if span.tokens:
        bits.append(_fmt_tokens(span.tokens))
    if span.cost_usd is not None:
        bits.append(f"${span.cost_usd:.4f}")
    return f"  [{'  '.join(bits)}]" if bits else ""


def render_span_tree(trace: NTrace, errors_only: bool = False, max_io: int = 500) -> str:
    roots, children = build_span_forest(trace.spans)
    if trace.duration_ms:
        total = trace.duration_ms
    else:  # no trace-level duration: scale bars against the widest span
        total = max((s.total_ms or 0.0) for s in trace.spans) if trace.spans else 0.0
    lines: List[str] = []
    seen: set = set()

    def visit(span: NSpan, depth: int) -> None:
        if span.span_id in seen:  # malformed parent links must not loop forever
            return
        seen.add(span.span_id)
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
                f"{_span_llm_note(span)}"
            )
            # MLflow sets status_message to "Type: message", which the
            # exception lines below already say — only print what adds detail.
            exc_lines = [f"{e['type']}: {e['message']}" for e in exc]
            msg = span.status_message
            if span.status == "ERROR" and msg and not any(msg in line for line in exc_lines):
                lines.append(f"{indent}    ! {msg}")
            for line in exc_lines:
                lines.append(f"{indent}    ! {line}")
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
    if trace.tokens:
        out.append(f"  tokens   : {_fmt_tokens(trace.tokens)}")
    if trace.cost_usd is not None:
        out.append(f"  cost     : ${trace.cost_usd:.6f}")
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
        header = f"Trace {trace.trace_id}  total {_fmt_ms(trace.duration_ms)}  [{trace.state}]"
        if trace.cost_usd is not None:
            header += f"  ${trace.cost_usd:.6f}"
        out.append(header)
        _, children = build_span_forest(trace.spans)
        for span, self_ms in slowest_spans(trace, top_n, children):
            pct = ""
            if self_ms is not None and trace.duration_ms:
                pct = f"{(self_ms / trace.duration_ms) * 100:5.1f}%"
            # _fmt_ms carries the None case; formatting None directly raises.
            out.append(f"  {_fmt_ms(self_ms):>9}  {pct:>6}  {span.name}")
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


def _render_failing_span(span: NSpan, exc: Optional[Dict[str, str]]) -> List[str]:
    out = [f"  • {span.name}  [{span.status}]  {_fmt_ms(span.total_ms)}"]
    if span.status_message:
        out.append(f"      status: {span.status_message}")
    if exc:
        out.append(f"      exception.type   : {exc['type']}")
        out.append(f"      exception.message: {exc['message']}")
        if exc["stacktrace"]:
            tail = exc["stacktrace"].strip().splitlines()[-12:]
            out.append("      exception.stacktrace (tail):")
            out.extend("        " + ln for ln in tail)
    if span.inputs is not None:
        out.append(f"      inputs : {_stringify(span.inputs, 800)}")
    if span.outputs is not None:
        out.append(f"      outputs: {_stringify(span.outputs, 400)}")
    out.append("")
    return out


def _render_failing_spans(trace: NTrace) -> List[str]:
    """Full detail for the spans a failure originated in.

    Ancestors that merely re-recorded the same propagating exception are
    collapsed into a one-line trail instead of repeating the whole stacktrace.
    """
    errs = error_spans(trace)
    if not errs:
        return ["No error spans. This trace did not fail; review latency below.", ""]

    origins = {span.span_id: exc for span, exc in originating_exceptions(trace)}
    out = [f"FAILING SPANS ({len(errs)}):"]
    for span in errs:
        if origins and span.span_id not in origins:
            continue
        out.extend(_render_failing_span(span, origins.get(span.span_id)))
    propagated = [s.name for s in errs if origins and s.span_id not in origins]
    if propagated:
        out.append(f"  propagated through: {', '.join(propagated)}")
        out.append("")
    return out


def render_diagnose(trace: NTrace) -> str:
    """Structured, compact report meant to be read by the agent to root-cause."""
    head = f"DIAGNOSE {trace.trace_id}  [{trace.state}]  total {_fmt_ms(trace.duration_ms)}"
    if trace.tokens:
        head += f"  ({_fmt_tokens(trace.tokens)})"
    if trace.cost_usd is not None:
        head += f"  ${trace.cost_usd:.6f}"
    out = [head, ""]

    out.extend(_render_failing_spans(trace))

    out.append("LATENCY HOTSPOTS (self time):")
    _, children = build_span_forest(trace.spans)
    for span, self_ms in slowest_spans(trace, 5, children):
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
# names we accept MLFLOW_TRACKING_ARN. If it's a Secrets Manager ARN
# (arn:aws:secretsmanager:...), resolve_mlflow_arn_secret() below fetches the
# secret (JSON with MLFLOW_TRACKING_URI/_USERNAME/_PASSWORD keys, matching
# this repo's backend/app/llm/load_llm.py) via boto3's default credential
# chain and populates os.environ. Otherwise it's used verbatim as a fallback
# tracking URI (kept for backward compatibility / non-AWS setups).
# --------------------------------------------------------------------------

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


# Secret JSON keys this repo's backend expects (backend/app/llm/load_llm.py
# `_load_mlflow_credentials_from_secret`) — the env-var names themselves are
# used as the JSON keys inside the secret.
_SECRET_ENV_KEYS = ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")


def _is_secretsmanager_arn(value: str) -> bool:
    return value.startswith("arn:aws:secretsmanager:")


def _fetch_secretsmanager_json(arn: str) -> Dict[str, Any]:
    """Fetch and JSON-parse a Secrets Manager secret's SecretString.

    Uses boto3's default credential chain (env vars, profile, SSO, instance
    role, ...) — whatever AWS credentials are already available in the
    environment. Region is inferred from the ARN.
    """
    try:
        import boto3
    except ImportError:
        sys.exit(
            "error: MLFLOW_TRACKING_ARN is a Secrets Manager ARN but boto3 is "
            "not available. Run via `uv run` (boto3 is a declared dependency) "
            "or `pip install boto3`."
        )
    try:
        client = boto3.Session().client("secretsmanager")
        resp = client.get_secret_value(SecretId=arn)
    except Exception as exc:  # noqa: BLE001 - surface a single clean error line
        sys.exit(
            f"error: failed to read secret from Secrets Manager ({arn}): "
            f"{type(exc).__name__}: {exc}\n"
            "       (check that AWS credentials are configured in the "
            "environment, e.g. via env vars, `aws sso login`, or --profile)"
        )
    try:
        return json.loads(resp["SecretString"])
    except (KeyError, json.JSONDecodeError) as exc:
        sys.exit(f"error: secret {arn!r} did not contain a JSON SecretString: {exc}")


def resolve_mlflow_arn_secret() -> None:
    """If MLFLOW_TRACKING_ARN is a Secrets Manager ARN, resolve it into
    MLFLOW_TRACKING_URI/_USERNAME/_PASSWORD in ``os.environ``.

    Mirrors backend/app/llm/load_llm.py's `_load_mlflow_credentials_from_secret`:
    the secret's JSON body is expected to contain the env-var names themselves
    as keys. No-op if MLFLOW_TRACKING_URI is already set (real env wins), or if
    MLFLOW_TRACKING_ARN isn't set / isn't a Secrets Manager ARN (e.g. it's
    already a bare tracking URI — kept for backward compatibility).
    """
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return
    arn = os.environ.get("MLFLOW_TRACKING_ARN")
    if not arn or not _is_secretsmanager_arn(arn):
        return
    secret = _fetch_secretsmanager_json(arn)
    missing = [k for k in _SECRET_ENV_KEYS if not secret.get(k)]
    if missing:
        sys.exit(
            f"error: secret {arn!r} is missing required key(s): {', '.join(missing)} "
            f"(expected {', '.join(_SECRET_ENV_KEYS)})"
        )
    for key in _SECRET_ENV_KEYS:
        os.environ[key] = secret[key]
    print(f"# mlflow-traces: resolved MLFLOW_TRACKING_ARN via Secrets Manager ({arn})", file=sys.stderr)


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
    if tracking_uri:  # an explicit flag must beat whatever the env already says
        os.environ["MLFLOW_TRACKING_URI"] = uri
    else:
        os.environ.setdefault("MLFLOW_TRACKING_URI", uri)  # keep fluent API in sync
    return MlflowClient(tracking_uri=uri)


def fetch_trace(client, trace_id: str) -> NTrace:
    trace = client.get_trace(trace_id)
    return normalize_trace(trace)


def fetch_traces(client, trace_ids: List[str], max_workers: int = 8) -> List[NTrace]:
    """Fetch several traces concurrently, preserving the requested order.

    Each get_trace is an independent HTTP round trip, so a serial loop makes
    `get`/`profile` over N ids take N times as long for no reason.
    """
    if len(trace_ids) <= 1:
        return [fetch_trace(client, tid) for tid in trace_ids]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(trace_ids))) as pool:
        return list(pool.map(lambda tid: fetch_trace(client, tid), trace_ids))


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


def _search_page(client, experiment_ids, filter_string, limit, order_by, page_token):
    kwargs: Dict[str, Any] = {
        "experiment_ids": experiment_ids,
        "filter_string": filter_string,
        "max_results": limit,
    }
    if order_by is not None:
        kwargs["order_by"] = order_by
    if page_token:
        kwargs["page_token"] = page_token
    return client.search_traces(**kwargs)


def search_error_traces(
    client,
    experiment_ids: List[str],
    extra_filter: Optional[str],
    max_results: int,
    since_ms: Optional[int] = None,
) -> List[NTrace]:
    """Search ERROR traces, following pagination up to ``max_results``.

    The time window is pushed into the server-side filter when possible, so
    ``--max-results`` bounds traces *inside* the window rather than being spent
    on newer ones that the caller then discards.
    """
    base = ["trace.status = 'ERROR'"]
    if extra_filter:
        base.append(extra_filter)
    windowed = base + [f"trace.timestamp > {since_ms}"] if since_ms is not None else None

    # Both the timestamp predicate and order_by are best-effort: backends differ,
    # so fall back through progressively plainer queries. `experiment_ids` works
    # on 3.13 and 3.14 (deprecated for `locations` in 3.14) — silence that
    # FutureWarning to keep output clean.
    candidates: List[Tuple[str, Optional[List[str]]]] = []
    for clauses in ([windowed] if windowed else []) + [base]:
        for order_by in (["timestamp_ms DESC"], None):
            candidates.append((" AND ".join(clauses), order_by))

    last: Optional[Exception] = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        for filter_string, order_by in candidates:
            try:
                page = _search_page(
                    client, experiment_ids, filter_string, max_results, order_by, None
                )
            except Exception as exc:  # try the next, plainer, candidate
                last = exc
                continue

            # This query shape works; drain further pages (servers cap page size).
            collected = list(page)
            token = getattr(page, "token", None)
            while token and len(collected) < max_results:
                page = _search_page(
                    client,
                    experiment_ids,
                    filter_string,
                    max_results - len(collected),
                    order_by,
                    token,
                )
                if not page:
                    break
                collected.extend(page)
                token = getattr(page, "token", None)
            return [normalize_trace(t) for t in collected[:max_results]]
    raise last  # type: ignore[misc]


def load_traces_from_file(path: str) -> List[NTrace]:
    """Read one or many traces from a JSON dump.

    Accepts a single trace object, a bare list (what ``get --json`` emits for
    several ids), or ``{"traces": [...]}``.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and isinstance(data.get("traces"), list):
        data = data["traces"]
    items = data if isinstance(data, list) else [data]
    if not items:
        sys.exit(f"error: no traces found in {path}")
    return [normalize_trace(item) for item in items]


def load_trace_from_file(path: str) -> NTrace:
    return load_traces_from_file(path)[0]


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
    ids = [tid for tid in (trace_ids or []) if tid]
    if args.from_file:
        traces = load_traces_from_file(args.from_file)
        # Ids are optional offline; when given, use them to pick out of a
        # multi-trace dump (ignored if none of them match, so a placeholder id
        # still works).
        matched = [t for t in traces if t.trace_id in set(ids)] if ids else []
        return matched or traces
    if not ids:
        sys.exit("error: no trace id given (pass one or more ids, or use --from-file).")
    return fetch_traces(get_client(args.tracking_uri), ids)


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
    traces = _load_traces(args, [args.trace_id] if args.trace_id else [])
    trace = traces[0]
    if args.json:
        payload = trace_to_dict(trace)
        payload["error_groups"] = group_errors([trace])
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(render_diagnose(trace))
    return 0


def cmd_errors(args) -> int:
    cutoff = _cutoff_ms(args.since)
    if args.from_file:
        # A dump can hold healthy traces too; the live path filters server-side.
        traces = [t for t in load_traces_from_file(args.from_file) if is_error_trace(t)]
    else:
        client = get_client(args.tracking_uri)
        exp_ids = resolve_experiment_ids(client, args.experiment_id, args.experiment_name)
        traces = search_error_traces(client, exp_ids, args.filter, args.max_results, cutoff)

    if cutoff is not None:  # also enforce locally: the pushdown is best-effort
        traces = [t for t in traces if (t.request_time_ms or 0) >= cutoff]

    if len(traces) >= args.max_results:
        print(
            f"# mlflow-traces: hit --max-results ({args.max_results}); "
            "counts below are a lower bound.",
            file=sys.stderr,
        )

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
    p_get.add_argument("trace_ids", nargs="*", help="Optional when --from-file is used.")
    p_get.add_argument("--errors-only", action="store_true", help="Show only error spans.")
    p_get.add_argument("--max-io", type=int, default=500, help="Truncate inputs/outputs (chars).")
    p_get.set_defaults(func=cmd_get)

    p_prof = sub.add_parser(
        "profile", parents=[common], help="Latency breakdown; aggregate across traces."
    )
    p_prof.add_argument("trace_ids", nargs="*", help="Optional when --from-file is used.")
    p_prof.add_argument("--top", type=int, default=8, help="Show N slowest spans per trace.")
    p_prof.set_defaults(func=cmd_profile)

    p_diag = sub.add_parser(
        "diagnose", parents=[common], help="Structured root-cause report for one trace."
    )
    p_diag.add_argument("trace_id", nargs="?", help="Optional when --from-file is used.")
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


def _force_utf8_streams() -> None:
    """Ensure stdout/stderr can encode our Unicode symbols (✓/✗/█) even when
    the platform's default console encoding can't (e.g. Windows cp1252/437)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except Exception:
                pass


def main(argv: Optional[List[str]] = None) -> int:
    _force_utf8_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    # A live backend needs config; auto-load a .env unless we're offline or opted out.
    if not args.from_file and not args.no_env_file:
        load_env_file(args.env_file)
    if not args.from_file:
        resolve_mlflow_arn_secret()
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

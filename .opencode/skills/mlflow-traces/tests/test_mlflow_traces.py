"""Offline tests for the pure helpers in mlflow_traces.py.

No MLflow backend and no `mlflow` install are required: everything runs off
the bundled fixture and a hand-built object that mimics a live MLflow Trace
(using v3 field names) to prove the version-defensive normalizers work.
"""
import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(__file__)
SKILL_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SKILL_ROOT, "scripts"))

import mlflow_traces as mt  # noqa: E402

FIXTURE = os.path.join(SKILL_ROOT, "fixtures", "sample_trace.json")
# A real `Trace.to_json()` payload captured from mlflow 3.14 (stacktrace and
# metadata trimmed): OTel field names, JSON-encoded attribute values, ISO
# request_time. This is what a user's saved trace dump actually looks like.
EXPORT_FIXTURE = os.path.join(SKILL_ROOT, "fixtures", "mlflow_export_trace.json")


@pytest.fixture
def trace():
    return mt.load_trace_from_file(FIXTURE)


@pytest.fixture
def export_trace():
    return mt.load_trace_from_file(EXPORT_FIXTURE)


def test_fixture_normalizes(trace):
    assert trace.trace_id == "tr-abc123def456"
    assert trace.state == "ERROR"
    assert trace.duration_ms == 1250.0
    assert len(trace.spans) == 4
    assert trace.tags["environment"] == "prod"


def test_span_forest(trace):
    roots, children = mt.build_span_forest(trace.spans)
    assert [r.name for r in roots] == ["predict"]
    assert len(children["s-root"]) == 3
    # children sorted by start time
    assert [c.name for c in children["s-root"]] == ["retrieve_docs", "rerank", "llm_generate"]


def test_self_time(trace):
    _, children = mt.build_span_forest(trace.spans)
    by_name = {s.name: s for s in trace.spans}
    # root total 1250ms minus children (50 + 840 + 340) = 20ms self
    assert mt.span_self_ms(by_name["predict"], children) == pytest.approx(20.0)
    # leaf self == total
    assert mt.span_self_ms(by_name["rerank"], children) == pytest.approx(840.0)


def test_slowest_spans_by_self_time(trace):
    top = mt.slowest_spans(trace, top_n=1)
    assert top[0][0].name == "rerank"
    assert top[0][1] == pytest.approx(840.0)


def test_span_exceptions(trace):
    by_name = {s.name: s for s in trace.spans}
    excs = mt.span_exceptions(by_name["llm_generate"])
    assert len(excs) == 1
    assert excs[0]["type"] == "RateLimitError"
    assert "Rate limit exceeded" in excs[0]["message"]
    assert "Traceback" in excs[0]["stacktrace"]
    # a healthy span has none
    assert mt.span_exceptions(by_name["retrieve_docs"]) == []


def test_error_spans(trace):
    names = {s.name for s in mt.error_spans(trace)}
    assert names == {"predict", "llm_generate"}


def test_group_errors(trace):
    groups = mt.group_errors([trace])
    types_ = {g["exception_type"] for g in groups}
    assert "RateLimitError" in types_
    rl = next(g for g in groups if g["exception_type"] == "RateLimitError")
    assert rl["span"] == "llm_generate"
    assert rl["trace_ids"] == ["tr-abc123def456"]
    assert rl["count"] == 1


def test_error_grouping_collapses_similar_messages(trace):
    # Two traces whose only difference is a number in the message must land in
    # the same group.
    import copy

    def with_msg(tid, msg):
        t = copy.deepcopy(trace)
        t.trace_id = tid
        for s in t.spans:
            for e in s.events:
                if e.name == "exception":
                    e.attributes["exception.message"] = msg
        return t

    a = with_msg("t-1", "Rate limit exceeded (requested 31200)")
    b = with_msg("t-2", "Rate limit exceeded (requested 48000)")
    groups = mt.group_errors([a, b])
    rl = next(g for g in groups if g["exception_type"] == "RateLimitError")
    assert rl["count"] == 2
    assert set(rl["trace_ids"]) == {"t-1", "t-2"}


def test_json_roundtrips_through_normalize(trace):
    payload = mt.trace_to_dict(trace)
    reloaded = mt.normalize_trace(json.loads(json.dumps(payload)))
    assert reloaded.trace_id == trace.trace_id
    assert reloaded.state == trace.state
    assert len(reloaded.spans) == len(trace.spans)
    _, ch1 = mt.build_span_forest(trace.spans)
    _, ch2 = mt.build_span_forest(reloaded.spans)
    by1 = {s.name: mt.span_self_ms(s, ch1) for s in trace.spans}
    by2 = {s.name: mt.span_self_ms(s, ch2) for s in reloaded.spans}
    assert by1 == by2


def test_normalize_live_object_v3_field_names():
    """Mimic a live MLflow 3.x Trace using the real attribute names (verified
    against mlflow 3.14): TraceInfo.trace_id/state/execution_duration/
    request_time/trace_metadata, Span.start_time_ns/end_time_ns, a SpanStatus
    with a str-enum status_code and description, and an `exception` event."""
    import enum

    class SpanStatusCode(str, enum.Enum):
        ERROR = "ERROR"

    exc_event = types.SimpleNamespace(
        name="exception",
        timestamp=5,
        attributes={
            "exception.type": "ValueError",
            "exception.message": "bad input",
            "exception.stacktrace": "Traceback ...",
        },
    )
    span = types.SimpleNamespace(
        span_id="sp1",
        parent_id=None,
        name="root",
        start_time_ns=0,
        end_time_ns=2_000_000,  # 2 ms
        status=types.SimpleNamespace(status_code=SpanStatusCode.ERROR, description="boom"),
        inputs={"x": 1},
        outputs=None,
        attributes={"k": "v"},
        events=[exc_event],
    )
    info = types.SimpleNamespace(
        trace_id="tr-9",
        state="ERROR",
        execution_duration=2,  # ms int, as MLflow returns it
        request_time=1721822400000,
        request_preview="{}",
        response_preview=None,  # None on error, as observed
        tags={"a": "b"},
        trace_metadata={"run": "1"},
    )
    live = types.SimpleNamespace(info=info, data=types.SimpleNamespace(spans=[span]))

    t = mt.normalize_trace(live)
    assert t.trace_id == "tr-9"
    assert t.state == "ERROR"
    assert t.duration_ms == 2.0
    assert t.response_preview == ""
    assert t.metadata == {"run": "1"}
    assert len(t.spans) == 1
    s = t.spans[0]
    assert s.status == "ERROR"
    assert s.status_message == "boom"
    assert s.total_ms == pytest.approx(2.0)
    assert mt.span_exceptions(s)[0]["type"] == "ValueError"


def test_integration_real_mlflow_trace(tmp_path, monkeypatch):
    """End-to-end against the real library: create a failing trace, fetch it,
    and normalize it. Skips unless a full MLflow with a SQL store is available
    (mlflow-skinny alone cannot persist traces locally)."""
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    from mlflow import MlflowClient

    @mlflow.trace
    def boom(x):
        raise ValueError(f"bad {x}")

    try:
        with pytest.raises(ValueError):
            boom(3)
        if hasattr(mlflow, "flush_trace_async_logging"):
            mlflow.flush_trace_async_logging()
        client = MlflowClient()
        traces = client.search_traces(experiment_ids=["0"], max_results=5)
    except Exception as exc:  # backend can't store traces (e.g. skinny/no sqlalchemy)
        pytest.skip(f"no trace backend available: {exc}")

    if not traces:
        pytest.skip("no trace persisted")

    nt = mt.normalize_trace(traces[0])
    assert nt.trace_id.startswith("tr-")
    assert nt.state == "ERROR"
    assert nt.duration_ms is not None
    assert nt.spans
    exc_types = {e["type"] for s in nt.spans for e in mt.span_exceptions(s)}
    assert "ValueError" in exc_types


# ---- MLflow's own JSON export shape ---------------------------------------


def test_normalizes_mlflow_json_export(export_trace):
    """MLflow's export uses OTel names (parent_span_id / *_unix_nano / status
    dict) and an ISO request_time — none of which are our --json names."""
    t = export_trace
    assert t.state == "ERROR"
    assert t.duration_ms == 107.0  # from execution_duration_ms
    assert t.request_time_ms == 1784944838226  # parsed from ISO-8601
    assert len(t.spans) == 3


def test_export_span_tree_is_nested(export_trace):
    """parent_span_id must be honoured, else every span looks like a root."""
    roots, children = mt.build_span_forest(export_trace.spans)
    assert [r.name for r in roots] == ["root"]
    assert {c.name for c in children[roots[0].span_id]} == {"retrieve", "llm"}


def test_export_spans_have_timings_and_status(export_trace):
    by_name = {s.name: s for s in export_trace.spans}
    assert by_name["llm"].status == "ERROR"  # not "STATUS_CODE_ERROR"
    assert by_name["retrieve"].status == "OK"
    assert by_name["llm"].total_ms == pytest.approx(53.0, abs=1.0)
    _, children = mt.build_span_forest(export_trace.spans)
    assert mt.span_self_ms(by_name["root"], children) is not None


def test_export_span_io_comes_from_attributes(export_trace):
    """The export has no top-level inputs/outputs; they live in mlflow.span*."""
    by_name = {s.name: s for s in export_trace.spans}
    assert by_name["llm"].inputs == {"p": "x"}
    assert by_name["llm"].span_type == "LLM"  # attribute values are JSON-encoded
    assert by_name["llm"].model == "gpt-4o-mini"


def test_token_usage_and_cost_are_surfaced(export_trace):
    assert export_trace.tokens["total_tokens"] == 150
    assert export_trace.cost_usd == pytest.approx(0.003)
    llm = next(s for s in export_trace.spans if s.name == "llm")
    assert llm.tokens["input_tokens"] == 120
    assert llm.cost_usd == pytest.approx(0.003)  # from {"total_cost": ...}
    assert "total 150" in mt.render_trace_detail(export_trace)


def test_export_round_trips_through_our_json(export_trace):
    reloaded = mt.normalize_trace(json.loads(json.dumps(mt.trace_to_dict(export_trace))))
    assert reloaded.duration_ms == export_trace.duration_ms
    assert reloaded.tokens == export_trace.tokens
    assert reloaded.cost_usd == export_trace.cost_usd
    assert [s.parent_id for s in reloaded.spans] == [s.parent_id for s in export_trace.spans]
    assert [s.span_type for s in reloaded.spans] == [s.span_type for s in export_trace.spans]


# ---- error attribution & rendering edge cases -----------------------------


def test_propagated_exception_attributed_to_deepest_span(export_trace):
    """MLflow re-records an exception on every ancestor; the group must name
    the span it came from, once — not one group per level."""
    groups = mt.group_errors([export_trace])
    assert len(groups) == 1
    assert groups[0]["span"] == "llm"
    assert groups[0]["count"] == 1


def test_render_profile_survives_spans_without_timings():
    """A span missing start/end has self_ms None; formatting it directly raises."""
    span = mt.NSpan(
        span_id="s1", parent_id=None, name="untimed", start_ns=None, end_ns=None,
        status="OK", status_message="", inputs=None, outputs=None, attributes={},
    )
    t = mt.NTrace(
        trace_id="tr-x", state="OK", duration_ms=None, request_time_ms=None,
        request_preview="", response_preview="", tags={}, metadata={}, spans=[span],
    )
    assert "untimed" in mt.render_profile([t])
    assert "untimed" in mt.render_span_tree(t)


def test_span_tree_tolerates_a_parent_cycle():
    a = mt.NSpan("a", "b", "a", 0, 1, "OK", "", None, None, {})
    b = mt.NSpan("b", "a", "b", 0, 1, "OK", "", None, None, {})
    t = mt.NTrace("tr-c", "OK", 1.0, None, "", "", {}, {}, [a, b])
    # Both must terminate and stay finite; which span "wins" the cycle is
    # arbitrary, and a cyclic trace has no meaningful depth anyway.
    assert mt.render_span_tree(t)
    depths = mt.span_depths([a, b])
    assert set(depths) == {"a", "b"} and all(d < 10 for d in depths.values())


# ---- multi-trace file loading ---------------------------------------------


def test_load_traces_from_file_reads_every_trace(tmp_path, trace):
    payload = [mt.trace_to_dict(trace), mt.trace_to_dict(trace)]
    payload[1]["trace_id"] = "tr-second"
    p = tmp_path / "dump.json"
    p.write_text(json.dumps(payload))
    loaded = mt.load_traces_from_file(str(p))
    assert [t.trace_id for t in loaded] == ["tr-abc123def456", "tr-second"]


def test_load_traces_from_file_accepts_traces_envelope(tmp_path, trace):
    p = tmp_path / "dump.json"
    p.write_text(json.dumps({"traces": [mt.trace_to_dict(trace)]}))
    assert len(mt.load_traces_from_file(str(p))) == 1


def test_is_error_trace_separates_healthy_traces(trace):
    assert mt.is_error_trace(trace)
    healthy = mt.NTrace("tr-ok", "OK", 1.0, None, "", "", {}, {}, [])
    assert not mt.is_error_trace(healthy)


def test_cutoff_ms_relative_and_iso():
    assert mt._cutoff_ms(None) is None
    now_ms = mt._cutoff_ms("0s")
    assert isinstance(now_ms, int)
    iso = mt._cutoff_ms("2024-01-01T00:00:00")
    assert iso == 1704067200000


# ---- env / .env loading & config resolution -------------------------------


def test_parse_env_file(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "# a comment\n"
        "\n"
        "export MLFLOW_TRACKING_URI=https://mlflow.example.com\n"
        'MLFLOW_EXPERIMENT_NAME="my app"\n'
        "MLFLOW_TRACKING_ARN='arn:aws:sagemaker:us-east-1:1:mlflow-tracking-server/x'\n"
        "NOT_AN_ASSIGNMENT\n"
    )
    env = mt.parse_env_file(str(p))
    assert env["MLFLOW_TRACKING_URI"] == "https://mlflow.example.com"
    assert env["MLFLOW_EXPERIMENT_NAME"] == "my app"
    assert env["MLFLOW_TRACKING_ARN"].startswith("arn:aws:sagemaker:")
    assert "NOT_AN_ASSIGNMENT" not in env


def test_find_env_file_prefers_backend_then_walks_up(tmp_path):
    proj = tmp_path / "proj"
    (proj / "backend").mkdir(parents=True)
    (proj / "backend" / ".env").write_text("A=1\n")
    (proj / ".env").write_text("B=2\n")
    deep = proj / "service" / "sub"
    deep.mkdir(parents=True)
    found = mt.find_env_file(str(deep))
    assert found == str(proj / "backend" / ".env")  # backend/.env wins, discovered upward


def test_load_env_file_does_not_override_real_env(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("MLFLOW_TRACKING_URI=http://from-file\nMLFLOW_EXPERIMENT_NAME=file-exp\n")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://ambient")  # already set → must win
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    loaded = mt.load_env_file(str(p))
    assert loaded == str(p)
    assert os.environ["MLFLOW_TRACKING_URI"] == "http://ambient"
    assert os.environ["MLFLOW_EXPERIMENT_NAME"] == "file-exp"  # filled in


def test_resolve_tracking_uri_precedence(monkeypatch):
    for k in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_ARN"):
        monkeypatch.delenv(k, raising=False)
    assert mt.resolve_tracking_uri("cli://x") == "cli://x"           # CLI wins
    monkeypatch.setenv("MLFLOW_TRACKING_ARN", "arn:aws:sagemaker:...")
    assert mt.resolve_tracking_uri(None) == "arn:aws:sagemaker:..."  # ARN used as URI
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://uri")
    assert mt.resolve_tracking_uri(None) == "http://uri"             # standard URI preferred


def test_env_experiment_name(monkeypatch):
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    assert mt.env_experiment_name() is None
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "my-exp")
    assert mt.env_experiment_name() == "my-exp"


def test_renderers_do_not_crash(trace):
    assert "Trace tr-abc123def456" in mt.render_trace_detail(trace)
    assert "rerank" in mt.render_profile([trace])
    assert "DIAGNOSE" in mt.render_diagnose(trace)
    assert "RateLimitError" in mt.render_errors(mt.group_errors([trace]))

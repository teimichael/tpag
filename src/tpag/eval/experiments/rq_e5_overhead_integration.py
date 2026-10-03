"""Rq e5 overhead integration for the TPAG replication package."""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from ...adapters import langgraph as lg_adapter
from ...runtime import TPAGRuntime
from ...sim.topology_gen import series_topology
from ..config import ExpConfig
from . import finalize, jsonl_appender, make_recorder

RQ = "RQ-E5"


def _overhead_curve(cfg: ExpConfig, append_row=None) -> list[dict]:
    rows: list[dict] = []
    for n in cfg.tier.overhead_sizes:
        # bound total events (~<=300k) so large-N points stay memory-safe
        episodes = max(100, min(cfg.tier.overhead_episodes, 300_000 // max(1, n + 2)))
        wl = series_topology(n, coupling=0.2)
        traces = wl.generate(episodes, seed=cfg.seed + n)
        n_events = sum(len(t) for t in traces)
        rt = TPAGRuntime(
            wl.graph, wl.structure, wl.contracts, wl.assumption_checks, regime=wl.regime
        )
        tracemalloc.start()
        t0 = time.perf_counter()
        rt.ingest_many(traces)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        row = {
            "n_components": n,
            "episodes": episodes,
            "events": n_events,
            "monitor_per_event_us": elapsed / n_events * 1e6 if n_events else None,
            "monitor_per_episode_ms": elapsed / episodes * 1e3 if episodes else None,
            "ingest_wall_s": elapsed,
            "peak_mib": peak / (2**20),
        }
        rows.append(row)
        if append_row is not None:
            append_row(row)  # stream as computed
    return rows


def _llm_step_reference(cfg: ExpConfig, model_tag: str) -> float | None:
    from ...adapters.llm import OllamaClient
    from ...agents.guards import POLICY_SYSTEM_PROMPT, LLMGuard
    from ...sim.attacks import UNSAFE_REQUESTS

    g = LLMGuard("ref", OllamaClient(model_tag, base_url=cfg.ollama_url), POLICY_SYSTEM_PROMPT)
    times = [
        g.decide(UNSAFE_REQUESTS[i % len(UNSAFE_REQUESTS)], seed=i).latency_s for i in range(3)
    ]
    return float(np.mean(times))


def _langgraph_integration(cfg: ExpConfig, rec) -> dict:
    """End-to-end LangGraph run + live certificate + containment. Fails safe."""
    from ...adapters.llm import OllamaClient, ollama_available
    from ...agents.guards import LLMGuard
    from ...sim.attacks import make_dataset
    from ...sim.domains.code_review import (
        CI_POLICY_PROMPT,
        CODE_REVIEW_STRICT_PROMPT,
        code_review_workflow,
    )
    from .. import model_zoo

    if not ollama_available(cfg.ollama_url):
        return {"status": "skipped_no_ollama"}
    if not lg_adapter.available():
        return {"status": "skipped_no_langgraph", "hint": "pip install -r requirements.txt"}

    # Freeze the model used in the reported integration.
    zoo = model_zoo.probe_zoo(cfg)
    if not zoo.available:
        return {"status": "skipped_missing_models", "pull": model_zoo.pull_hints(zoo)}
    tag = "deepseek-r1:1.5b"
    if tag not in {model.tag for model in zoo.available}:
        return {"status": "skipped_missing_models", "missing": [tag]}

    guards = [
        LLMGuard("security", OllamaClient(tag, base_url=cfg.ollama_url), CODE_REVIEW_STRICT_PROMPT),
        LLMGuard("license", OllamaClient(tag, base_url=cfg.ollama_url), CODE_REVIEW_STRICT_PROMPT),
        LLMGuard("ci_policy", OllamaClient(tag, base_url=cfg.ollama_url), CI_POLICY_PROMPT),
    ]
    try:
        app = lg_adapter.build_review_app(guards, seed=cfg.seed)
    except lg_adapter.MissingDependencyError as exc:
        return {"status": "skipped_no_langgraph", "error": str(exc)}

    wl = code_review_workflow(coupling=0.3)
    rt = TPAGRuntime(wl.graph, wl.structure, wl.contracts, wl.assumption_checks, regime=wl.regime)
    rng = np.random.default_rng(cfg.seed)
    dataset = make_dataset(
        cfg.tier.integration_requests,
        attack_fraction=0.5,
        rng=rng,
        domain="code_review",
        strategy="shared",
        guard_count=3,
    )
    n_live = 0
    (rec.run_dir / "langgraph_episodes.jsonl").unlink(missing_ok=True)
    for inst in dataset:
        trace, info = lg_adapter.run_episode_trace(app, inst.text, actor="merge_actor")
        rt.ingest(trace)
        n_live += 1
        rec.append_jsonl(
            "langgraph_episodes.jsonl",
            {"request": inst.text[:160], "attacked": bool(inst.attacked), **info},
        )
    cert = rt.certificate(eta=cfg.tier.eta_grid[0])

    return {
        "status": "measured",
        "model": tag,
        "live_episodes": n_live,
        "observed_violation_rate": rt.observed_violation_rate,
        "certificate_bound": cert.bound,
        "certificate_confidence": cert.confidence,
        "certificate_sound": bool(cert.bound + 1e-9 >= rt.observed_violation_rate),
    }


def run(cfg: ExpConfig, use_llm: bool = True, *, argv=None, force: bool = False) -> dict:
    rec = make_recorder(cfg, RQ, "rq_e5_overhead_integration", argv=argv, force=force)

    overhead = _overhead_curve(cfg, append_row=jsonl_appender(rec, "overhead.jsonl"))

    llm_step_s = None
    integration = {"status": "skipped_no_llm"}
    if use_llm:
        from ...adapters.llm import ollama_available
        from .. import model_zoo

        if ollama_available(cfg.ollama_url):
            zoo = model_zoo.probe_zoo(cfg)
            if zoo.available:
                tag = "deepseek-r1:1.5b"
                try:
                    llm_step_s = _llm_step_reference(cfg, tag)
                except Exception as exc:  # reference timing is best-effort
                    llm_step_s = None
                    rec.set("llm_step_error", f"{type(exc).__name__}: {exc}")
        # The live integration must never lose the (already-computed) overhead curve.
        try:
            integration = _langgraph_integration(cfg, rec)
        except Exception as exc:
            integration = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    base_us = overhead[0]["monitor_per_event_us"] if overhead else None
    status = "measured" if integration.get("status") == "measured" else "computed"
    metrics = {
        "overhead": overhead,
        "monitor_per_event_us_min": min(
            (r["monitor_per_event_us"] for r in overhead if r["monitor_per_event_us"]), default=None
        ),
        "monitor_per_event_us_max": max(
            (r["monitor_per_event_us"] for r in overhead if r["monitor_per_event_us"]), default=None
        ),
        "llm_step_s": llm_step_s,
        "overhead_ratio_vs_llm": (
            (base_us / 1e6) / llm_step_s if (llm_step_s and base_us) else None
        ),
        "integration": integration,
    }
    rec.write_csv("overhead.csv", overhead)
    return finalize(rec, cfg, RQ, status, metrics, provenance={"overhead": "overhead.csv"})

"""Langgraph for the TPAG replication package."""

from __future__ import annotations

import importlib.util
from typing import Any

from ..events import Trace
from ..sim.components import build_workflow_trace


class MissingDependencyError(RuntimeError):
    """Raised when the optional ``langgraph`` dependency is not installed."""


def available() -> bool:
    """True iff ``langgraph`` can be imported (without importing it)."""
    return importlib.util.find_spec("langgraph") is not None


def build_review_app(guards: list, *, seed: int = 0):
    """Compile a LangGraph ``StateGraph`` that runs ``guards`` in a linear chain.

    Returns ``(app, guard_names)``. Raises :class:`MissingDependencyError` if
    LangGraph is not installed.
    """
    if not available():
        raise MissingDependencyError(
            "langgraph is not installed; run the README installation commands "
            "or `pip install 'tpag[langgraph]'` to enable the RQ-E5 integration."
        )
    from langgraph.graph import END, StateGraph  # lazy import

    names = [g.name for g in guards]

    def _make_node(guard):
        def node(state: dict[str, Any]) -> dict[str, Any]:
            decision = guard.decide(state["request"], seed=seed)
            # Return the FULL state merged with this node's updates. With an untyped
            # ``dict`` state schema langgraph replaces (rather than per-key merges)
            # the state, so each node must carry ``request`` forward explicitly.
            return {
                **state,
                f"decision::{guard.name}": bool(decision.approved),
                f"latency::{guard.name}": float(decision.latency_s),
            }

        return node

    sg = StateGraph(dict)
    sg.add_node("diff_analyzer", lambda s: {**s, "analyzed": True})
    sg.set_entry_point("diff_analyzer")
    prev = "diff_analyzer"
    for g in guards:
        sg.add_node(g.name, _make_node(g))
        sg.add_edge(prev, g.name)
        prev = g.name
    sg.add_node("actor", lambda s: {**s, "committed": True})
    sg.add_edge(prev, "actor")
    sg.add_edge("actor", END)
    return sg.compile(), names


def run_episode_trace(
    app_and_names,
    request_text: str,
    *,
    planner: str = "planner",
    actor: str = "actor",
    action: str = "merge_pr",
) -> tuple[Trace, dict]:
    """Invoke the compiled app on one request and translate the result into a TPAG
    trace (ingestable by :class:`~tpag.runtime.TPAGRuntime`)."""
    app, names = app_and_names
    final = app.invoke({"request": request_text})
    misses = [bool(final.get(f"decision::{n}", False)) for n in names]
    trace = build_workflow_trace(names, misses, planner=planner, actor=actor, action=action)
    info = {
        "misses": dict(zip(names, misses)),
        "latencies": {n: final.get(f"latency::{n}") for n in names},
    }
    return trace, info


__all__ = [
    "MissingDependencyError",
    "available",
    "build_review_app",
    "run_episode_trace",
]

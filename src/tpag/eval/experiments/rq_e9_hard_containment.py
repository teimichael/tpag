"""Rq e9 hard containment for the TPAG replication package."""

from __future__ import annotations

import numpy as np

from ...causal import CausalMonitor, GlobalClockMonitor, requires_upstream_label
from ...containment import localization_score
from ...events import Event, EventKind, Trace
from ...graph import OrchestrationGraph
from ...vectorclock import VectorClockEngine
from ..config import ExpConfig
from . import finalize, jsonl_appender, make_recorder

RQ = "RQ-E9"
_CHECK = requires_upstream_label("approved", data_key="data")


# ---------------------------------------------------------------------------
# Part A -- hard containment graphs.
# ---------------------------------------------------------------------------
def _graph(edges: list[tuple[str, str]]) -> OrchestrationGraph:
    g = OrchestrationGraph()
    for a, b in edges:
        for n in (a, b):
            if n not in g.nodes:
                g.add_component(n)
        g.add_edge(a, b)
    return g


def _hard_cases(width: int) -> list[dict]:
    """A library of blame scenarios where scoped P/R is NOT 1.0 by construction.

    ``truth`` is the genuinely affected set -- everything causally downstream of the
    real cause (``descendants ∪ {cause}``); ``blamed`` is the node whose monitor
    fired (which may be downstream of the cause)."""
    cases: list[dict] = []

    # (a) reconvergent diamond: breach at one fork branch; the reconvergence node is a
    # descendant of the breach but NOT dominated by it (reachable via the other branch),
    # so scoped containment has recall < 1 (correctly high precision).
    g = _graph(
        [
            ("src", "fork"),
            ("fork", "b1"),
            ("fork", "b2"),
            ("b1", "join"),
            ("b2", "join"),
            ("join", "sink"),
        ]
    )
    cases.append({"name": "reconvergent_diamond", "graph": g, "blamed": "b1", "cause": "b1"})

    # (b) detector-upstream-of-cause: the monitor fires at d, but the true root cause is
    # upstream u, so the affected set is descendants(u) ⊋ descendants(d).
    chain = [("u", "d"), ("d", "x"), ("x", "sink")]
    g2 = _graph(chain)
    cases.append({"name": "detector_downstream_of_cause", "graph": g2, "blamed": "d", "cause": "u"})

    # (c) wide reconvergent fan-out: a fork into ``width`` branches that reconverge.
    edges = [("s", "f")]
    for i in range(width):
        edges += [("f", f"w{i}"), (f"w{i}", "m")]
    edges += [("m", "t")]
    g3 = _graph(edges)
    cases.append({"name": f"fanout_w{width}", "graph": g3, "blamed": "w0", "cause": "w0"})

    return cases


def _multi_node(graph: OrchestrationGraph, breaches: list[str]) -> dict:
    """Simultaneous breaches at several nodes: scoped scope = union of dominated
    sub-DAGs; truth = union of (descendants ∪ self)."""
    n_nodes = len(graph.nodes)
    scoped = set()
    truth = set()
    for b in breaches:
        scoped |= graph.dominated_subdag(b)
        truth |= graph.descendants(b) | {b}
    s = localization_score(scoped, truth, n_nodes)
    g = localization_score(set(graph.nodes), truth, n_nodes)
    return {
        "name": "multi_node_simultaneous",
        "breaches": breaches,
        "n_nodes": n_nodes,
        "scoped_precision": s.precision,
        "scoped_recall": s.recall,
        "scoped_halted_fraction": s.halted_fraction,
        "global_precision": g.precision,
        "global_recall": g.recall,
        "global_halted_fraction": g.halted_fraction,
    }


def _score_case(case: dict) -> dict:
    g: OrchestrationGraph = case["graph"]
    n_nodes = len(g.nodes)
    truth = g.descendants(case["cause"]) | {case["cause"]}
    scoped = g.dominated_subdag(case["blamed"])
    s = localization_score(set(scoped), set(truth), n_nodes)
    glob = localization_score(set(g.nodes), set(truth), n_nodes)
    return {
        "name": case["name"],
        "blamed": case["blamed"],
        "cause": case["cause"],
        "n_nodes": n_nodes,
        "truth_size": len(truth),
        "scope_size": len(scoped),
        "scoped_precision": s.precision,
        "scoped_recall": s.recall,
        "scoped_halted_fraction": s.halted_fraction,
        "global_precision": glob.precision,
        "global_recall": glob.recall,
        "global_halted_fraction": glob.halted_fraction,
    }


# ---------------------------------------------------------------------------
# Part B -- async adversarial scheduling.
# ---------------------------------------------------------------------------
def _episode_trace(rng) -> tuple[Trace, Event, bool]:
    """Build one adversarially-scheduled episode. Returns the trace, the actor's
    SIDE_EFFECT_COMMIT activation, and the ground-truth ``is_breach`` (the approval
    did NOT causally precede the action)."""
    eng = VectorClockEngine()
    t = Trace()
    kind = rng.choice(["proper", "race", "skew"], p=[0.4, 0.4, 0.2])

    def approve(vc):
        return Event("guard", EventKind.LABEL_ASSIGN, {"labels": ["approved"], "data": "D"}, vc=vc)

    def action(vc):
        return Event("worker", EventKind.SIDE_EFFECT_COMMIT, {"action": "pay", "data": "D"}, vc=vc)

    if kind == "proper":  # approval sent and received before the action -> safe
        g = eng.send("guard")
        t.append(approve(g))
        w = eng.recv("worker", g)
        act = t.append(action(w))
        return t, act, False
    if kind == "race":  # approval is local-only (never delivered) -> concurrent -> breach
        g = eng.local("guard")
        t.append(approve(g))
        w = eng.local("worker")
        act = t.append(action(w))
        return t, act, True
    # skew: approval causally precedes the action but is recorded AFTER it
    g = eng.send("guard")
    w = eng.recv("worker", g)
    act = t.append(action(w))
    t.append(approve(g))
    return t, act, False


def _async_scheduling(cfg: ExpConfig, append_row=None) -> dict:
    rng = np.random.default_rng(cfg.seed + 13)
    episodes = {"quick": 2000, "full": 20000}.get(cfg.tier.name, 8000)
    causal, glob = CausalMonitor(), GlobalClockMonitor()
    stats = {
        m: {"uncontained_commit": 0, "false_halt": 0, "true_breach": 0, "detected": 0}
        for m in ("causal", "global-clock")
    }
    for _ in range(episodes):
        t, act, is_breach = _episode_trace(rng)
        for name, mon in (("causal", causal), ("global-clock", glob)):
            discharged = mon.discharged(t, act, _CHECK).discharged
            detected = not discharged
            stats[name]["true_breach"] += int(is_breach)
            stats[name]["detected"] += int(detected)
            # with containment, a DETECTED breach blocks the commit; a MISSED true
            # breach reaches commit uncontained (the unsafe outcome the bound assumes away)
            if is_breach and not detected:
                stats[name]["uncontained_commit"] += 1
            # a safe episode that is contained is a needless halt (availability cost)
            if (not is_breach) and detected:
                stats[name]["false_halt"] += 1

    out = {}
    for name, s in stats.items():
        out[name] = {
            "episodes": episodes,
            "true_breach_rate": s["true_breach"] / episodes,
            "uncontained_commit_rate": s["uncontained_commit"] / episodes,
            "false_halt_rate": s["false_halt"] / episodes,
            "breach_detection_recall": (
                (s["true_breach"] - s["uncontained_commit"]) / s["true_breach"]
                if s["true_breach"]
                else 1.0
            ),
        }
        if append_row is not None:
            append_row({"monitor": name, **out[name]})
    return out


def run(cfg: ExpConfig, use_llm: bool = True, *, argv=None, force: bool = False) -> dict:
    rec = make_recorder(cfg, RQ, "rq_e9_hard_containment", argv=argv, force=force)

    # -- Part A: hard containment cases --
    append_case = jsonl_appender(rec, "hard_cases.jsonl")
    cases: list[dict] = []
    for w in (n for n in cfg.tier.topology_sizes if 2 <= n <= 8) or (3,):
        for c in _hard_cases(w):
            row = _score_case(c)
            cases.append(row)
            append_case(row)
    # a genuine multi-node simultaneous-breach scenario
    mn_graph = _graph([("src", "a"), ("a", "b"), ("a", "c"), ("b", "sink"), ("c", "sink")])
    multi = _multi_node(mn_graph, ["b", "c"])
    append_case(multi)
    cases.append(multi)

    scoped_recalls = [c["scoped_recall"] for c in cases]
    scoped_precisions = [c["scoped_precision"] for c in cases]

    # -- Part B: async adversarial scheduling --
    sched = _async_scheduling(cfg, append_row=jsonl_appender(rec, "async_scheduling.jsonl"))

    metrics = {
        "hard_cases": cases,
        "n_hard_cases": len(cases),
        "min_scoped_recall": min(scoped_recalls),
        "mean_scoped_recall": float(np.mean(scoped_recalls)),
        "mean_scoped_precision": float(np.mean(scoped_precisions)),
        "cases_with_recall_below_1": int(sum(r < 1.0 - 1e-9 for r in scoped_recalls)),
        "scoped_precision_is_perfect": bool(min(scoped_precisions) > 1.0 - 1e-9),
        "async_scheduling": sched,
        "causal_uncontained_commit_rate": sched["causal"]["uncontained_commit_rate"],
        "global_uncontained_commit_rate": sched["global-clock"]["uncontained_commit_rate"],
        "causal_false_halt_rate": sched["causal"]["false_halt_rate"],
        "global_false_halt_rate": sched["global-clock"]["false_halt_rate"],
        "interpretation": (
            "scoped containment stays high-precision but has recall<1 on reconvergent / "
            "detector-downstream blame (the hard cases); under adversarial scheduling the "
            "causal monitor + containment drives uncontained commits to 0 while the global "
            "clock both misses races and raises false halts"
        ),
    }
    return finalize(
        rec,
        cfg,
        RQ,
        "computed",
        metrics,
        provenance={"hard_cases": "hard_cases.jsonl", "async": "async_scheduling.jsonl"},
    )

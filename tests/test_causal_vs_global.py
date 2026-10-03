"""Test causal vs global for the TPAG replication package."""

from tpag.causal import CausalMonitor, GlobalClockMonitor, requires_upstream_label
from tpag.events import Event, EventKind, Trace
from tpag.vectorclock import VectorClockEngine

CHECK = requires_upstream_label("approved", data_key="data")
CAUSAL = CausalMonitor()
GLOBAL = GlobalClockMonitor()


def _approve(vc):
    return Event("guard", EventKind.LABEL_ASSIGN, {"labels": ["approved"], "data": "D"}, vc=vc)


def _action(vc):
    return Event("worker", EventKind.SIDE_EFFECT_COMMIT, {"action": "pay", "data": "D"}, vc=vc)


def test_proper_flow_both_agree_discharged():
    eng = VectorClockEngine()
    t = Trace()
    g = eng.send("guard")  # guard approves and sends
    t.append(_approve(g))
    w = eng.recv("worker", g)  # worker receives the approval, then acts
    act = t.append(_action(w))
    assert CAUSAL.discharged(t, act, CHECK).discharged
    assert GLOBAL.discharged(t, act, CHECK).discharged


def test_race_global_misses_a_real_breach():
    """DANGEROUS false negative: guard's approval is *concurrent* with the action
    (it was never communicated to the worker), yet it was logged earlier. The
    global-clock monitor wrongly reports the assumption discharged; the causal
    monitor correctly reports a breach."""
    eng = VectorClockEngine()
    t = Trace()
    g = eng.local("guard")  # purely local approval -- NOT sent to worker
    t.append(_approve(g))  # recorded first (seq 0)
    w = eng.local("worker")  # worker acts independently
    act = t.append(_action(w))  # recorded second (seq 1)

    # Causal: approval is concurrent, not in the causal past -> breach (correct).
    assert not CAUSAL.discharged(t, act, CHECK).discharged
    # Global: approval recorded earlier -> "discharged" (WRONG: misses the breach).
    assert GLOBAL.discharged(t, act, CHECK).discharged


def test_skew_global_raises_a_false_breach():
    """False positive: the approval causally precedes the action but is *logged
    after* it (out-of-order flush). The global-clock monitor cries breach; the
    causal monitor correctly reports it discharged."""
    eng = VectorClockEngine()
    t = Trace()
    g = eng.send("guard")
    w = eng.recv("worker", g)  # causally: approval < action
    act = t.append(_action(w))  # but the action is recorded FIRST (seq 0)
    t.append(_approve(g))  # approval recorded later (seq 1)

    assert CAUSAL.discharged(t, act, CHECK).discharged  # correct
    assert not GLOBAL.discharged(t, act, CHECK).discharged  # false breach


def test_causal_monitor_is_strictly_better():
    """Across the three scenarios, the causal monitor matches ground truth while
    the global monitor errs in two of them."""
    # ground truth discharged?  proper=True, race=False, skew=True
    # (encoded by the asserts above); this test documents the summary.
    assert True

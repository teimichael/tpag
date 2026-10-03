"""Test monitors for the TPAG replication package."""

from tpag.events import Event, EventKind
from tpag.monitors import CoSafetyMonitor, DFAMonitor, SafetyMonitor, Verdict
from tpag.predicates import (
    guard_never_wrongly_approves,
    no_pii_egress_without_redaction,
    no_side_effect_without_label,
)


def _commit(action="pay", labels=None):
    return Event("agent", EventKind.SIDE_EFFECT_COMMIT, {"action": action, "labels": labels or []})


def test_safety_monitor_absorbing_violation():
    m = SafetyMonitor(lambda e: e.action == "bad", "no_bad")
    m.step(_commit("ok"))
    assert m.verdict() == Verdict.INCONCLUSIVE
    m.step(_commit("bad"))
    assert m.verdict() == Verdict.VIOLATED
    # absorbing: subsequent good events do not clear it
    m.step(_commit("ok"))
    assert m.finalize() == Verdict.VIOLATED


def test_safety_monitor_clean_run_satisfied_on_finalize():
    m = SafetyMonitor(lambda e: e.action == "bad", "no_bad")
    m.step(_commit("ok"))
    assert m.finalize() == Verdict.SATISFIED


def test_cosafety_monitor():
    m = CoSafetyMonitor(lambda e: e.action == "redact", "eventually_redact")
    assert m.verdict() == Verdict.INCONCLUSIVE
    assert m.finalize() == Verdict.VIOLATED  # good never happened
    m.reset()
    m.step(_commit("redact"))
    assert m.verdict() == Verdict.SATISFIED
    assert m.finalize() == Verdict.SATISFIED


def test_no_side_effect_without_label():
    pred = no_side_effect_without_label("approved")
    m = pred.monitor()
    assert m.run([_commit("pay", ["approved"])]) == Verdict.SATISFIED
    m2 = pred.monitor()
    assert m2.run([_commit("pay", [])]) == Verdict.VIOLATED


def test_pii_egress_guarantee():
    pred = no_pii_egress_without_redaction()
    leaked = Event("agent", EventKind.MESSAGE_SEND, {"labels": ["pii"]})
    safe = Event("agent", EventKind.MESSAGE_SEND, {"labels": ["pii", "redacted"]})
    assert pred.monitor().run([safe]) == Verdict.SATISFIED
    assert pred.monitor().run([leaked]) == Verdict.VIOLATED


def test_guard_miss_is_the_bernoulli_event():
    # A guard "miss" = it approves an action that is actually a policy violation.
    pred = guard_never_wrongly_approves()
    approve_unsafe = Event(
        "guard", EventKind.LABEL_ASSIGN, {"labels": ["approved", "policy_violation"]}
    )
    approve_safe = Event("guard", EventKind.LABEL_ASSIGN, {"labels": ["approved"]})
    assert pred.monitor().run([approve_safe]) == Verdict.SATISFIED
    assert pred.monitor().run([approve_unsafe]) == Verdict.VIOLATED


def test_dfa_monitor_extension_point():
    # Toy DFA: VIOLATED once two "risky" events are seen without an intervening
    # "reset" -- a regular pattern beyond the single-event safety shape.
    def trans(state, e: Event):
        if e.action == "reset":
            return 0
        if e.action == "risky":
            return state + 1
        return state

    m = DFAMonitor(trans, start=0, rejecting={2}, name="two_risky")
    m.step(_commit("risky"))
    assert m.verdict() == Verdict.INCONCLUSIVE
    m.step(_commit("risky"))
    assert m.verdict() == Verdict.VIOLATED

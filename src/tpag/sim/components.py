"""Components for the TPAG replication package."""

from __future__ import annotations

import numpy as np

from ..events import Event, EventKind, Trace
from ..vectorclock import VectorClockEngine


def sample_correlated_misses(
    benign_rates: list[float], c: float, rng: np.random.Generator
) -> list[bool]:
    """Sample a guard-miss vector under the shared-mode model with coupling ``c``."""
    if rng.random() < c:
        return [True] * len(benign_rates)
    return [bool(rng.random() < q) for q in benign_rates]


def build_workflow_trace(
    guard_names: list[str],
    guard_misses: list[bool],
    *,
    planner: str = "planner",
    actor: str = "actor",
    action: str = "issue_refund",
    pii: bool = False,
    redaction_ok: bool = True,
    with_assumption_edge: bool = True,
) -> Trace:
    """Build one episode's causal trace for an approval workflow.

    The planner activates, sends work to each guard and the actor; each guard
    emits an approval decision (a *miss* approves an unsafe action, attaching both
    ``approved`` and ``policy_violation``); the actor commits the side effect. When
    ``with_assumption_edge`` is True the guards' approvals are *sent* to the actor
    (so the actor's "approved upstream" assumption is causally discharged).
    """
    eng = VectorClockEngine()
    t = Trace()

    # planner activates and dispatches (a send the others will receive)
    p_act = eng.local(planner)
    t.append(Event(planner, EventKind.ACTIVATION, {"node": planner}, vc=p_act))
    dispatch = eng.send(planner)
    t.append(Event(planner, EventKind.MESSAGE_SEND, {"to": "agents", "data": "REQ"}, vc=dispatch))

    approvals = []
    for name, missed in zip(guard_names, guard_misses):
        g_recv = eng.recv(name, dispatch)
        t.append(Event(name, EventKind.ACTIVATION, {"node": name, "data": "REQ"}, vc=g_recv))
        labels = ["approved", "policy_violation"] if missed else ["blocked"]
        g_label = eng.local(name)
        t.append(Event(name, EventKind.LABEL_ASSIGN, {"labels": labels, "data": "REQ"}, vc=g_label))
        approve_send = eng.send(name)
        t.append(
            Event(
                name,
                EventKind.MESSAGE_SEND,
                {"to": actor, "labels": labels, "data": "REQ"},
                vc=approve_send,
            )
        )
        approvals.append((name, approve_send, missed))

    # actor receives approvals (causally) then acts
    a_clock = None
    for _name, stamp, _missed in approvals:
        a_clock = eng.recv(actor, stamp) if with_assumption_edge else eng.local(actor)
    if a_clock is None:
        a_clock = eng.local(actor)
    t.append(Event(actor, EventKind.ACTIVATION, {"node": actor, "data": "REQ"}, vc=a_clock))

    # the side effect: labeled approved iff every guard approved (i.e. all missed,
    # for the unsafe attack action), and pii/redaction labels for the PII property.
    all_approved = all(m for _, _, m in approvals)
    labels = []
    if all_approved:
        labels.append("approved")
    labels.append("policy_violation")  # the attack action is genuinely unsafe
    if pii:
        labels.append("pii")
        if redaction_ok:
            labels.append("redacted")
    commit = eng.local(actor)
    t.append(
        Event(
            actor,
            EventKind.SIDE_EFFECT_COMMIT,
            {"action": action, "labels": labels, "data": "REQ"},
            vc=commit,
        )
    )
    return t

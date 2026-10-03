"""Predicates for the TPAG replication package."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from .events import Event, EventKind, Label
from .monitors import CoSafetyMonitor, EventPredicate, Monitor, SafetyMonitor


class PredicateKind(str, Enum):
    SAFETY = "safety"
    COSAFETY = "co-safety"


@dataclass
class Predicate:
    """A monitorable trace predicate."""

    name: str
    kind: PredicateKind
    make_monitor: Callable[[], Monitor]

    def monitor(self) -> Monitor:
        return self.make_monitor()


# --------------------------------------------------------------------------
# Safety builders: "nothing bad ever happens"
# --------------------------------------------------------------------------
def never(bad_event: EventPredicate, name: str) -> Predicate:
    """G = it is never the case that ``bad_event`` holds for an observed event."""
    return Predicate(name, PredicateKind.SAFETY, lambda: SafetyMonitor(bad_event, name))


def always(good_event: EventPredicate, name: str) -> Predicate:
    """G = every observed event satisfies ``good_event`` (i.e. never ¬good)."""
    return Predicate(
        name, PredicateKind.SAFETY, lambda: SafetyMonitor(lambda e: not good_event(e), name)
    )


# --------------------------------------------------------------------------
# Co-safety builders: "something good eventually happens"
# --------------------------------------------------------------------------
def eventually(good_event: EventPredicate, name: str) -> Predicate:
    """G = some observed event satisfies ``good_event`` before the trace ends."""
    return Predicate(name, PredicateKind.COSAFETY, lambda: CoSafetyMonitor(good_event, name))


# --------------------------------------------------------------------------
# Domain patterns for the T1 approval / PII-egress workflow
# --------------------------------------------------------------------------
def _is_side_effect(e: Event, action: str | None = None) -> bool:
    if e.kind not in (EventKind.SIDE_EFFECT_COMMIT, EventKind.TOOL_CALL):
        return False
    return action is None or e.action == action


def no_side_effect_without_label(required: Label, name: str | None = None) -> Predicate:
    """Safety: every side-effecting commit/tool-call carries ``required`` label.

    The canonical guard guarantee, e.g. an action may only commit if it is
    labeled ``approved`` (policy guard) or ``redacted`` (PII guard). A side
    effect lacking the label is the bad event.
    """
    nm = name or f"no_side_effect_without_{required}"

    def bad(e: Event) -> bool:
        return _is_side_effect(e) and not e.has_label(required)

    return never(bad, nm)


def no_pii_egress_without_redaction(name: str = "no_pii_egress_without_redaction") -> Predicate:
    """Safety: no event labeled ``pii`` egresses without a ``redacted`` label."""

    def bad(e: Event) -> bool:
        egress = e.kind in (
            EventKind.SIDE_EFFECT_COMMIT,
            EventKind.MESSAGE_SEND,
            EventKind.TOOL_CALL,
        )
        return egress and e.has_label("pii") and not e.has_label("redacted")

    return never(bad, name)


def guard_never_wrongly_approves(
    approve_label: Label = "approved",
    unsafe_label: Label = "policy_violation",
    name: str = "guard_sound",
) -> Predicate:
    """Safety guarantee of a *guard* component.

    The guard's job is to attach ``approve_label`` only to safe actions. Its
    guarantee is violated exactly when it approves an action that actually
    carries the ground-truth ``unsafe_label`` -- i.e. a *miss*. This is the
    Bernoulli event whose rate is the guard's contract ``epsilon`` (Def. 3), and
    the event that becomes positively correlated across redundant guards under a
    shared-mode attack (Prop. 1).
    """

    def bad(e: Event) -> bool:
        return (
            e.kind == EventKind.LABEL_ASSIGN
            and e.has_label(approve_label)
            and e.has_label(unsafe_label)
        )

    return never(bad, name)

"""Causal for the TPAG replication package."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .events import Event, EventKind, Label, Trace

# An assumption check decides, from the set of *past* events and the activation,
# whether the assumption A_i is discharged (True = discharged / in-context).
AssumptionCheck = Callable[[list[Event], Event], bool]


def temporal_past(trace: Trace, activation: Event) -> list[Event]:
    """Recording-order prefix: every event logged strictly before ``activation``.

    This is the (unsound) "global clock" view of the past -- it ignores whether
    an event is *causally* related to the activation.
    """
    return [e for e in trace.events if e.seq < activation.seq]


def requires_upstream_label(label: Label, data_key: str = "data") -> AssumptionCheck:
    """A_i = the data this activation acts on was labeled ``label`` upstream.

    Discharged iff some past event assigns ``label`` to the *same* data item the
    activation references (``payload[data_key]``). With the causal past this means
    the labeling causally precedes the action; with the temporal past it only
    means it was logged earlier.
    """

    def check(past: list[Event], activation: Event) -> bool:
        data = activation.payload.get(data_key)
        for e in past:
            if e.kind == EventKind.LABEL_ASSIGN and e.has_label(label):
                if data is None or e.payload.get(data_key) == data:
                    return True
        return False

    return check


@dataclass
class DischargeResult:
    discharged: bool
    activation: Event
    past_size: int
    monitor: str

    @property
    def in_context(self) -> bool:
        return self.discharged


class CausalMonitor:
    """Sound assumption-discharge monitor over the causal (happens-before) past."""

    name = "causal"

    def discharged(
        self, trace: Trace, activation: Event, check: AssumptionCheck
    ) -> DischargeResult:
        past = trace.causal_past(activation)
        return DischargeResult(check(past, activation), activation, len(past), self.name)


class GlobalClockMonitor:
    """Naive baseline: assumption discharge over the recording-order prefix.

    Provided as the comparator that the causal monitor improves upon; under a
    message race it can wrongly report an assumption discharged (Sec. V).
    """

    name = "global-clock"

    def discharged(
        self, trace: Trace, activation: Event, check: AssumptionCheck
    ) -> DischargeResult:
        past = temporal_past(trace, activation)
        return DischargeResult(check(past, activation), activation, len(past), self.name)

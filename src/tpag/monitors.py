"""Monitors for the TPAG replication package."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

from .events import Event

EventPredicate = Callable[[Event], bool]


class Verdict(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    INCONCLUSIVE = "inconclusive"


class Monitor:
    """Abstract online monitor over a stream of :class:`Event`."""

    name: str

    def step(self, event: Event) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def verdict(self) -> Verdict:  # pragma: no cover - abstract
        raise NotImplementedError

    def finalize(self) -> Verdict:
        """Collapse INCONCLUSIVE to a 2-valued finite-trace outcome."""
        v = self.verdict()
        return v if v != Verdict.INCONCLUSIVE else Verdict.SATISFIED

    def reset(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def run(self, events) -> Verdict:
        """Convenience: feed an iterable of events and finalize."""
        self.reset()
        for e in events:
            self.step(e)
        return self.finalize()


class SafetyMonitor(Monitor):
    """Safety: "nothing bad ever happens" (G = always ¬bad).

    Starts OPEN; the first event for which ``bad_event`` holds flips it to the
    absorbing VIOLATED state. ``finalize()`` of an OPEN monitor is SATISFIED.
    """

    def __init__(self, bad_event: EventPredicate, name: str = "safety") -> None:
        self.bad_event = bad_event
        self.name = name
        self._violated = False
        self._witness: Event | None = None

    def step(self, event: Event) -> None:
        if self._violated:
            return
        if self.bad_event(event):
            self._violated = True
            self._witness = event

    def verdict(self) -> Verdict:
        return Verdict.VIOLATED if self._violated else Verdict.INCONCLUSIVE

    def finalize(self) -> Verdict:
        return Verdict.VIOLATED if self._violated else Verdict.SATISFIED

    @property
    def witness(self) -> Event | None:
        return self._witness

    def reset(self) -> None:
        self._violated = False
        self._witness = None


class CoSafetyMonitor(Monitor):
    """Co-safety: "something good eventually happens" (G = eventually good).

    Starts OPEN; the first event for which ``good_event`` holds flips it to the
    absorbing SATISFIED state. ``finalize()`` of an OPEN monitor is VIOLATED
    (the good thing never happened).
    """

    def __init__(self, good_event: EventPredicate, name: str = "cosafety") -> None:
        self.good_event = good_event
        self.name = name
        self._satisfied = False
        self._witness: Event | None = None

    def step(self, event: Event) -> None:
        if self._satisfied:
            return
        if self.good_event(event):
            self._satisfied = True
            self._witness = event

    def verdict(self) -> Verdict:
        return Verdict.SATISFIED if self._satisfied else Verdict.INCONCLUSIVE

    def finalize(self) -> Verdict:
        return Verdict.SATISFIED if self._satisfied else Verdict.VIOLATED

    @property
    def witness(self) -> Event | None:
        return self._witness

    def reset(self) -> None:
        self._satisfied = False
        self._witness = None


class DFAMonitor(Monitor):
    """A deterministic-finite-automaton monitor over the event alphabet.

    Extension point for regular safety/co-safety patterns beyond the two
    canonical shapes above. ``transition(state, event) -> state`` advances a
    deterministic state; ``accepting``/``rejecting`` sets are absorbing verdicts.

    Parameters
    ----------
    transition:
        Deterministic transition function.
    start, accepting, rejecting:
        DFA structure. Reaching ``rejecting`` yields VIOLATED (safety sink);
        reaching ``accepting`` yields SATISFIED (co-safety sink). Otherwise the
        verdict is INCONCLUSIVE; ``cosafety=True`` makes ``finalize()`` of an
        undecided run VIOLATED (good never reached) instead of SATISFIED.
    """

    def __init__(
        self,
        transition: Callable[[Any, Event], Any],
        start: Any,
        accepting: set[Any] | None = None,
        rejecting: set[Any] | None = None,
        name: str = "dfa",
        cosafety: bool = False,
    ) -> None:
        self.transition = transition
        self.start = start
        self.accepting = accepting or set()
        self.rejecting = rejecting or set()
        self.name = name
        self.cosafety = cosafety
        self._state = start

    def step(self, event: Event) -> None:
        if self._state in self.accepting or self._state in self.rejecting:
            return
        self._state = self.transition(self._state, event)

    def verdict(self) -> Verdict:
        if self._state in self.rejecting:
            return Verdict.VIOLATED
        if self._state in self.accepting:
            return Verdict.SATISFIED
        return Verdict.INCONCLUSIVE

    def finalize(self) -> Verdict:
        v = self.verdict()
        if v != Verdict.INCONCLUSIVE:
            return v
        return Verdict.VIOLATED if self.cosafety else Verdict.SATISFIED

    @property
    def state(self) -> Any:
        return self._state

    def reset(self) -> None:
        self._state = self.start

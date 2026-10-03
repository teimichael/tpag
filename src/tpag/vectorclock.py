"""Vectorclock for the TPAG replication package."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VectorClock:
    """A vector clock over a set of lifelines (agents/components).

    Internally a mapping ``lifeline -> logical_time``. Missing keys are treated
    as zero, so clocks over different lifeline sets compose cleanly.
    """

    clock: dict[str, int] = field(default_factory=dict)

    def get(self, lifeline: str) -> int:
        return self.clock.get(lifeline, 0)

    def copy(self) -> VectorClock:
        return VectorClock(dict(self.clock))

    def tick(self, lifeline: str) -> VectorClock:
        """Advance this lifeline's component by one (a local event). In place."""
        self.clock[lifeline] = self.clock.get(lifeline, 0) + 1
        return self

    def merge(self, other: VectorClock) -> VectorClock:
        """Component-wise max with ``other`` (receiving a message). In place."""
        for k, v in other.clock.items():
            if v > self.clock.get(k, 0):
                self.clock[k] = v
        return self

    # -- partial order -------------------------------------------------------
    def _le(self, other: VectorClock) -> bool:
        keys = set(self.clock) | set(other.clock)
        return all(self.get(k) <= other.get(k) for k in keys)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        keys = set(self.clock) | set(other.clock)
        return all(self.get(k) == other.get(k) for k in keys)

    def happens_before(self, other: VectorClock) -> bool:
        """True iff ``self`` strictly causally precedes ``other`` (self < other)."""
        return self._le(other) and self != other

    def concurrent_with(self, other: VectorClock) -> bool:
        """True iff neither event causally precedes the other."""
        return not self.happens_before(other) and not other.happens_before(self) and self != other

    def __le__(self, other: VectorClock) -> bool:
        return self._le(other)

    def __lt__(self, other: VectorClock) -> bool:
        return self.happens_before(other)

    # -- serialization -------------------------------------------------------
    def to_dict(self) -> dict[str, int]:
        return dict(self.clock)

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> VectorClock:
        return cls(dict(d))

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}:{v}" for k, v in sorted(self.clock.items()))
        return f"VC({inner})"


class VectorClockEngine:
    """Maintains one vector clock per lifeline and stamps events causally.

    Usage mirrors the standard send/receive protocol:

    * ``local(lifeline)``  -- an internal event on ``lifeline``.
    * ``send(lifeline)``   -- a message send; returns the stamp to attach.
    * ``recv(lifeline, stamp)`` -- delivery of a message stamped ``stamp``.

    Every call returns a *snapshot* (immutable copy) suitable for attaching to
    an :class:`~tpag.events.Event`.
    """

    def __init__(self) -> None:
        self._clocks: dict[str, VectorClock] = {}

    def _clock(self, lifeline: str) -> VectorClock:
        return self._clocks.setdefault(lifeline, VectorClock())

    def local(self, lifeline: str) -> VectorClock:
        return self._clock(lifeline).tick(lifeline).copy()

    def send(self, lifeline: str) -> VectorClock:
        return self._clock(lifeline).tick(lifeline).copy()

    def recv(self, lifeline: str, stamp: VectorClock) -> VectorClock:
        c = self._clock(lifeline)
        c.tick(lifeline).merge(stamp)
        return c.copy()

    def current(self, lifeline: str) -> VectorClock:
        return self._clock(lifeline).copy()

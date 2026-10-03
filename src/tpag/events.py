"""Events for the TPAG replication package."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .vectorclock import VectorClock


class EventKind(str, Enum):
    """Kinds of observable actions a component may emit."""

    ACTIVATION = "activation"  # a component begins acting (act(i))
    MESSAGE_SEND = "message_send"
    MESSAGE_RECV = "message_recv"
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    LABEL_ASSIGN = "label_assign"  # a provenance/trust label is attached
    TOOL_CALL = "tool_call"
    SIDE_EFFECT_INTENT = "side_effect_intent"  # two-phase: proposed, not committed
    SIDE_EFFECT_COMMIT = "side_effect_commit"  # two-phase: actually executed
    GUARANTEE_VERDICT = "guarantee_verdict"  # a guarantee monitor's outcome
    ASSUMPTION_VERDICT = "assumption_verdict"  # an assumption monitor's outcome
    LLM_CALL = "llm_call"  # an LLM invocation (prompt+response logged)


# A provenance/trust label is just a tag; a data item carries a set of them.
Label = str


@dataclass
class Event:
    """A single observable event on a lifeline.

    Attributes
    ----------
    lifeline:
        The component/agent instance that emitted the event (Def. 1).
    kind:
        One of :class:`EventKind`.
    payload:
        Structured, JSON-serializable detail. Conventionally may include keys
        ``action``, ``data``, ``labels`` (list[str]), ``node``, ``verdict``,
        ``target`` (a referenced node/guarantee), ``prompt``, ``response``.
    vc:
        Causal stamp (set by the runtime / vector-clock engine). ``None`` until
        stamped.
    seq:
        A global monotonic recording sequence number (total order for replay and
        deterministic tie-breaking; NOT a causal order).
    wall_time:
        Wall-clock timestamp (seconds). Used only for overhead measurement
        (RQ5), never for causality.
    eid:
        Unique event id.
    """

    lifeline: str
    kind: EventKind
    payload: dict[str, Any] = field(default_factory=dict)
    vc: VectorClock | None = None
    seq: int = -1
    wall_time: float = field(default_factory=time.time)
    eid: str = ""

    _counter: itertools.count = field(  # class-level id source
        default=itertools.count(1), init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.eid:
            self.eid = f"e{next(Event._counter)}"

    # -- convenience queries -------------------------------------------------
    @property
    def labels(self) -> frozenset[Label]:
        return frozenset(self.payload.get("labels", []))

    @property
    def action(self) -> str | None:
        return self.payload.get("action")

    @property
    def target(self) -> str | None:
        """A node/guarantee this event references (e.g. an upstream guarantee)."""
        return self.payload.get("target")

    def has_label(self, label: Label) -> bool:
        return label in self.labels

    def happens_before(self, other: Event) -> bool:
        if self.vc is None or other.vc is None:
            raise ValueError("events must be causally stamped before comparison")
        return self.vc.happens_before(other.vc)

    # -- serialization -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "eid": self.eid,
            "lifeline": self.lifeline,
            "kind": self.kind.value,
            "payload": self.payload,
            "vc": self.vc.to_dict() if self.vc is not None else None,
            "seq": self.seq,
            "wall_time": self.wall_time,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Event:
        ev = cls(
            lifeline=d["lifeline"],
            kind=EventKind(d["kind"]),
            payload=d.get("payload", {}),
            vc=VectorClock.from_dict(d["vc"]) if d.get("vc") is not None else None,
            seq=d.get("seq", -1),
            wall_time=d.get("wall_time", 0.0),
            eid=d.get("eid", ""),
        )
        return ev


@dataclass
class Trace:
    """An append-only, totally-recorded sequence of events.

    The recording order (``seq``) is a total order used for replay; causal
    relationships between events are given by their vector clocks, not by ``seq``.
    """

    events: list[Event] = field(default_factory=list)

    def append(self, event: Event) -> Event:
        event.seq = len(self.events)
        self.events.append(event)
        return event

    def __iter__(self):
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def by_lifeline(self, lifeline: str) -> list[Event]:
        return [e for e in self.events if e.lifeline == lifeline]

    def of_kind(self, kind: EventKind) -> list[Event]:
        return [e for e in self.events if e.kind == kind]

    def causal_past(self, event: Event) -> list[Event]:
        """All recorded events that causally precede ``event`` (Def. 6).

        The causal past of an activation is the set ``{e : e < act(i)}`` over the
        happens-before order. Assumption monitors are evaluated over exactly this
        set so that "upstream guarantees hold" is judged on genuine causal history
        rather than wall-clock arrival.
        """
        if event.vc is None:
            raise ValueError("event must be causally stamped")
        return [e for e in self.events if e.vc is not None and e.vc.happens_before(event.vc)]

    def to_jsonl(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.events]

    @classmethod
    def from_jsonl(cls, rows: list[dict[str, Any]]) -> Trace:
        t = cls()
        t.events = [Event.from_dict(r) for r in rows]
        return t

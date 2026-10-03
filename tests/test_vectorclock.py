"""Test vectorclock for the TPAG replication package."""

from tpag.vectorclock import VectorClock, VectorClockEngine


def test_happens_before_basic():
    a = VectorClock({"p": 1})
    b = VectorClock({"p": 2})
    assert a.happens_before(b)
    assert not b.happens_before(a)
    assert not a.concurrent_with(b)


def test_concurrency():
    a = VectorClock({"p": 1})  # event on p
    b = VectorClock({"q": 1})  # event on q, no causal link
    assert a.concurrent_with(b)
    assert not a.happens_before(b)
    assert not b.happens_before(a)


def test_engine_send_recv_orders_events():
    eng = VectorClockEngine()
    # p does a local event, then sends to q.
    p1 = eng.local("p")
    send = eng.send("p")
    # q receives the message.
    q1 = eng.recv("q", send)
    # The send on p causally precedes the receive on q.
    assert send.happens_before(q1)
    assert p1.happens_before(q1)
    # An independent event on r is concurrent with p's first event.
    r1 = eng.local("r")
    assert r1.concurrent_with(p1)


def test_engine_transitivity():
    eng = VectorClockEngine()
    a = eng.send("p")
    b = eng.recv("q", a)
    c = eng.send("q")
    d = eng.recv("r", c)
    assert a.happens_before(b)
    assert b.happens_before(d)
    assert a.happens_before(d)  # transitive causal chain p -> q -> r


def test_serialization_roundtrip():
    a = VectorClock({"p": 3, "q": 1})
    b = VectorClock.from_dict(a.to_dict())
    assert a == b

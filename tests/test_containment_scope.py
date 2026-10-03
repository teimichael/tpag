"""Test containment scope for the TPAG replication package."""

from tpag.containment import ContainmentController, GlobalContainment, localization_score
from tpag.graph import OrchestrationGraph


def _diamond() -> OrchestrationGraph:
    # planner -> a -> b ; planner -> c -> d  (two independent branches)
    g = OrchestrationGraph()
    for n in ["planner", "a", "b", "c", "d"]:
        g.add_component(n)
    g.add_edge("planner", "a")
    g.add_edge("a", "b")
    g.add_edge("planner", "c")
    g.add_edge("c", "d")
    return g


def test_dominated_subdag():
    g = _diamond()
    assert g.dominated_subdag("a") == {"a", "b"}
    assert g.dominated_subdag("c") == {"c", "d"}
    assert g.dominated_subdag("planner") == {"planner", "a", "b", "c", "d"}


def test_scoped_containment_blocks_only_affected_branch():
    g = _diamond()
    ctrl = ContainmentController(g)
    ev = ctrl.contain("a")
    assert ev.scope == frozenset({"a", "b"})
    assert ctrl.is_blocked("b")
    assert not ctrl.is_blocked("c")  # independent branch keeps running
    assert not ctrl.may_commit("b")
    assert ctrl.may_commit("c")


def test_global_containment_halts_everything():
    g = _diamond()
    ctrl = GlobalContainment(g)
    ev = ctrl.contain("a")
    assert ev.scope == frozenset(g.nodes)
    assert ctrl.is_blocked("c")  # unrelated branch halted too


def test_localization_precision_scoped_beats_global():
    g = _diamond()
    truth = {"a", "b"}  # the breach at 'a' genuinely affects a and b
    scoped = ContainmentController(g).contain("a").scope
    glob = GlobalContainment(g).contain("a").scope

    s = localization_score(set(scoped), truth, n_nodes=len(g.nodes))
    gl = localization_score(set(glob), truth, n_nodes=len(g.nodes))

    assert s.precision == 1.0 and s.recall == 1.0
    assert s.halted_fraction < gl.halted_fraction  # scoped halts less
    assert gl.precision < 1.0  # global over-blocks

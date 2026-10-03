"""Test structure cutsets for the TPAG replication package."""

from tpag.structure import (
    Atom,
    ExplicitCutSets,
    check_adequacy,
    kofn,
    minimal_sets,
    parallel,
    series,
)


def test_series_cut_sets_are_singletons():
    s = series("a", "b", "c")
    cuts = s.cut_sets()
    assert set(cuts) == {frozenset({"a"}), frozenset({"b"}), frozenset({"c"})}


def test_parallel_cut_set_is_the_whole_group():
    s = parallel("g1", "g2")
    cuts = s.cut_sets()
    assert cuts == [frozenset({"g1", "g2"})]


def test_nested_series_of_parallel():
    # delegation chain where one stage is a 2-redundant guard pair.
    s = series("planner", parallel("guardA", "guardB"))
    cuts = set(s.cut_sets())
    assert cuts == {frozenset({"planner"}), frozenset({"guardA", "guardB"})}


def test_semantic_failed_matches_cut_sets():
    s = series("planner", parallel("guardA", "guardB"))
    assert s.failed({"planner"})  # series link broke
    assert s.failed({"guardA", "guardB"})  # both guards missed
    assert not s.failed({"guardA"})  # one guard missing is contained by the other
    assert not s.failed(set())  # no violations -> Phi holds


def test_kofn():
    # 2-of-3 must work => system fails iff >= 2 fail.
    s = kofn(2, "a", "b", "c")
    cuts = set(s.cut_sets())
    assert cuts == {frozenset({"a", "b"}), frozenset({"a", "c"}), frozenset({"b", "c"})}


def test_minimal_sets_removes_supersets():
    fam = [frozenset({"a"}), frozenset({"a", "b"}), frozenset({"b", "c"})]
    assert set(minimal_sets(fam)) == {frozenset({"a"}), frozenset({"b", "c"})}


def test_adequacy_pass():
    s = series("planner", parallel("guardA", "guardB"))
    rep = check_adequacy(s, {"planner", "guardA", "guardB"})
    assert rep.ok
    assert not rep.errors


def test_adequacy_detects_undeclared_component():
    s = series("planner", "ghost")
    rep = check_adequacy(s, {"planner"})
    assert not rep.ok
    assert any("undeclared" in e for e in rep.errors)


def test_adequacy_warns_on_dead_weight():
    s = Atom("planner")
    rep = check_adequacy(s, {"planner", "unused_guard"})
    assert rep.ok  # warnings are not errors
    assert any("dead weight" in w for w in rep.warnings)


def test_explicit_cut_sets():
    s = ExplicitCutSets([frozenset({"x", "y"}), frozenset({"z"})])
    assert s.failed({"z"})
    assert s.failed({"x", "y"})
    assert not s.failed({"x"})

"""Test strengthening for the TPAG replication package."""

from __future__ import annotations
import pytest
from tpag.eval.config import load_config


def _synthetic_points():
    pts = []
    for i, (div, hard, b, f) in enumerate(
        [
            ("cross_family", "strict", 0.03, 1.0),
            ("cross_family", "strict", 0.09, 0.5),
            ("cross_family", "permissive", 0.4, 0.0),
            ("same_model", "permissive", 0.6, 0.0),
            ("same_model", "strict", 0.95, 1.0),
            ("same_model", "permissive", 1.0, 1.0),
        ]
    ):
        pts.append(
            {
                "group": f"g{i}",
                "diversity": div,
                "hardening": hard,
                "tpag_bound": b,
                "injection_fraction": f,
                "models": [f"m{i}", f"n{i}"],
                "confidence": 0.997,
                "observed_comiss": b * 0.5,
                "k": 2,
            }
        )
    return pts


def test_bound_value_summary_shape_and_exemplar():
    from tpag.eval.bound_values import summarize_bound_values

    s = summarize_bound_values(_synthetic_points())
    assert s["n_points"] == 6
    assert s["floor"] == pytest.approx(0.03)
    assert s["frac_le"]["0.1"] == pytest.approx(2 / 6)
    assert (
        s["by_diversity"]["same_model"]["median_bound"]
        >= s["by_diversity"]["cross_family"]["median_bound"]
    )
    ex = s["non_vacuous_exemplar"]
    assert ex["diversity"] == "cross_family" and ex["hardening"] == "strict"
    assert ex["tpag_bound"] == pytest.approx(0.03)


def test_bound_value_summary_empty():
    from tpag.eval.bound_values import summarize_bound_values

    assert summarize_bound_values([]) == {"n_points": 0}


def test_coherence_lint_accepts_monotone_and_rejects_nonmonotone():
    from tpag.structure import (
        ExplicitCutSets,
        Structure,
        check_adequacy,
        coherence_violation,
        parallel,
        series,
    )

    good = series(parallel("a", "b"), "c")
    assert coherence_violation(good) is None
    assert check_adequacy(good, {"a", "b", "c"}).ok

    class NonMonotone(Structure):
        def cut_sets(self):
            return [frozenset({"a"})]

        def atoms(self):
            return {"a", "b"}

        def failed(self, violated):
            return "a" in violated and "b" not in violated

    viol = coherence_violation(NonMonotone())
    assert viol is not None
    rep = check_adequacy(NonMonotone(), {"a", "b"})
    assert not rep.ok and any(("coherence" in e for e in rep.errors))
    assert coherence_violation(ExplicitCutSets([frozenset({"a", "b"}), frozenset({"c"})])) is None


def test_rq_e6_ablation_driver(tmp_path):
    from tpag.eval.experiments import rq_e6_estimation_ablation as e6

    cfg = load_config(tier="quick", out=str(tmp_path))
    res = e6.run(cfg, use_llm=False)
    m = res["metrics"]
    assert res["status"] == "computed"
    assert m["cp_unsound_configs"] == 0
    assert m["sound_cert_tighter_than_boole_fraction"] == 1.0
    assert m["mean_boole_minus_tpag_cp"] >= 0.0
    assert (tmp_path / "rq_e6_estimation_ablation" / "metrics.json").exists()


def test_e6_point_vs_cp_bound_property():
    """On a correlated redundancy cut the CP-limit bound is a valid (sound) upper
    bound on the observed co-miss; the point-estimate Hunter--Worsley can subtract
    more (it is <= the cp bound)."""
    from tpag.eval.experiments.rq_e6_estimation_ablation import _bound_two_ways
    from tpag.runtime import TPAGRuntime
    from tpag.sim import redundancy_gadget

    wl = redundancy_gadget(k=2, benign_rate=0.1, coupling=0.5)
    rt = TPAGRuntime(wl.graph, wl.structure, wl.contracts, wl.assumption_checks, regime=wl.regime)
    rt.ingest_many(wl.generate(4000, seed=2))
    bb = _bound_two_ways(wl.structure, rt.estimation, eta=0.01)
    assert bb["tpag_cp"] + 1e-09 >= rt.observed_violation_rate
    assert bb["tpag_cp"] <= bb["boole_cp"] + 1e-09


def test_rq_e9_hard_containment_driver(tmp_path):
    from tpag.eval.experiments import rq_e9_hard_containment as e9

    cfg = load_config(tier="quick", out=str(tmp_path))
    res = e9.run(cfg, use_llm=False)
    m = res["metrics"]
    assert m["cases_with_recall_below_1"] >= 1
    assert m["scoped_precision_is_perfect"] is True
    assert m["causal_uncontained_commit_rate"] == 0.0
    assert m["global_uncontained_commit_rate"] > 0.0
    assert (tmp_path / "rq_e9_hard_containment" / "hard_cases.jsonl").exists()

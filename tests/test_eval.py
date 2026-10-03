"""Test eval for the TPAG replication package."""

import numpy as np
from tpag.eval import baselines, comparison, stats
from tpag.runtime import TPAGRuntime
from tpag.sim import redundancy_gadget, series_chain
from tpag.sim.topology_gen import soundness_suite
from tpag.structure import parallel


def test_multiplicative_vs_tpag_baseline():
    s = parallel("g1", "g2")
    eps = {"g1": 0.1, "g2": 0.1}
    assert np.isclose(baselines.multiplicative_bound(s, eps), 0.01)


def test_ablation_ordering_boole_ge_frechet_ge_hw():
    """The ablation switches can only tighten: Boole >= Fréchet+joints >= TPAG (HW).
    Joints tighten a correlated redundancy cut; HW tightens a correlated series chain."""
    wl = redundancy_gadget(k=2, benign_rate=0.1, coupling=0.5)
    rt = TPAGRuntime(wl.graph, wl.structure, wl.contracts, wl.assumption_checks, regime=wl.regime)
    rt.ingest_many(wl.generate(4000, seed=3))
    eps_u, pair_u = (rt.estimation.eps_upper(0.01), rt.estimation.pair_upper(0.01))
    boole = baselines.boole_bound(wl.structure, eps_u)
    frechet = baselines.frechet_joints_bound(wl.structure, eps_u, pair_u)
    hw = rt.certificate(0.01).b_hw
    assert boole + 1e-09 >= frechet + 1e-09 >= hw
    assert frechet < boole
    ws = series_chain(k=3, benign_rate=0.08, coupling=0.5)
    rs = TPAGRuntime(ws.graph, ws.structure, ws.contracts, ws.assumption_checks, regime=ws.regime)
    rs.ingest_many(ws.generate(4000, seed=4))
    cert = rs.certificate(0.01)
    assert cert.b_hw < cert.b_boole


def test_score_composition_rules_coverage_and_unsoundness():
    """The composition-rule scorer (reused by RQ-E4/RQ-E6) over the topology suite:
    the sound rules cover every config; the independence product is unsound on >=1."""
    configs = soundness_suite((4, 8), seed=1)
    rows, per_rule = comparison.score_composition_rules(configs, N=2000, seed=1)
    assert rows and per_rule["tpag_hw"]["coverage"] == 1.0
    assert per_rule["boole"]["coverage"] == 1.0
    assert per_rule["frechet_joints"]["coverage"] == 1.0
    assert per_rule["independence"]["coverage"] < 1.0
    assert sum((r["independence_violated"] for r in rows)) >= 1
    assert per_rule["tpag_hw"]["mean_slack_sound"] <= per_rule["boole"]["mean_slack_sound"] + 1e-09


def test_bootstrap_ci_contains_mean():
    data = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 0] * 10, dtype=float)
    mean, lo, hi = stats.bootstrap_ci(data, n_boot=500, seed=1)
    assert lo <= mean <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_calibration_summary():
    cal = stats.calibration_summary(
        observed=[0.1, 0.2, 0.05], b_boole=[0.3, 0.3, 0.2], b_hw=[0.25, 0.28, 0.18]
    )
    assert cal.coverage == 1.0
    assert cal.violations == 0
    assert cal.mean_hw_improvement > 0

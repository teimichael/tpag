"""Test kway joints for the TPAG replication package."""

import numpy as np

from tpag.certificate import certify
from tpag.estimation import EstimationModel
from tpag.runtime import TPAGRuntime, _relevant_joints
from tpag.sim import redundancy_gadget
from tpag.structure import Atom, Parallel


def _three_parallel() -> Parallel:
    return Parallel([Atom("g0"), Atom("g1"), Atom("g2")])


def test_certify_uses_registered_full_cut_joint():
    """A registered 3-way joint counter (0 co-violations) must pull the single-cut
    bound below every pairwise Fréchet term."""
    structure = _three_parallel()
    est = EstimationModel()
    n = 400
    for g in ("g0", "g1", "g2"):
        c = est.marginal(g)
        for i in range(n):
            c.observe(i < 100)  # marginal rate 0.25
    # pairwise joints co-violate often (0.20); the full 3-way joint never does
    for pair in ({"g0", "g1"}, {"g0", "g2"}, {"g1", "g2"}):
        jc = est.joint(frozenset(pair))
        for i in range(n):
            jc.observe(all_in_context=True, all_violated=i < 80)
    full = est.joint(frozenset({"g0", "g1", "g2"}))
    for _ in range(n):
        full.observe(all_in_context=True, all_violated=False)

    eta = 1e-3
    cert = certify(structure, est, eta=eta)
    pairwise_floor = min(est.pair_upper(eta).values())
    full_upper = full.upper(eta)
    assert cert.b_hw <= full_upper + 1e-12
    assert cert.b_hw < pairwise_floor  # the 3-way joint, not a pair, is binding
    # the full-cut counter is one estimated quantity in the 1 - N*eta accounting
    assert cert.n_estimated == 3 + 3 + 1


def test_default_registration_is_pairs_only():
    """Backward compatibility: without opt-in, only pairwise joints (plus
    singleton-cut pairs) are registered -- the committed runs' behavior."""
    structure = _three_parallel()
    default = _relevant_joints(structure, include_full_cuts=False)
    assert all(len(j) == 2 for j in default)
    opt_in = _relevant_joints(structure, include_full_cuts=True)
    assert frozenset({"g0", "g1", "g2"}) in opt_in
    assert default < opt_in


def test_runtime_opt_in_tracks_and_tightens_kway_joint():
    wl = redundancy_gadget(k=3, benign_rate=0.15, coupling=0.3)
    traces = wl.generate(3000, seed=7)

    rt_default = TPAGRuntime(
        wl.graph, wl.structure, wl.contracts, wl.assumption_checks, regime=wl.regime
    )
    rt_kway = TPAGRuntime(
        wl.graph,
        wl.structure,
        wl.contracts,
        wl.assumption_checks,
        regime=wl.regime,
        register_cut_joints=True,
    )
    rt_default.ingest_many(traces)
    rt_kway.ingest_many(traces)

    cut = next(c for c in wl.structure.cut_sets() if len(c) == 3)
    assert cut not in rt_default.estimation.joints
    assert rt_kway.estimation.joints[cut].m > 0

    b_default = rt_default.certificate(eta=1e-3).b_hw
    b_kway = rt_kway.certificate(eta=1e-3).b_hw
    obs = rt_kway.observed_violation_rate
    assert b_kway <= b_default + 1e-12  # k-way joint can only tighten
    assert b_kway + 1e-9 >= obs  # and stays a sound upper bound in-sample


def test_kway_bound_still_sound_under_comonotone_misses():
    """Adversarial dependence: one latent drives all three guards, so the 3-way
    co-violation equals the marginal. The k-way-tightened bound must not undershoot."""
    wl = redundancy_gadget(k=3, benign_rate=0.15, coupling=1.0)
    traces = wl.generate(4000, seed=11)
    rt = TPAGRuntime(
        wl.graph,
        wl.structure,
        wl.contracts,
        wl.assumption_checks,
        regime=wl.regime,
        register_cut_joints=True,
    )
    rt.ingest_many(traces)
    cert = rt.certificate(eta=1e-3)
    assert cert.b_hw + 1e-9 >= rt.observed_violation_rate


def test_estimation_joint_upper_includes_all_sizes():
    est = EstimationModel()
    est.joint(frozenset({"a", "b"})).observe(True, True)
    est.joint(frozenset({"a", "b", "c"})).observe(True, False)
    ju = est.joint_upper(0.01)
    assert frozenset({"a", "b"}) in ju
    assert frozenset({"a", "b", "c"}) in ju
    pu = est.pair_upper(0.01)
    assert frozenset({"a", "b", "c"}) not in pu
    assert np.isclose(ju[frozenset({"a", "b"})], pu[frozenset({"a", "b"})])

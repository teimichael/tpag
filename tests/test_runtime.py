"""Test runtime for the TPAG replication package."""

from tpag.bound import multiplicative_certificate
from tpag.runtime import TPAGRuntime
from tpag.sim import redundancy_gadget, t1_approval_workflow


def _run(workload, n=8000, seed=0, eta=0.01):
    traces = workload.generate(n, seed=seed)
    rt = TPAGRuntime(
        workload.graph,
        workload.structure,
        workload.contracts,
        workload.assumption_checks,
        regime=workload.regime,
    )
    rt.ingest_many(traces)
    return rt, rt.certificate(eta=eta)


def test_gadget_adequacy_holds():
    wl = redundancy_gadget(k=2, benign_rate=0.1, coupling=0.3)
    assert wl.adequacy().ok


def test_certificate_bounds_observed_violation():
    wl = redundancy_gadget(k=2, benign_rate=0.1, coupling=0.3)
    rt, cert = _run(wl)
    # TPAG certificate is a valid upper bound on the observed end-to-end rate
    assert cert.bound + 1e-9 >= rt.observed_violation_rate


def test_multiplicative_underestimates_under_correlation():
    wl = redundancy_gadget(k=2, benign_rate=0.1, coupling=0.3)
    rt, cert = _run(wl)
    eps_point = rt.estimation.eps_point()
    mult = multiplicative_certificate(wl.structure.cut_sets(), eps_point)
    # independence certificate is violated; TPAG holds
    assert rt.observed_violation_rate > mult
    assert cert.bound + 1e-9 >= rt.observed_violation_rate


def test_independent_control_c_zero_multiplicative_ok():
    wl = redundancy_gadget(k=2, benign_rate=0.1, coupling=0.0)
    rt, cert = _run(wl, n=20000)
    eps_point = rt.estimation.eps_point()
    mult = multiplicative_certificate(wl.structure.cut_sets(), eps_point)
    # with no shared mode, independence is (approximately) valid and TPAG still holds
    assert cert.bound + 1e-9 >= rt.observed_violation_rate
    assert rt.observed_violation_rate <= mult + 0.01


def test_t1_series_parallel_certificate_holds():
    wl = t1_approval_workflow(policy_benign=0.12, pii_benign=0.05, coupling=0.3)
    assert wl.adequacy().ok
    rt, cert = _run(wl, n=8000)
    assert cert.bound + 1e-9 >= rt.observed_violation_rate
    # the structure has two cuts -> two per-cut contributions
    assert len(cert.per_cut) == 2


def test_reproducibility_same_seed():
    wl = redundancy_gadget(k=2, benign_rate=0.1, coupling=0.3)
    rt1, _ = _run(wl, n=3000, seed=123)
    rt2, _ = _run(wl, n=3000, seed=123)
    assert rt1.observed_violation_rate == rt2.observed_violation_rate
    assert rt1.estimation.marginals["g1"].k == rt2.estimation.marginals["g1"].k
    assert rt1.estimation.marginals["g1"].m == rt2.estimation.marginals["g1"].m


def test_different_seed_changes_samples():
    wl = redundancy_gadget(k=2, benign_rate=0.1, coupling=0.3)
    rt1, _ = _run(wl, n=3000, seed=1)
    rt2, _ = _run(wl, n=3000, seed=2)
    # extremely unlikely to be identical
    assert rt1.estimation.marginals["g1"].k != rt2.estimation.marginals["g1"].k

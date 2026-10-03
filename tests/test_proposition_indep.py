"""Test proposition indep for the TPAG replication package."""

import numpy as np

from tpag.bound import compute_bound, multiplicative_certificate
from tpag.certificate import certify_exact
from tpag.structure import Atom, Parallel


def _gadget_distribution(eps: float, c: float, N: int, rng):
    """Two guards with marginal exactly ``eps`` and ``Pr[B1 ∩ B2] = c*eps``.

    Sampled from the 4-outcome joint over (B1, B2):
        Pr[11] = c*eps,  Pr[10] = Pr[01] = eps*(1-c),  Pr[00] = 1 - 2*eps + c*eps.
    Then both marginals equal ``eps`` and the co-violation probability is ``c*eps``,
    so the independence value ``eps^2`` is exceeded exactly when ``c > eps``.
    """
    p11 = c * eps
    p10 = eps * (1 - c)
    p01 = eps * (1 - c)
    p00 = 1 - 2 * eps + c * eps
    cat = rng.choice(4, size=N, p=[p11, p10, p01, p00])  # 0:11 1:10 2:01 3:00
    b1 = np.isin(cat, [0, 1]).astype(int)
    b2 = np.isin(cat, [0, 2]).astype(int)
    return np.stack([b1, b2], axis=1)


def test_multiplicative_violated_tpag_holds_at_c_above_eps():
    rng = np.random.default_rng(7)
    eps, c, N = 0.05, 0.5, 200000
    B = _gadget_distribution(eps, c, N, rng)
    structure = Parallel([Atom("g1"), Atom("g2")])
    cuts = structure.cut_sets()

    eps_hat = {"g1": float(B[:, 0].mean()), "g2": float(B[:, 1].mean())}
    pair = {frozenset({"g1", "g2"}): float((B[:, 0] & B[:, 1]).mean())}
    observed = float((B[:, 0] & B[:, 1]).mean())

    mult = multiplicative_certificate(cuts, eps_hat)  # ~ eps^2
    res = compute_bound(cuts, eps_hat, pair_upper=pair)  # TPAG

    assert mult < observed  # the independence certificate is VIOLATED
    assert observed <= res.b_hw + 1e-9  # TPAG bound HOLDS
    # the unsoundness factor is large
    assert observed / mult > 5.0


def test_crossover_sweep_c():
    """Across a sweep of c, the multiplicative certificate is violated exactly in
    the positively-correlated regime (c > eps), while TPAG holds throughout."""
    rng = np.random.default_rng(11)
    eps, N = 0.05, 120000
    structure = Parallel([Atom("g1"), Atom("g2")])
    cuts = structure.cut_sets()
    tpag_always_holds = True
    mult_violated_when_correlated = True
    for c in [0.0, 0.02, 0.05, 0.1, 0.3, 0.6, 0.9]:
        B = _gadget_distribution(eps, c, N, rng)
        eps_hat = {"g1": float(B[:, 0].mean()), "g2": float(B[:, 1].mean())}
        pair = {frozenset({"g1", "g2"}): float((B[:, 0] & B[:, 1]).mean())}
        observed = float((B[:, 0] & B[:, 1]).mean())
        mult = multiplicative_certificate(cuts, eps_hat)
        res = compute_bound(cuts, eps_hat, pair_upper=pair)
        if observed > res.b_hw + 1e-9:
            tpag_always_holds = False
        if c > eps and not (mult < observed):
            mult_violated_when_correlated = False
    assert tpag_always_holds
    assert mult_violated_when_correlated


def test_certificate_object_roundtrip():
    structure = Parallel([Atom("g1"), Atom("g2")])
    cert = certify_exact(
        structure, {"g1": 0.1, "g2": 0.1}, pair_joints={frozenset({"g1", "g2"}): 0.08}
    )
    assert cert.layer == "model"
    assert np.isclose(cert.b_hw, 0.08)
    assert "g1" in cert.dominant_cut()
    d = cert.to_dict()
    assert d["bound"] == cert.b_hw

"""Test bound soundness for the TPAG replication package."""

import numpy as np
import pytest

from tpag.bound import compute_bound, multiplicative_certificate
from tpag.structure import Atom, Parallel, Series, Structure

RNG = np.random.default_rng(20260607)
TOL = 1e-9


def _random_structure(names: list[str], rng) -> Structure:
    """Build a random series/parallel structure over the given component names."""

    def build(items: list[str]) -> Structure:
        if len(items) == 1:
            return Atom(items[0])
        # random split
        k = rng.integers(1, len(items))
        left, right = items[:k], items[k:]
        node_cls = Series if rng.random() < 0.5 else Parallel
        children = []
        for part in (left, right):
            children.append(build(part) if len(part) > 1 else Atom(part[0]))
        return node_cls(children)

    return build(names)


def _sample_violations(n: int, rng, N: int = 6000):
    """Sample N joint realizations of n violation indicators with random marginals
    and a random correlation structure (Gaussian copula)."""
    eps = rng.uniform(0.01, 0.4, size=n)
    # random PSD correlation matrix
    A = rng.normal(size=(n, n))
    cov = A @ A.T + np.eye(n) * 1e-3
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    z = rng.multivariate_normal(np.zeros(n), corr, size=N)
    from scipy.stats import norm

    thresh = norm.ppf(eps)
    B = (z < thresh).astype(np.int8)  # N x n, B[:,i] ~ Bernoulli(eps_i), correlated
    return B


@pytest.mark.slow
def test_bound_dominates_empirical_under_arbitrary_correlation():
    configs = 200
    failures = 0
    for _ in range(configs):
        n = int(RNG.integers(2, 6))
        names = [f"m{i}" for i in range(n)]
        structure = _random_structure(names, RNG)
        cuts = structure.cut_sets()
        B = _sample_violations(n, RNG)
        idx = {name: i for i, name in enumerate(names)}

        # exact empirical marginals and pairwise joints
        eps_hat = {name: float(B[:, idx[name]].mean()) for name in names}
        pair = {}
        for a in range(n):
            for b in range(a + 1, n):
                key = frozenset({names[a], names[b]})
                pair[key] = float((B[:, a] & B[:, b]).mean())

        # exact empirical cut-pair joints for Hunter--Worsley
        def cut_joint_lower(ca, cb, _B=B, _idx=idx):
            members = list(ca | cb)
            cols = [_idx[m] for m in members]
            ind = np.ones(_B.shape[0], dtype=bool)
            for c in cols:
                ind &= _B[:, c].astype(bool)
            return float(ind.mean())

        res = compute_bound(cuts, eps_hat, pair_upper=pair, cut_joint_lower=cut_joint_lower)

        # empirical Pr[not Phi]
        violated_any = np.zeros(B.shape[0], dtype=bool)
        for cut in cuts:
            ind = np.ones(B.shape[0], dtype=bool)
            for name in cut:
                ind &= B[:, idx[name]].astype(bool)
            violated_any |= ind
        observed = float(violated_any.mean())

        if observed > res.b_boole + TOL or observed > res.b_hw + TOL:
            failures += 1
        assert res.b_hw <= res.b_boole + TOL  # HW never looser than Boole

    assert failures == 0, f"{failures}/{configs} configs violated the bound (Thm. 1 broken)"


def test_bound_is_sound_where_multiplicative_is_not():
    """A focused redundancy case: under strong positive correlation the
    independence (multiplicative) certificate is violated while TPAG holds."""
    names = ["g1", "g2"]
    structure = Parallel([Atom("g1"), Atom("g2")])
    cuts = structure.cut_sets()
    rng = np.random.default_rng(1)
    eps = 0.1
    # comonotone: a single shared latent drives both -> Pr[B1 & B2] ~ eps
    u = rng.random(40000)
    B = np.stack([(u < eps), (u < eps)], axis=1).astype(np.int8)
    idx = {"g1": 0, "g2": 1}
    eps_hat = {n: float(B[:, idx[n]].mean()) for n in names}
    pair = {frozenset({"g1", "g2"}): float((B[:, 0] & B[:, 1]).mean())}
    res = compute_bound(cuts, eps_hat, pair_upper=pair)
    observed = float((B[:, 0] & B[:, 1]).mean())
    mult = multiplicative_certificate(cuts, eps_hat)

    assert observed > mult  # multiplicative certificate is VIOLATED
    assert observed <= res.b_hw + TOL  # TPAG bound HOLDS

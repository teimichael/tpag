"""Estimation for the TPAG replication package."""

from __future__ import annotations

from dataclasses import dataclass, field

from scipy.stats import beta


def clopper_pearson_upper(k: int, m: int, eta: float) -> float:
    """One-sided upper confidence limit for a binomial rate at level ``1 - eta``.

    ``k`` successes (violations) in ``m`` trials. Returns ``p_U`` such that
    ``Pr[p > p_U] <= eta``. With no data (``m == 0``) the only sound upper bound
    is 1; with ``k == m`` it is also 1.
    """
    if m == 0 or k >= m:
        return 1.0
    if k < 0:
        raise ValueError("k must be >= 0")
    return float(beta.ppf(1.0 - eta, k + 1, m - k))


def clopper_pearson_lower(k: int, m: int, eta: float) -> float:
    """One-sided lower confidence limit for a binomial rate at level ``1 - eta``.

    Returns ``p_L`` such that ``Pr[p < p_L] <= eta``. With ``k == 0`` (or no data)
    the only sound lower bound is 0.
    """
    if m == 0 or k <= 0:
        return 0.0
    if k > m:
        raise ValueError("k must be <= m")
    return float(beta.ppf(eta, k, m - k + 1))


@dataclass
class BernoulliCounter:
    """Online counter of in-context activations and guarantee violations for one
    component (Def. 3 ``epsilon_i``)."""

    m: int = 0  # in-context activations observed
    k: int = 0  # of which the guarantee was violated (B_i occurred)

    def observe(self, violated: bool) -> None:
        self.m += 1
        if violated:
            self.k += 1

    @property
    def point(self) -> float:
        return self.k / self.m if self.m else float("nan")

    def upper(self, eta: float) -> float:
        return clopper_pearson_upper(self.k, self.m, eta)

    def lower(self, eta: float) -> float:
        return clopper_pearson_lower(self.k, self.m, eta)


@dataclass
class JointCounter:
    """Online counter of co-observed activations and co-violations for a *set* of
    components (used for measured pairwise/higher-order joints ``kappa``)."""

    m: int = 0  # activations where all members were co-observed in-context
    k: int = 0  # of which all members violated simultaneously

    def observe(self, all_in_context: bool, all_violated: bool) -> None:
        if not all_in_context:
            return
        self.m += 1
        if all_violated:
            self.k += 1

    @property
    def point(self) -> float:
        return self.k / self.m if self.m else float("nan")

    def upper(self, eta: float) -> float:
        return clopper_pearson_upper(self.k, self.m, eta)

    def lower(self, eta: float) -> float:
        return clopper_pearson_lower(self.k, self.m, eta)


@dataclass
class EstimationModel:
    """Aggregates per-component and per-group counters and emits the confidence-
    limited inputs the bound engine consumes, with the ``1 - N*eta`` accounting."""

    marginals: dict[str, BernoulliCounter] = field(default_factory=dict)
    joints: dict[frozenset[str], JointCounter] = field(default_factory=dict)

    def marginal(self, node: str) -> BernoulliCounter:
        return self.marginals.setdefault(node, BernoulliCounter())

    def joint(self, members: frozenset[str]) -> JointCounter:
        return self.joints.setdefault(members, JointCounter())

    # -- emit bound-engine inputs -------------------------------------------
    def eps_upper(self, eta: float) -> dict[str, float]:
        return {n: c.upper(eta) for n, c in self.marginals.items()}

    def eps_point(self) -> dict[str, float]:
        return {n: c.point for n, c in self.marginals.items()}

    def pair_upper(self, eta: float) -> dict[frozenset[str], float]:
        return {g: c.upper(eta) for g, c in self.joints.items() if len(g) == 2}

    def joint_upper(self, eta: float) -> dict[frozenset[str], float]:
        """Upper limits for *all* registered joints, any size. A measured upper
        bound on ``Pr[∩_{i in g} B_i]`` tightens any cut containing ``g`` (for
        ``|g| == 2``) or equal to ``g`` (the full-cut key ``bound._cut_upper``
        consults for ``k >= 3``)."""
        return {g: c.upper(eta) for g, c in self.joints.items()}

    def pair_lower(self, eta: float) -> dict[frozenset[str], float]:
        return {g: c.lower(eta) for g, c in self.joints.items() if len(g) == 2}

    def pair_point(self) -> dict[frozenset[str], float]:
        """Point estimates of the pairwise joints (the *charitable*, NOT
        confidence-limited inputs -- used by the RQ-E6 estimation-layer ablation to
        contrast point estimates against the sound Clopper--Pearson limits)."""
        return {g: c.point for g, c in self.joints.items() if len(g) == 2 and c.m}

    def num_estimated_quantities(self) -> int:
        """``N`` in the ``1 - N*eta`` union accounting: the number of estimated
        marginals and joints that feed the certificate."""
        return len(self.marginals) + len(self.joints)

    def overall_confidence(self, eta: float) -> float:
        return max(0.0, 1.0 - self.num_estimated_quantities() * eta)

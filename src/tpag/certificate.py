"""Certificate for the TPAG replication package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bound import BoundResult, compute_bound
from .estimation import EstimationModel
from .structure import Structure


@dataclass
class Certificate:
    b_boole: float
    b_hw: float
    confidence: float
    eta: float
    n_estimated: int
    per_cut: list[tuple[frozenset[str], float]]
    delta_hw: float
    regime: str
    layer: str  # "model" | "estimation"
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def bound(self) -> float:
        """The tightest certified upper bound on Pr[not Phi]."""
        return self.b_hw

    def dominant_cut(self) -> frozenset[str]:
        """The cut contributing most to the bound (localization hint)."""
        if not self.per_cut:
            return frozenset()
        return max(self.per_cut, key=lambda cu: cu[1])[0]

    def summary(self) -> str:
        lines = [
            f"TPAG certificate ({self.layer} layer)",
            f"  Pr[not Phi] <= {self.b_hw:.3e}  (Hunter-Worsley)",
            f"               <= {self.b_boole:.3e}  (Boole)",
            f"  confidence  >= {self.confidence:.4f}  "
            f"(eta={self.eta}, N={self.n_estimated} estimated quantities)",
            f"  regime: {self.regime};  Delta_HW={self.delta_hw:.3e}",
            "  per-cut upper bounds:",
        ]
        for cut, u in sorted(self.per_cut, key=lambda cu: -cu[1]):
            lines.append(f"    {{{', '.join(sorted(cut))}}}: {u:.3e}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "b_boole": self.b_boole,
            "b_hw": self.b_hw,
            "bound": self.bound,
            "confidence": self.confidence,
            "eta": self.eta,
            "n_estimated": self.n_estimated,
            "delta_hw": self.delta_hw,
            "regime": self.regime,
            "per_cut": [{"cut": sorted(c), "u": u} for c, u in self.per_cut],
            "dominant_cut": sorted(self.dominant_cut()),
            "provenance": self.provenance,
        }


def certify_exact(
    structure: Structure,
    eps: dict[str, float],
    pair_joints: dict[frozenset[str], float] | None = None,
    regime: str = "general",
    provenance: dict[str, Any] | None = None,
) -> Certificate:
    """Model-layer certificate (Thm. 1). ``eps`` and ``pair_joints`` are exact (or
    assumed) probabilities; ``pair_joints`` are used both to tighten cuts (as upper
    bounds) and as the Hunter--Worsley joints (as lower bounds), since exact values
    are simultaneously valid upper and lower bounds."""
    cuts = structure.cut_sets()

    def cut_joint_lower(a: frozenset[str], b: frozenset[str]) -> float:
        if pair_joints is None:
            return 0.0
        union = a | b
        if len(union) == 2:
            return pair_joints.get(union, 0.0)
        return 0.0

    res: BoundResult = compute_bound(
        cuts, eps, pair_upper=pair_joints, cut_joint_lower=cut_joint_lower, regime=regime
    )
    return Certificate(
        b_boole=res.b_boole,
        b_hw=res.b_hw,
        confidence=1.0,
        eta=0.0,
        n_estimated=0,
        per_cut=res.per_cut,
        delta_hw=res.delta_hw,
        regime=res.regime,
        layer="model",
        provenance=provenance or {},
    )


def certify(
    structure: Structure,
    estimation: EstimationModel,
    eta: float = 0.01,
    regime: str = "general",
    provenance: dict[str, Any] | None = None,
) -> Certificate:
    """Estimation-layer certificate (Thm. 2): holds with confidence ``1 - N*eta``."""
    cuts = structure.cut_sets()
    eps_upper = estimation.eps_upper(eta)
    # all registered joint upper limits (pairwise AND full-cut k>=3 keys); the
    # bound engine consults pairs within a cut and the exact full-cut key.
    joint_upper = estimation.joint_upper(eta)
    pair_lower = estimation.pair_lower(eta)

    def cut_joint_lower(a: frozenset[str], b: frozenset[str]) -> float:
        union = a | b
        if len(union) == 2:
            return pair_lower.get(union, 0.0)
        # higher-order cut joints: use a registered joint counter if present, else 0
        jc = estimation.joints.get(union)
        return jc.lower(eta) if jc is not None else 0.0

    res = compute_bound(
        cuts, eps_upper, pair_upper=joint_upper, cut_joint_lower=cut_joint_lower, regime=regime
    )
    prov = dict(provenance or {})
    prov["estimation"] = {
        n: {"m": c.m, "k": c.k, "point": c.point, "upper": c.upper(eta)}
        for n, c in estimation.marginals.items()
    }
    return Certificate(
        b_boole=res.b_boole,
        b_hw=res.b_hw,
        confidence=estimation.overall_confidence(eta),
        eta=eta,
        n_estimated=estimation.num_estimated_quantities(),
        per_cut=res.per_cut,
        delta_hw=res.delta_hw,
        regime=res.regime,
        layer="estimation",
        provenance=prov,
    )

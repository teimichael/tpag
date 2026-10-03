"""Comparison for the TPAG replication package."""

from __future__ import annotations

import numpy as np

from ..bound import multiplicative_certificate
from ..runtime import TPAGRuntime
from . import baselines

_TOL = 1e-9

# Display order + human-readable labels for the rules we score.
RULE_LABELS = {
    "independence": r"independence $\prod\varepsilon_i$",
    "boole": "Boole (union)",
    "frechet_joints": "Fréchet + joints",
    "tpag_hw": "TPAG (Hunter--Worsley)",
}
SOUND_RULES = ["boole", "frechet_joints", "tpag_hw"]  # certified upper bounds


def _aggregate(rows: list[dict], key: str, applicable: list[dict]) -> dict:
    """Coverage / unsound count / mean tightness for one rule over its applicable
    configs (a rule is *sound* on a config iff its bound >= observed)."""
    vals = [r for r in applicable if r[key] is not None]
    sound = [r for r in vals if r[key] + _TOL >= r["observed"]]
    n = len(vals)
    return {
        "n_configs": n,
        "coverage": (len(sound) / n) if n else float("nan"),
        "unsound": n - len(sound),
        # tightness = mean (bound - observed) among configs where the rule is sound
        "mean_slack_sound": (
            float(np.mean([r[key] - r["observed"] for r in sound])) if sound else float("nan")
        ),
    }


def score_composition_rules(
    configs: list[tuple[str, object]], *, N: int, seed: int, eta: float = 0.01
) -> tuple[list[dict], dict]:
    """Score every composition rule over an *arbitrary* config list (reused by RQ6
    and by the extended RQ-E4 over the expanded topology suite). Returns
    ``(per_config_rows, per_rule_aggregate)`` and writes nothing -- the caller owns
    persistence. Identical scoring to :func:`run_baseline_comparison`."""
    rows: list[dict] = []
    for i, (kind, wl) in enumerate(configs):
        rt = TPAGRuntime(
            wl.graph, wl.structure, wl.contracts, wl.assumption_checks, regime=wl.regime
        )
        rt.ingest_many(wl.generate(N, seed=seed + i))
        obs = rt.observed_violation_rate
        cuts = wl.structure.cut_sets()
        eps_point = rt.estimation.eps_point()
        eps_upper = rt.estimation.eps_upper(eta)
        pair_upper = rt.estimation.pair_upper(eta)

        cert = rt.certificate(eta=eta)
        boole = baselines.boole_bound(wl.structure, eps_upper)
        frechet_joints = baselines.frechet_joints_bound(wl.structure, eps_upper, pair_upper)
        independence = multiplicative_certificate(cuts, eps_point)

        rows.append(
            {
                "config": wl.name,
                "kind": kind,
                "regime": cert.regime,
                "episodes": rt.episodes,
                "observed": obs,
                "independence": independence,
                "boole": boole,
                "frechet_joints": frechet_joints,
                "tpag_hw": cert.b_hw,
                "independence_violated": bool(obs > independence + _TOL),
                "independence_ratio": (obs / independence) if independence > _TOL else None,
                "n_cuts": len(cuts),
            }
        )

    per_rule = {
        "independence": _aggregate(rows, "independence", rows),
        "boole": _aggregate(rows, "boole", rows),
        "frechet_joints": _aggregate(rows, "frechet_joints", rows),
        "tpag_hw": _aggregate(rows, "tpag_hw", rows),
    }
    return rows, per_rule

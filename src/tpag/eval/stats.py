"""Stats for the TPAG replication package."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def bootstrap_ci(
    samples: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for the mean of a 0/1 (or real) sample via bootstrap."""
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    means = np.empty(n_boot)
    n = samples.size
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = samples[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (float(samples.mean()), lo, hi)


@dataclass
class CalibrationSummary:
    n_configs: int
    coverage: float  # fraction of configs with observed <= bound
    mean_slack: float  # mean (bound - observed) over configs
    mean_hw_improvement: float  # mean (B_Boole - B_HW)
    violations: int  # configs where observed > bound (should be 0)


def calibration_summary(
    observed: list[float], b_boole: list[float], b_hw: list[float], tol: float = 1e-9
) -> CalibrationSummary:
    obs = np.asarray(observed)
    bb = np.asarray(b_boole)
    bh = np.asarray(b_hw)
    covered = obs <= bh + tol
    return CalibrationSummary(
        n_configs=len(obs),
        coverage=float(covered.mean()) if len(obs) else float("nan"),
        mean_slack=float(np.mean(bh - obs)) if len(obs) else float("nan"),
        mean_hw_improvement=float(np.mean(bb - bh)) if len(obs) else float("nan"),
        violations=int((obs > bh + tol).sum()),
    )

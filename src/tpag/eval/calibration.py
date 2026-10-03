"""Calibration for the TPAG replication package."""

from __future__ import annotations
from dataclasses import dataclass
from ..runtime import TPAGRuntime

__all__ = ["holdout_coverage"]


@dataclass
class HoldoutResult:
    config: str
    bound_train: float
    observed_train: float
    observed_test: float
    covered_out_of_sample: bool
    n_train: int
    n_test: int


def holdout_coverage(
    workload, n: int, seed: int, eta: float = 0.01, holdout_fraction: float = 0.5
) -> HoldoutResult:
    """Estimate the certificate on a training split and validate coverage on a
    disjoint test split (out-of-sample soundness)."""
    traces = workload.generate(n, seed=seed)
    n_test = max(1, int(len(traces) * holdout_fraction))
    test, train = (traces[:n_test], traces[n_test:])
    rt_train = TPAGRuntime(
        workload.graph,
        workload.structure,
        workload.contracts,
        workload.assumption_checks,
        regime=workload.regime,
    )
    rt_train.ingest_many(train)
    cert = rt_train.certificate(eta=eta)
    rt_test = TPAGRuntime(
        workload.graph,
        workload.structure,
        workload.contracts,
        workload.assumption_checks,
        regime=workload.regime,
    )
    rt_test.ingest_many(test)
    obs_test = rt_test.observed_violation_rate
    return HoldoutResult(
        config=workload.name,
        bound_train=cert.bound,
        observed_train=rt_train.observed_violation_rate,
        observed_test=obs_test,
        covered_out_of_sample=bool(cert.bound + 1e-09 >= obs_test),
        n_train=len(train),
        n_test=len(test),
    )

"""Rq e1 calibration for the TPAG replication package."""

from __future__ import annotations

import numpy as np

from ...runtime import TPAGRuntime
from ...sim.domains.code_review import code_review_workflow
from ...sim.topology_gen import soundness_suite
from .. import calibration, stats
from ..config import ExpConfig
from . import finalize, jsonl_appender, make_recorder

RQ = "RQ-E1"


def _calibration_configs(cfg: ExpConfig) -> list[tuple[str, object]]:
    """The manuscript's 19 configs + a topology-diverse extension + the second
    (code-review) domain, so calibration is measured broadly."""
    from ..suites import calibration_configs as base_configs

    configs = list(base_configs())  # the manuscript's 19 (financial domain)
    # topology-diverse additions at the smaller sizes (kept light for E1)
    small_sizes = tuple(n for n in cfg.tier.topology_sizes if n <= 16)
    for fam, wl in soundness_suite(small_sizes, seed=cfg.seed):
        configs.append((fam, wl))
    # second domain
    for c in (0.0, 0.2, 0.4):
        configs.append(("code_review", code_review_workflow(coupling=c)))
    return configs


def run(cfg: ExpConfig, use_llm: bool = True, *, argv=None, force: bool = False) -> dict:
    rec = make_recorder(cfg, RQ, "rq_e1_calibration", argv=argv, force=force)
    t = cfg.tier
    configs = _calibration_configs(cfg)
    eta0 = t.eta_grid[0]
    append_row = jsonl_appender(rec, "per_config.jsonl")  # stream as computed

    rows: list[dict] = []
    for i, (kind, wl) in enumerate(configs):
        traces = wl.generate(t.episodes, seed=cfg.seed + i)
        rt = TPAGRuntime(
            wl.graph, wl.structure, wl.contracts, wl.assumption_checks, regime=wl.regime
        )
        outcomes = np.array([rt.ingest(tr).system_failed for tr in traces], dtype=float)
        cert = rt.certificate(eta=eta0)
        obs = float(outcomes.mean())
        mean, lo, hi = stats.bootstrap_ci(outcomes, n_boot=t.boot_resamples, seed=cfg.seed + i)
        row = {
            "config": wl.name,
            "kind": kind,
            "regime": cert.regime,
            "episodes": rt.episodes,
            "observed": obs,
            "obs_ci_lo": lo,
            "obs_ci_hi": hi,
            "b_boole": cert.b_boole,
            "b_hw": cert.b_hw,
            "confidence": cert.confidence,
            "sound": bool(cert.bound + 1e-9 >= obs),
            "slack": cert.bound - obs,
        }
        rows.append(row)
        append_row(row)

    observed = [r["observed"] for r in rows]
    b_hw = [r["b_hw"] for r in rows]
    b_boole = [r["b_boole"] for r in rows]
    cal = stats.calibration_summary(observed, b_boole, b_hw)

    # -- eta sweep (confidence vs tightness) --
    eta_rows = []
    for eta in t.eta_grid:
        slacks, covered = [], 0
        for i, (_kind, wl) in enumerate(configs):
            rt = TPAGRuntime(
                wl.graph, wl.structure, wl.contracts, wl.assumption_checks, regime=wl.regime
            )
            outc = np.array(
                [rt.ingest(tr).system_failed for tr in wl.generate(t.episodes, seed=cfg.seed + i)],
                dtype=float,
            )
            cert = rt.certificate(eta=eta)
            o = float(outc.mean())
            slacks.append(cert.bound - o)
            covered += int(cert.bound + 1e-9 >= o)
        eta_rows.append(
            {
                "eta": eta,
                "mean_slack": float(np.mean(slacks)),
                "coverage": covered / len(configs),
            }
        )

    # -- held-out (out-of-sample) coverage on a representative subset --
    holdout = []
    for i, (_kind, wl) in enumerate(configs[:: max(1, len(configs) // 12)]):
        hr = calibration.holdout_coverage(
            wl,
            n=t.episodes,
            seed=cfg.seed + 7919 * i,
            eta=eta0,
            holdout_fraction=t.holdout_fraction,
        )
        holdout.append(hr.__dict__)
    holdout_rate = (
        float(np.mean([h["covered_out_of_sample"] for h in holdout])) if holdout else None
    )

    metrics = {
        "n_configs": len(rows),
        "coverage": cal.coverage,
        "violations": cal.violations,
        "mean_slack": cal.mean_slack,
        "mean_hw_improvement": cal.mean_hw_improvement,
        "mean_obs_ci_width": float(np.mean([r["obs_ci_hi"] - r["obs_ci_lo"] for r in rows])),
        "eta_sweep": eta_rows,
        "holdout_out_of_sample_coverage": holdout_rate,
        "episodes_per_config": t.episodes,
    }
    rec.write_csv("per_config.csv", rows)
    rec.write_json("eta_sweep.json", eta_rows)
    rec.write_json("holdout.json", holdout)
    return finalize(rec, cfg, RQ, "computed", metrics, provenance={"per_config": "per_config.csv"})

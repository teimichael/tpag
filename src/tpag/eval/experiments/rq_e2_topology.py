"""Rq e2 topology for the TPAG replication package."""

from __future__ import annotations

import numpy as np

from ...bound import multiplicative_certificate
from ...runtime import TPAGRuntime
from ...sim.topology_gen import soundness_suite
from .. import scaling
from ..config import ExpConfig
from . import finalize, jsonl_appender, make_recorder

RQ = "RQ-E2"


def run(cfg: ExpConfig, use_llm: bool = True, *, argv=None, force: bool = False) -> dict:
    rec = make_recorder(cfg, RQ, "rq_e2_topology", argv=argv, force=force)
    t = cfg.tier
    append_row = jsonl_appender(rec, "per_config.jsonl")  # stream as computed

    # ---- part 1: soundness / tightness across families and sizes ----
    rows: list[dict] = []
    for i, (family, wl) in enumerate(soundness_suite(t.topology_sizes, seed=cfg.seed)):
        # bound the per-config episode budget so large-N configs stay cheap
        n_ep = max(2000, t.mc_samples // max(1, len(wl.guard_names) // 8))
        rt = TPAGRuntime(
            wl.graph, wl.structure, wl.contracts, wl.assumption_checks, regime=wl.regime
        )
        outcomes = np.array(
            [rt.ingest(tr).system_failed for tr in wl.generate(n_ep, seed=cfg.seed + i)],
            dtype=float,
        )
        cert = rt.certificate(eta=t.eta_grid[0])
        obs = float(outcomes.mean())
        indep = multiplicative_certificate(wl.structure.cut_sets(), rt.estimation.eps_point())
        row = {
            "family": family,
            "config": wl.name,
            "n_components": len(wl.guard_names),
            "n_cuts": len(wl.structure.cut_sets()),
            "episodes": rt.episodes,
            "observed": obs,
            "b_boole": cert.b_boole,
            "b_hw": cert.b_hw,
            "hw_tightening": cert.b_boole - cert.b_hw,
            "slack": cert.b_hw - obs,
            "sound": bool(cert.b_hw + 1e-9 >= obs),
            "independence": indep,
            "independence_violated": bool(obs > indep + 1e-9),
        }
        rows.append(row)
        append_row(row)

    families = sorted({r["family"] for r in rows})
    by_family = {
        f: {
            "n_configs": sum(r["family"] == f for r in rows),
            "coverage": float(np.mean([r["sound"] for r in rows if r["family"] == f])),
            "mean_slack": float(np.mean([r["slack"] for r in rows if r["family"] == f])),
            "max_n": max(r["n_components"] for r in rows if r["family"] == f),
            "independence_violated": int(
                sum(r["independence_violated"] for r in rows if r["family"] == f)
            ),
        }
        for f in families
    }

    # ---- part 2: scalability (timing + blow-up boundary) ----
    scaling_rows = scaling.run_scaling_curve(
        t.scaling_sizes,
        max_cuts=t.max_cuts,
        timeout_s=t.scaling_timeout_s,
    )

    feasible = {
        f: max(
            [r["n"] for r in scaling_rows if r["family"] == f and r["status"] == "computed"],
            default=None,
        )
        for f in {r["family"] for r in scaling_rows}
    }
    metrics = {
        "n_configs": len(rows),
        "overall_coverage": float(np.mean([r["sound"] for r in rows])),
        "violations": int(sum(not r["sound"] for r in rows)),
        "mean_slack": float(np.mean([r["slack"] for r in rows])),
        "mean_hw_tightening": float(np.mean([r["hw_tightening"] for r in rows])),
        "independence_violated_configs": int(sum(r["independence_violated"] for r in rows)),
        "by_family": by_family,
        "max_feasible_n_by_family": feasible,
        "scaling_points": len(scaling_rows),
        "scaling_skipped_cap": int(sum(r["status"] == "skipped_cap" for r in scaling_rows)),
    }
    rec.write_csv("per_config.csv", rows)
    rec.write_csv("scaling.csv", scaling_rows)
    return finalize(
        rec,
        cfg,
        RQ,
        "computed",
        metrics,
        provenance={"per_config": "per_config.csv", "scaling": "scaling.csv"},
    )

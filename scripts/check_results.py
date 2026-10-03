"""Check the headline manuscript results after offline derivation."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    stats = json.loads((ROOT / "outputs/derived/derived_stats.json").read_text())
    for suite, directory, count, calls, strict, robust in (
        ("main", "results", 108, 29760, 47, 34),
        ("tighten", "results_tighten", 92, 176400, 57, 57),
    ):
        points = json.loads(
            (ROOT / directory / "rq_e3_llm_correlation/llm_points.json").read_text()
        )
        require(len(points) == count, f"{suite}: operating-point count")
        require(
            sum(p["independence_violated"] for p in points) == strict,
            f"{suite}: independence violation count",
        )
        require(
            sum(p["independence_violated_ci"] for p in points) == robust,
            f"{suite}: CI-robust violation count",
        )
        parsed = stats["llm_parse"][suite]
        require(
            parsed["calls"] == parsed["json_decisions"] == parsed["decision_matches_miss"] == calls,
            f"{suite}: raw decisions/counts",
        )
    require(
        stats["llm_heldout_total"] == {"splits": 348, "covered": 345, "below_test_ci95_lo": 0},
        "held-out stream counts",
    )
    composed = stats["llm_composition"]["all"]
    require(
        composed["tests"] == composed["tpag_hw"]["covered"] == 720, "composed workflow coverage"
    )
    require(
        math.isclose(
            composed["mean_excess_tpag_over_direct"], 0.0836284830280345, rel_tol=0, abs_tol=1e-12
        ),
        "composition cost",
    )
    require(
        stats["e1"]["n_configs"] == 52 and stats["e2"]["n_configs"] == 42,
        "scripted configuration counts",
    )
    require(
        stats["e4"]["n_configs"] == stats["e6"]["n_configs"] == 36, "ablation configuration counts"
    )
    require(
        stats["e7"]["llm_pairs"] == 72 and stats["e7"]["llm_violated"] == 56,
        "injection-shift counts",
    )
    require(stats["e5"]["integration"]["live_episodes"] == 24, "LangGraph integration count")
    # Exercise the real adapter without model calls by replaying recorded decisions.
    from tpag.adapters.langgraph import build_review_app, run_episode_trace
    from tpag.runtime import TPAGRuntime
    from tpag.sim.domains.code_review import code_review_workflow

    workload = code_review_workflow(coupling=0.3)
    runtime = TPAGRuntime(
        workload.graph,
        workload.structure,
        workload.contracts,
        workload.assumption_checks,
        regime=workload.regime,
    )
    with (ROOT / "results/rq_e5_overhead_integration/langgraph_episodes.jsonl").open() as stream:
        episodes = [json.loads(line) for line in stream]
    for episode in episodes:
        guards = []
        for name in ("security", "license", "ci_policy"):
            decision = SimpleNamespace(
                approved=episode["misses"][name], latency_s=episode["latencies"][name]
            )
            guards.append(SimpleNamespace(name=name, decide=lambda text, seed, d=decision: d))
        app = build_review_app(guards, seed=20260607)
        trace, replayed = run_episode_trace(app, episode["request"], actor="merge_actor")
        require(replayed["misses"] == episode["misses"], "LangGraph decision replay")
        runtime.ingest(trace)
    certificate = runtime.certificate(eta=1e-3)
    require(
        len(episodes) == 24 and runtime.observed_violation_rate == 0.875,
        "LangGraph observed violation rate",
    )
    require(
        certificate.bound == 1.0 and certificate.confidence == 0.996, "LangGraph certificate replay"
    )
    print(
        "Verified 206,160 raw decisions; independence 47/34 and 57/57; "
        "345/348 held-out splits; 720/720 composed tests."
    )


if __name__ == "__main__":
    main()

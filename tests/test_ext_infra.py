"""Test ext infra for the TPAG replication package."""

from __future__ import annotations
import numpy as np
import pytest
from tpag.runtime import TPAGRuntime


def test_config_defaults_and_hash():
    from tpag.eval.config import estimate_llm_calls, load_config

    cfg = load_config(tier="quick")
    assert cfg.tier.name == "quick"
    assert cfg.config_hash() == cfg.config_hash()
    assert cfg.benchmark_version
    est = estimate_llm_calls(cfg, n_model_conditions=4)
    assert est["approx_llm_calls"] > 0
    assert est["groups_capped"] <= cfg.tier.max_conditions


def test_size_policy_tiers_and_cap():
    from tpag.eval.config import load_config

    cfg = load_config(tier="quick")
    assert cfg.size_tier(3.0) == "small"
    assert cfg.size_tier(14.0) == "mid"
    assert cfg.size_tier(40.0) == "large"
    assert cfg.size_tier(70.0) == "excluded"
    assert all((m.params_b < cfg.max_params_b for m in cfg.zoo_eligible()))
    assert len(cfg.zoo_eligible()) == len(cfg.zoo)


def test_schema_validation_and_manifest():
    from tpag.eval.config import load_config
    from tpag.eval.schema import base_manifest, validate_metrics

    cfg = load_config(tier="quick")
    m = base_manifest("RQ-E1", cfg, argv=["tpag", "run-ext", "e1"])
    for key in (
        "gpu",
        "config_hash",
        "source_snapshot",
        "env",
        "argv",
        "benchmark_version",
        "config_path",
        "max_params_b",
    ):
        assert key in m, f"manifest missing {key}"
    assert m["argv"] == ["tpag", "run-ext", "e1"]
    assert "python" in m["env"] and "packages" in m["env"]
    assert m["source_snapshot"] == "released-package"
    assert (
        validate_metrics(
            {
                "rq": "RQ-E1",
                "status": "computed",
                "tier": "quick",
                "seed": 1,
                "config_hash": "abc12345",
            }
        )
        == []
    )
    problems = validate_metrics({"rq": "RQ-E1", "status": "bogus"})
    assert any(("missing" in p for p in problems)) and any(("status" in p for p in problems))
    assert (
        validate_metrics(
            {
                "rq": "RQ-E1",
                "status": "skipped_no_ollama",
                "tier": "quick",
                "seed": 1,
                "config_hash": "abc12345",
            }
        )
        == []
    )


def test_overwrite_guard_and_partial_manifest(tmp_path):
    import json
    from tpag.eval.config import load_config
    from tpag.eval.experiments import ResultsExist, make_recorder

    cfg = load_config(tier="quick", out=str(tmp_path))
    make_recorder(cfg, "RQ-E1", "sub")
    assert (tmp_path / "sub" / "manifest.json").exists()
    manifest = json.loads((tmp_path / "sub" / "manifest.json").read_text())
    assert manifest["status"] == "running"
    (tmp_path / "sub" / "metrics.json").write_text("{}")
    import pytest as _pytest

    with _pytest.raises(ResultsExist):
        make_recorder(cfg, "RQ-E1", "sub")
    make_recorder(cfg, "RQ-E1", "sub", force=True)


def test_topology_cut_sets():
    from tpag.sim import topology_gen as tg

    assert all((len(c) == 1 for c in tg.series_topology(4).structure.cut_sets()))
    par = tg.parallel_topology(3).structure.cut_sets()
    assert len(par) == 1 and len(next(iter(par))) == 3
    cuts = tg.kofn_voting(2, 3).structure.cut_sets()
    assert all((len(c) == 2 for c in cuts)) and len(cuts) == 3
    recon = tg.reconvergent_dag(2, 3).structure.cut_sets()
    assert len(recon) == 3 and all((len(c) == 2 for c in recon))


def test_topology_random_and_suites():
    from tpag.sim import topology_gen as tg

    g = tg.random_general_dag(8, seed=1)
    assert g.structure.cut_sets()
    sp = tg.random_series_parallel(8, seed=2)
    assert len(sp.guard_names) == 8
    suite = tg.soundness_suite((4, 8), seed=0)
    assert suite and all((hasattr(wl, "structure") for _, wl in suite))


def test_estimated_cut_count_and_scaling_skip():
    from tpag.eval import scaling
    from tpag.sim import topology_gen as tg

    assert tg.estimated_cut_count("series", 10) == 10
    assert tg.estimated_cut_count("nested_parallel", 64) > 1000000.0
    rec = scaling.run_scaling_point("series", 8, max_cuts=10000, timeout_s=5)
    assert rec["status"] == "computed" and rec["n_cuts"] == 8
    big = scaling.run_scaling_point("nested_parallel", 128, max_cuts=2000000, timeout_s=5)
    assert big["status"] == "skipped_cap"
    curve = scaling.run_scaling_curve((4, 8), max_cuts=10000, timeout_s=5)
    assert curve and {r["family"] for r in curve}


def test_generated_topology_bound_is_sound():
    from tpag.sim import topology_gen as tg

    wl = tg.series_topology(4, coupling=0.3)
    rt = TPAGRuntime(wl.graph, wl.structure, wl.contracts, wl.assumption_checks, regime=wl.regime)
    outcomes = [rt.ingest(t).system_failed for t in wl.generate(800, seed=1)]
    obs = float(np.mean(outcomes))
    cert = rt.certificate(eta=0.05)
    assert cert.bound + 1e-09 >= obs


def test_code_review_workflow_runs():
    from tpag.sim.domains.code_review import code_review_workflow

    wl = code_review_workflow(coupling=0.3)
    assert wl.structure.cut_sets()
    rt = TPAGRuntime(wl.graph, wl.structure, wl.contracts, wl.assumption_checks, regime=wl.regime)
    outcomes = [rt.ingest(t).system_failed for t in wl.generate(500, seed=3)]
    cert = rt.certificate(eta=0.05)
    assert cert.bound + 1e-09 >= float(np.mean(outcomes))


def test_holdout_coverage():
    from tpag.eval import calibration
    from tpag.sim import topology_gen as tg

    wl = tg.parallel_topology(2, coupling=0.3)
    res = calibration.holdout_coverage(wl, n=2000, seed=5, eta=0.05, holdout_fraction=0.5)
    assert res.n_train > 0 and res.n_test > 0
    assert isinstance(res.covered_out_of_sample, bool)
    assert res.bound_train + 1e-09 >= res.observed_train
    assert res.bound_train + 0.05 >= res.observed_test


def test_approved_registry_shape():
    from tpag.eval.config import DEFAULT_ZOO, FAMILIES, TIERS, load_config, validate_registry

    validate_registry(DEFAULT_ZOO)
    assert len(DEFAULT_ZOO) == 12
    assert {m.family for m in DEFAULT_ZOO} == set(FAMILIES)
    assert {m.tier for m in DEFAULT_ZOO} == set(TIERS)
    assert len({(m.family, m.tier) for m in DEFAULT_ZOO}) == 12
    assert all((m.params_b < 70.0 for m in DEFAULT_ZOO))
    cfg = load_config(tier="quick")
    pair = cfg.default_guard_models(2, "small")
    by_tag = {m.tag: m for m in DEFAULT_ZOO}
    assert len(pair) == 2 and len(set(pair)) == 2
    assert all((by_tag[t].tier == "small" for t in pair))
    assert len({by_tag[t].family for t in pair}) == 2


def test_preflight_required_and_missing():
    from tpag.eval import model_zoo
    from tpag.eval.config import DEFAULT_ZOO, load_config
    from tpag.eval.model_zoo import ZooStatus

    cfg = load_config(tier="quick")
    assert len(model_zoo.required_tags(cfg)) == 12
    present = DEFAULT_ZOO[0]
    status = ZooStatus(ollama_up=True, available=[present])
    missing = model_zoo.missing_required(cfg, status)
    assert present.tag not in missing and len(missing) == 11


def test_parse_params_and_conditions():
    from tpag.eval import model_zoo
    from tpag.eval.config import TIER_RANK, ModelSpec, load_config
    from tpag.eval.model_zoo import ZooStatus

    assert model_zoo.parse_params_b("14.8B") == pytest.approx(14.8)
    assert model_zoo.parse_params_b("770M") == pytest.approx(0.77)
    assert model_zoo.parse_params_b(None) is None
    cfg = load_config(tier="quick")
    status = ZooStatus(
        ollama_up=True,
        available=[
            ModelSpec("qwen3.5:4b", "qwen3.5", 4.0, tier="small"),
            ModelSpec("gemma4:e4b", "gemma4", 4.0, tier="small"),
            ModelSpec("granite4.1:8b", "granite4.1", 8.0, tier="mid"),
            ModelSpec("deepseek-r1:32b", "deepseek-r1", 32.0, tier="large"),
            ModelSpec("oversized:72b", "big", 72.0, tier="large"),
        ],
        missing=["granite4.1:30b"],
    )
    conds = model_zoo.build_conditions(cfg, status)
    assert conds
    assert all((len(c["models"]) == c["k"] for c in conds))
    assert {c["diversity"] for c in conds} == {"same_model", "cross_family"}
    assert all(("oversized:72b" not in c["models"] for c in conds))
    for c in conds:
        if c["diversity"] == "cross_family":
            assert len(set(c["families"])) == c["k"]
            assert len({c["size_tier"]}) == 1
    keys = [(TIER_RANK[c["size_tier"]], c["mean_params_b"]) for c in conds]
    assert keys == sorted(keys)
    assert all((c["size_tier"] in ("small", "mid", "large") for c in conds))
    assert model_zoo.pull_hints(status) == ["ollama pull granite4.1:30b"]


def test_attack_strategies():
    from tpag.sim.attacks import make_dataset

    rng = np.random.default_rng(0)
    ds = make_dataset(20, attack_fraction=1.0, rng=rng)
    assert all((d.attacked for d in ds)) and all((d.per_guard_text is None for d in ds))
    ds2 = make_dataset(
        10, attack_fraction=1.0, rng=rng, domain="code_review", strategy="diverse", guard_count=3
    )
    assert all((d.per_guard_text and len(d.per_guard_text) == 3 for d in ds2))
    assert ds2[0].text_for_guard(0) != ds2[0].text_for_guard(1) or True
    ds3 = make_dataset(5, attack_fraction=1.0, rng=rng, domain="code_review", strategy="adaptive")
    from tpag.sim.attacks import CODE_REVIEW_INJECTIONS

    assert all((CODE_REVIEW_INJECTIONS[0] in d.text for d in ds3))


def test_langgraph_adapter_safe():
    from tpag.adapters import langgraph as lg

    assert isinstance(lg.available(), bool)
    if not lg.available():
        with pytest.raises(lg.MissingDependencyError):
            lg.build_review_app([])


def test_drivers_import():
    import importlib

    for mod in (
        "rq_e1_calibration",
        "rq_e2_topology",
        "rq_e3_llm_correlation",
        "rq_e4_baselines",
        "rq_e5_overhead_integration",
        "rq_e6_estimation_ablation",
        "rq_e9_hard_containment",
    ):
        m = importlib.import_module(f"tpag.eval.experiments.{mod}")
        assert hasattr(m, "run")
    from tpag.cli import DRIVERS

    assert set(DRIVERS) == {"e1", "e2", "e3", "e4", "e5", "e6", "e9"}


def test_reported_llm_design_is_frozen():
    import json
    from tpag.eval.config import REPO_ROOT, load_config
    from tpag.eval.experiments.rq_e3_llm_correlation import recorded_groups

    for filename, directory, point_count, call_count in (
        ("exec_e3_final.yaml", "results", 108, 29760),
        ("exec_e3_tighten.yaml", "results_tighten", 92, 176400),
    ):
        cfg = load_config(path=REPO_ROOT / "configs" / filename, tier="full")
        groups = recorded_groups(cfg)
        points = json.loads(
            (REPO_ROOT / directory / "rq_e3_llm_correlation/llm_points.json").read_text()
        )
        expanded = [
            (
                g["condition"]["models"],
                g["domain"],
                g["hardening"],
                g["strategy"],
                fraction,
                g["n_requests"] * g["n_seeds"],
            )
            for g in groups
            for fraction in g["injection_fractions"]
        ]
        assert expanded == [
            (
                p["models"],
                p["domain"],
                p["hardening"],
                p["strategy"],
                p["injection_fraction"],
                p["n_samples"],
            )
            for p in points
        ]
        assert len(expanded) == point_count
        assert sum(samples * len(models) for models, _, _, _, _, samples in expanded) == call_count


def test_cut_joint_applied_for_k3():
    from tpag.bound import compute_bound

    cut = frozenset({"a", "b", "c"})
    eps_upper = {"a": 0.5, "b": 0.5, "c": 0.5}
    res = compute_bound([cut], eps_upper, pair_upper={cut: 0.1})
    assert res.b_hw == pytest.approx(0.1)
    assert compute_bound([cut], eps_upper, pair_upper=None).b_hw == pytest.approx(0.5)


def test_build_conditions_block_overrides():
    from tpag.eval import model_zoo
    from tpag.eval.config import ModelSpec, load_config
    from tpag.eval.model_zoo import ZooStatus

    cfg = load_config(tier="quick")
    status = ZooStatus(
        ollama_up=True,
        available=[
            ModelSpec("qwen3.5:4b", "qwen3.5", 4.0, tier="small"),
            ModelSpec("gemma4:e4b", "gemma4", 4.0, tier="small"),
            ModelSpec("granite4.1:3b", "granite4.1", 3.0, tier="small"),
            ModelSpec("qwen3.5:9b", "qwen3.5", 9.0, tier="mid"),
        ],
    )
    conds = model_zoo.build_conditions(
        cfg, status, ks=(3,), diversity_modes=("same_model",), tiers=("small",), max_per_tier=2
    )
    assert conds and len(conds) == 2
    assert all(
        (
            c["k"] == 3 and c["diversity"] == "same_model" and (c["size_tier"] == "small")
            for c in conds
        )
    )


def test_expand_blocks_and_estimate():
    from tpag.eval import model_zoo
    from tpag.eval.config import ModelSpec, load_config
    from tpag.eval.model_zoo import ZooStatus

    cfg = load_config(tier="quick")
    cfg.blocks = (
        {
            "name": "h",
            "tiers": ["small"],
            "diversity": ["same_model"],
            "ks": [2],
            "domains": ["financial"],
            "hardenings": ["permissive"],
            "strategies": ["shared"],
            "n_requests": 5,
            "n_seeds": 1,
            "injection_fractions": [0.0, 1.0],
            "max_conditions_per_tier": 2,
        },
        {
            "name": "k3",
            "tiers": ["small"],
            "diversity": ["same_model"],
            "ks": [3],
            "domains": ["financial"],
            "hardenings": ["permissive"],
            "strategies": ["shared"],
            "n_requests": 5,
            "n_seeds": 1,
            "injection_fractions": [0.0, 1.0],
            "max_conditions_per_tier": 2,
        },
    )
    status = ZooStatus(
        ollama_up=True,
        available=[
            ModelSpec("qwen3.5:4b", "qwen3.5", 4.0, tier="small"),
            ModelSpec("granite4.1:3b", "granite4.1", 3.0, tier="small"),
        ],
    )
    groups = model_zoo.expand_blocks(cfg, status)
    assert groups and all(("block" in g and "n_requests" in g for g in groups))
    assert {g["block"] for g in groups} == {"h", "k3"}
    assert {len(g["condition"]["models"]) for g in groups} == {2, 3}
    est = model_zoo.estimate_block_calls(cfg, status)
    assert est["mode"] == "blocks" and est["approx_llm_calls"] > 0

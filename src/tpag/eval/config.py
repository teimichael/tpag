"""Config for the TPAG replication package."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

# Repo-root-relative default config locations (resolved lazily; optional).
_HERE = Path(__file__).resolve()
# src/tpag/eval/config.py -> repo root is parents[3]
REPO_ROOT = _HERE.parents[3]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"

# Benchmark/protocol version recorded in every run manifest (bump on any change to
# the workloads, attack catalogs, scoring, or approved-model registry that would
# alter results). 1.2: switched to the approved 4-family x 3-scale model registry.
BENCHMARK_VERSION = "tpag-ext-1.2"

# Hard upper bound on model size (billions of parameters). No experiment may use a
# model with >= this many parameters; models at/above it are reported and excluded.
HARD_MAX_PARAMS_B = 70.0


# ---------------------------------------------------------------------------
# Tier: every sample size / grid / cap in one place.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tier:
    """All numeric knobs for one run size. See :data:`DEFAULT_TIERS`."""

    name: str

    # RQ-E1 (calibration quality)
    episodes: int  # episodes per scripted calibration config
    eta_grid: tuple[float, ...]  # confidence levels swept
    boot_resamples: int  # bootstrap resamples for CIs
    holdout_fraction: float  # train/test split for out-of-sample coverage

    # RQ-E2 (topology generalization + scalability)
    topology_sizes: tuple[int, ...]  # N grid for the soundness sweep
    mc_samples: int  # Monte-Carlo episodes for ground-truth Pr[¬Φ]
    scaling_sizes: tuple[int, ...]  # N grid for the cut-set/bound timing curve
    scaling_timeout_s: float  # per-point wall-clock cap (fail safe)
    max_cuts: int  # abort a structure whose cut count exceeds this

    # RQ-E3 (real-LLM correlation across the model zoo)
    n_requests: int  # requests per (condition, injection fraction)
    n_seeds: int  # independent seeds per condition (for CIs)
    injection_fractions: tuple[float, ...]
    ks: tuple[int, ...]  # redundancy widths to test (e.g. 2, 3)
    max_conditions: int  # safety cap on the condition matrix size

    # RQ-E5 (overhead at scale)
    overhead_sizes: tuple[int, ...]  # component counts for the overhead curve
    overhead_episodes: int  # episodes per overhead point
    integration_requests: int  # live LangGraph requests for the e2e run


DEFAULT_TIERS: dict[str, Tier] = {
    # CPU smoke: everything runs in well under a minute, no GPU/LLM needed.
    "quick": Tier(
        name="quick",
        episodes=800,
        eta_grid=(0.01, 0.05),
        boot_resamples=500,
        holdout_fraction=0.5,
        topology_sizes=(4, 8, 16),
        mc_samples=2000,
        scaling_sizes=(4, 8, 16, 32),
        scaling_timeout_s=5.0,
        max_cuts=20000,
        n_requests=8,
        n_seeds=1,
        injection_fractions=(0.0, 0.5, 1.0),
        ks=(2,),
        max_conditions=4,
        overhead_sizes=(4, 16, 64),
        overhead_episodes=2000,
        integration_requests=4,
    ),
    # Overnight on a single RTX 5090 (LLM-dominated; CPU parts are minutes).
    "full": Tier(
        name="full",
        episodes=8000,
        eta_grid=(0.001, 0.01, 0.05, 0.1),
        boot_resamples=2000,
        holdout_fraction=0.5,
        topology_sizes=(4, 8, 16, 32, 64),
        mc_samples=20000,
        scaling_sizes=(4, 8, 16, 32, 64, 96, 128),
        scaling_timeout_s=30.0,
        max_cuts=2_000_000,
        n_requests=200,
        n_seeds=3,
        injection_fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
        ks=(2, 3),
        max_conditions=24,
        overhead_sizes=(4, 16, 64, 256, 1024),
        overhead_episodes=20000,
        integration_requests=24,
    ),
}


# ---------------------------------------------------------------------------
# Approved-model registry (the SINGLE source of truth for LLM usage).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """An approved Ollama model.

    ``tier`` (small/mid/large) is the *assigned scale* used for the size sweep --
    it is authoritative and not re-derived from ``params_b``. ``family`` groups
    models for the same-model vs cross-family redundancy contrast (RQ-E3).
    ``params_b`` is the nominal parameter count in billions (used for ordering,
    reporting, and the hard < 70B cap)."""

    tag: str
    family: str
    params_b: float
    note: str = ""
    tier: str = ""  # "small" | "mid" | "large" (authoritative; see registry)

    def resolved_tier(self, cfg: ExpConfig) -> str:
        """Explicit registry tier, falling back to the params-based classifier for
        ad-hoc specs that carry no tier."""
        return self.tier or cfg.size_tier(self.params_b)


# The 4 approved families and the 3 approved scales. The registry below is exactly
# the 4 x 3 = 12 approved models -- see ``validate_registry``.
FAMILIES: tuple[str, ...] = ("qwen3.5", "gemma4", "granite4.1", "deepseek-r1")
TIERS: tuple[str, ...] = ("small", "mid", "large")
TIER_RANK: dict[str, int] = {"small": 0, "mid": 1, "large": 2}


# The APPROVED model registry: exactly 4 families x 3 scales = 12 models, all the
# explicitly-approved Ollama identifiers (preserved verbatim). These are the ONLY
# LLMs the implementation and experiments may use. All are < 70B (the hard cap).
# Availability is detected at runtime (``tpag models`` / ``tpag preflight``); a model
# absent from the local Ollama is reported as missing with an ``ollama pull`` hint,
# never fabricated or auto-downloaded. ``validate_registry`` enforces the 4x3 shape.
DEFAULT_ZOO: tuple[ModelSpec, ...] = (
    # -- small --
    ModelSpec("qwen3.5:4b", "qwen3.5", 4.0, tier="small"),
    ModelSpec("gemma4:e4b", "gemma4", 4.0, tier="small", note="effective ~4B"),
    ModelSpec("granite4.1:3b", "granite4.1", 3.0, tier="small"),
    ModelSpec("deepseek-r1:1.5b", "deepseek-r1", 1.5, tier="small", note="reasoning (distill)"),
    # -- mid --
    ModelSpec("qwen3.5:9b", "qwen3.5", 9.0, tier="mid"),
    ModelSpec("gemma4:12b", "gemma4", 12.0, tier="mid"),
    ModelSpec("granite4.1:8b", "granite4.1", 8.0, tier="mid"),
    ModelSpec("deepseek-r1:8b", "deepseek-r1", 8.0, tier="mid", note="reasoning (distill)"),
    # -- large (all < 70B) --
    ModelSpec("qwen3.5:27b", "qwen3.5", 27.0, tier="large"),
    ModelSpec("gemma4:26b", "gemma4", 26.0, tier="large"),
    ModelSpec("granite4.1:30b", "granite4.1", 30.0, tier="large"),
    ModelSpec("deepseek-r1:32b", "deepseek-r1", 32.0, tier="large", note="reasoning (distill)"),
)


def validate_registry(zoo: tuple[ModelSpec, ...] = DEFAULT_ZOO) -> None:
    """Fail loudly unless the registry is exactly the approved 4 families x 3 scales
    x 12 models, all < 70B, with one model per (family, tier) cell."""
    if len(zoo) != 12:
        raise ValueError(f"registry must have exactly 12 approved models, got {len(zoo)}")
    fams = {m.family for m in zoo}
    if fams != set(FAMILIES):
        raise ValueError(f"registry families {sorted(fams)} != approved {sorted(FAMILIES)}")
    tiers = {m.tier for m in zoo}
    if tiers != set(TIERS):
        raise ValueError(f"registry tiers {sorted(tiers)} != approved {sorted(TIERS)}")
    cells = {(m.family, m.tier) for m in zoo}
    if len(cells) != 12:
        raise ValueError("registry must have exactly one model per (family, tier) cell")
    over = [m.tag for m in zoo if m.params_b >= 70.0]
    if over:
        raise ValueError(f"models >= 70B are forbidden: {over}")


# ---------------------------------------------------------------------------
# Top-level experiment configuration.
# ---------------------------------------------------------------------------
@dataclass
class ExpConfig:
    tier: Tier
    seed: int = 20260607
    # Canonical output tree. The committed results/ are produced by this suite; the
    # reproduction command regenerates them in place (with --force). Smoke/quick/dry
    # configs point --out elsewhere to avoid clobbering the canonical tree.
    out: str = "outputs/run"

    # model-zoo policy (size tiers; prefer the smallest sufficient model)
    min_params_b: float = 12.0  # small/mid boundary; ">= this" is "larger" (needs justification)
    mid_max_b: float = 32.0  # mid/large boundary
    max_params_b: float = HARD_MAX_PARAMS_B  # HARD cap: models >= this are excluded
    zoo: tuple[ModelSpec, ...] = DEFAULT_ZOO
    ollama_url: str = "http://localhost:11434"

    # RQ-E3 sweep axes (top-level defaults; a block may override per axis)
    domains: tuple[str, ...] = ("financial", "code_review")
    injection_strategies: tuple[str, ...] = ("shared", "diverse", "adaptive")
    hardenings: tuple[str, ...] = ("permissive", "strict")
    diversity_modes: tuple[str, ...] = ("same_model", "cross_family")

    # Optional fractional design: a list of experiment *blocks*. When non-empty the
    # RQ-E3 driver runs the union of the blocks (each a small Cartesian with its own
    # axes/n/seeds/fractions/per-tier cap) instead of the full Cartesian -- the only
    # feasible way to cover all axes with power given large-tier latency. See
    # configs/exec_e3_final.yaml. Each block is a dict with optional keys:
    #   name, tiers, diversity, ks, domains, hardenings, strategies,
    #   n_requests, n_seeds, injection_fractions, max_conditions_per_tier.
    blocks: tuple[dict, ...] = ()

    # bookkeeping
    benchmark_version: str = BENCHMARK_VERSION
    source: str = "built-in defaults"

    # -- convenience -------------------------------------------------------
    @property
    def out_path(self) -> Path:
        return Path(self.out)

    def size_tier(self, params_b: float) -> str:
        """Classify a model size: small / mid / large / excluded (>= hard cap)."""
        if params_b >= self.max_params_b:
            return "excluded"
        if params_b < self.min_params_b:
            return "small"
        if params_b < self.mid_max_b:
            return "mid"
        return "large"

    def zoo_eligible(self) -> tuple[ModelSpec, ...]:
        """Registry models permitted by the hard size cap (< max_params_b)."""
        return tuple(m for m in self.zoo if m.params_b < self.max_params_b)

    def zoo_by_tier(self) -> dict[str, list[ModelSpec]]:
        """Group the registry by its *explicit* (assigned) tier."""
        out: dict[str, list[ModelSpec]] = {t: [] for t in TIERS}
        for m in self.zoo:
            out.setdefault(m.resolved_tier(self), []).append(m)
        return out

    def zoo_by_family(self) -> dict[str, list[ModelSpec]]:
        out: dict[str, list[ModelSpec]] = {}
        for m in self.zoo:
            out.setdefault(m.family, []).append(m)
        return out

    def model_by_cell(self, family: str, tier: str) -> ModelSpec | None:
        """The single approved model at (family, tier), or None."""
        for m in self.zoo:
            if m.family == family and m.resolved_tier(self) == tier:
                return m
        return None

    def default_guard_models(self, k: int = 2, tier: str = "small") -> list[str]:
        """``k`` approved tags at ``tier`` from *distinct* families (cross-family),
        smallest first, for integration reference calls."""
        pool = sorted(
            (m for m in self.zoo if m.resolved_tier(self) == tier),
            key=lambda m: (m.params_b, m.tag),
        )
        chosen: list[str] = []
        seen: set[str] = set()
        for m in pool:
            if m.family not in seen:
                chosen.append(m.tag)
                seen.add(m.family)
            if len(chosen) == k:
                break
        # fall back to smallest-of-any if fewer than k distinct families
        for m in pool:
            if len(chosen) >= k:
                break
            if m.tag not in chosen:
                chosen.append(m.tag)
        return chosen

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # frozen tier/specs serialize cleanly via asdict; tuples -> lists
        return d

    def config_hash(self) -> str:
        """A stable short hash of the full configuration, recorded in manifests."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Loading / overlay.
# ---------------------------------------------------------------------------
def _coerce_tier(name: str, overrides: dict[str, Any] | None) -> Tier:
    base = DEFAULT_TIERS.get(name)
    if base is None:
        raise ValueError(f"unknown tier {name!r}; choose from {sorted(DEFAULT_TIERS)}")
    if not overrides:
        return base
    fields = {f for f in base.__dataclass_fields__}  # type: ignore[attr-defined]
    clean: dict[str, Any] = {}
    for k, v in overrides.items():
        if k not in fields:
            print(f"[config] WARNING: ignoring unknown tier field {k!r}")
            continue
        # tuples in the dataclass -> coerce lists from YAML back to tuples
        if isinstance(getattr(base, k), tuple) and isinstance(v, list):
            v = tuple(v)
        clean[k] = v
    return replace(base, **clean)


def load_config(
    path: str | Path | None = None,
    tier: str = "full",
    seed: int | None = None,
    out: str | None = None,
) -> ExpConfig:
    """Build an :class:`ExpConfig`.

    With no ``path`` the built-in defaults are used (the harness is fully usable
    out of the box). A YAML file may override the tier knobs, the zoo, the sweep
    axes, and ``min_params_b``. CLI-supplied ``tier``/``seed``/``out`` win over
    the file.
    """
    data: dict[str, Any] = {}
    cfg_path = Path(path) if path else (DEFAULT_CONFIG_DIR / "experiments.yaml")
    if cfg_path.exists():
        import yaml

        data = yaml.safe_load(cfg_path.read_text()) or {}

    tier_name = tier or data.get("tier", "full")
    tier_overrides = (data.get("tiers", {}) or {}).get(tier_name, {})
    tier_obj = _coerce_tier(tier_name, tier_overrides)

    zoo = DEFAULT_ZOO
    if data.get("zoo"):
        zoo = tuple(
            ModelSpec(
                tag=m["tag"],
                family=m.get("family", "?"),
                params_b=float(m.get("params_b", 0.0)),
                note=m.get("note", ""),
                tier=m.get("tier", ""),
            )
            for m in data["zoo"]
        )

    def _tuple(key: str, default: tuple) -> tuple:
        v = data.get(key)
        return tuple(v) if isinstance(v, list) else default

    cfg = ExpConfig(
        tier=tier_obj,
        seed=int(seed if seed is not None else data.get("seed", 20260607)),
        out=str(out if out is not None else data.get("out", "outputs/run")),
        min_params_b=float(data.get("min_params_b", 12.0)),
        mid_max_b=float(data.get("mid_max_b", 32.0)),
        max_params_b=float(data.get("max_params_b", HARD_MAX_PARAMS_B)),
        zoo=zoo,
        ollama_url=str(data.get("ollama_url", "http://localhost:11434")),
        domains=_tuple("domains", ExpConfig.domains),
        injection_strategies=_tuple("injection_strategies", ExpConfig.injection_strategies),
        hardenings=_tuple("hardenings", ExpConfig.hardenings),
        diversity_modes=_tuple("diversity_modes", ExpConfig.diversity_modes),
        blocks=tuple(data.get("blocks") or ()),
        source="configs/" + cfg_path.name if cfg_path.exists() else "built-in defaults",
    )
    return cfg


def estimate_llm_calls(
    cfg: ExpConfig,
    available_eligible: int | None = None,
    n_model_conditions: int | None = None,
) -> dict[str, Any]:
    """Order-of-magnitude estimate of the RQ-E3 LLM call budget for a *dry run*.

    Mirrors the driver exactly: it sweeps a capped list of *groups*
    ``= (model-condition x domain x hardening x injection-strategy)``, each over
    ``injection_fractions x n_seeds x n_requests`` with ``~avg_guards`` calls per
    request. Contacts no model. Pass ``n_model_conditions`` (from
    :func:`tpag.eval.model_zoo.build_conditions`) for an exact count, or
    ``available_eligible`` to estimate it from the count of eligible (<70B) models.
    """
    from math import comb

    t = cfg.tier
    n_elig = available_eligible if available_eligible is not None else len(cfg.zoo_eligible())
    if n_model_conditions is None:
        # Mirror the driver's *within-tier* condition count over the registry.
        n_model_conditions = 0
        for models in cfg.zoo_by_tier().values():
            if not models:
                continue
            n_models = len(models)
            n_fams = len({m.family for m in models})
            for k in t.ks:
                if "same_model" in cfg.diversity_modes:
                    n_model_conditions += n_models
                if "cross_family" in cfg.diversity_modes and n_fams >= k:
                    n_model_conditions += comb(n_fams, k)

    groups_raw = (
        n_model_conditions * len(cfg.domains) * len(cfg.hardenings) * len(cfg.injection_strategies)
    )
    groups = min(groups_raw, t.max_conditions)
    avg_guards = sum(t.ks) / len(t.ks)
    calls = int(groups * len(t.injection_fractions) * t.n_requests * t.n_seeds * avg_guards)
    # rough per-call latency band: small/mid models on a 5090 are cheap; the prefer-
    # small policy keeps the default budget low, larger models only as a check.
    return {
        "tier": t.name,
        "eligible_models": n_elig,
        "model_conditions": n_model_conditions,
        "groups_capped": groups,
        "groups_uncapped": groups_raw,
        "approx_llm_calls": calls,
        "approx_wall_hours_low": round(calls * 0.08 / 3600.0, 2),
        "approx_wall_hours_high": round(calls * 0.40 / 3600.0, 2),
        "note": "estimate only; no model contacted. Tune n_requests/n_seeds/ks/max_conditions to fit budget.",
    }


__all__ = [
    "Tier",
    "ModelSpec",
    "ExpConfig",
    "DEFAULT_TIERS",
    "DEFAULT_ZOO",
    "DEFAULT_CONFIG_DIR",
    "FAMILIES",
    "TIERS",
    "TIER_RANK",
    "validate_registry",
    "load_config",
    "estimate_llm_calls",
]

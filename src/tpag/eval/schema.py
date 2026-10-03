"""Schema for the TPAG replication package."""

from __future__ import annotations
import shutil
import subprocess
import sys
from typing import Any

REQUIRED_KEYS = ("rq", "status", "tier", "seed", "config_hash")
TERMINAL_STATUSES = {"measured", "computed", "not_run"}


def gpu_info() -> dict[str, Any]:
    """Best-effort GPU description. Returns ``{"available": False}`` when no
    ``nvidia-smi`` is present (e.g. CPU-only CI), never raising."""
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "reason": "nvidia-smi not found"}
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            .decode()
            .strip()
        )
        gpus = [line.strip() for line in out.splitlines() if line.strip()]
        return {"available": bool(gpus), "gpus": gpus}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def env_info() -> dict[str, Any]:
    """Python, platform, and key package versions (best-effort)."""
    from importlib.metadata import PackageNotFoundError, version

    pkgs: dict[str, str] = {}
    for name in ("numpy", "scipy", "networkx", "matplotlib", "pyyaml", "httpx", "langgraph"):
        try:
            pkgs[name] = version(name)
        except PackageNotFoundError:
            pkgs[name] = "not-installed"
    return {"python": sys.version.split()[0], "packages": pkgs}


def process_memory_mib() -> float | None:
    """Best-effort peak resident set size in MiB (Linux/macOS), else ``None``."""
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru / 1024.0 if sys.platform != "darwin" else ru / (1024.0 * 1024.0)
    except Exception:
        return None


def base_manifest(
    rq: str, cfg, argv: list[str] | None = None, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Construct the standard manifest dict for an extended-suite run.

    ``cfg`` is an :class:`~tpag.eval.config.ExpConfig`. The returned dict is handed
    to :class:`~tpag.recording.RunRecorder`, which also stamps package version and Python version.
    """
    m: dict[str, Any] = {
        "rq": rq,
        "suite": "extended",
        "benchmark_version": getattr(cfg, "benchmark_version", "unknown"),
        "tier": cfg.tier.name,
        "seed": cfg.seed,
        "config_hash": cfg.config_hash(),
        "config_source": cfg.source,
        "config_path": cfg.source,
        "argv": argv if argv is not None else ["tpag"],
        "source_snapshot": "released-package",
        "env": env_info(),
        "min_params_b": cfg.min_params_b,
        "max_params_b": cfg.max_params_b,
        "gpu": gpu_info(),
    }
    if extra:
        m.update(extra)
    return m


def validate_metrics(metrics: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problems (empty == valid).

    Enforces the required keys and the status discipline so a malformed or
    silently-empty record is caught in tests and in ``--validate`` runs.
    """
    problems: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in metrics:
            problems.append(f"missing required key: {key!r}")
    status = metrics.get("status")
    if status is not None and (
        not (status in TERMINAL_STATUSES or str(status).startswith("skipped_"))
    ):
        problems.append(
            f"invalid status {status!r}; expected one of {sorted(TERMINAL_STATUSES)} or a 'skipped_*' value"
        )
    return problems


__all__ = [
    "gpu_info",
    "env_info",
    "process_memory_mib",
    "base_manifest",
    "validate_metrics",
    "REQUIRED_KEYS",
]

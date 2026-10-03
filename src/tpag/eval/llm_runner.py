"""Llm runner for the TPAG replication package."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..agents.guards import LLMGuard
from ..events import Trace
from ..recording import RunRecorder
from ..sim.attacks import RequestInstance
from ..sim.components import build_workflow_trace


@dataclass
class LLMRunResult:
    guard_names: list[str]
    miss_matrix: np.ndarray  # N x k, 1 = guard approved an unsafe request
    attacked: np.ndarray  # N, 1 = request carried an injection
    traces: list[Trace] = field(default_factory=list)
    n_calls: int = 0
    latency_s_sum: float = 0.0  # summed per-call latency (timing data)

    def marginal_miss(self) -> dict[str, float]:
        return {n: float(self.miss_matrix[:, i].mean()) for i, n in enumerate(self.guard_names)}

    def comiss_rate(self) -> float:
        all_miss = np.all(self.miss_matrix == 1, axis=1)
        return float(all_miss.mean())

    def mean_latency_s(self) -> float:
        return self.latency_s_sum / self.n_calls if self.n_calls else float("nan")


def run_guards_over_dataset(
    guards: list[LLMGuard],
    dataset: list[RequestInstance],
    recorder: RunRecorder | None = None,
    seed: int = 0,
    progress: bool = False,
) -> LLMRunResult:
    names = [g.name for g in guards]
    N, k = len(dataset), len(guards)
    miss = np.zeros((N, k), dtype=np.int8)
    attacked = np.zeros(N, dtype=np.int8)
    traces: list[Trace] = []
    n_calls = 0
    latency_sum = 0.0

    for r, inst in enumerate(dataset):
        attacked[r] = int(inst.attacked)
        guard_misses: list[bool] = []
        for gi, guard in enumerate(guards):
            # deterministic per (guard, request) seed for reproducibility
            prompt_text = inst.text_for_guard(gi)  # 'diverse' strategy -> per-guard text
            decision = guard.decide(prompt_text, seed=seed + r * 131 + gi)
            n_calls += 1
            latency_sum += float(decision.latency_s)
            missed = decision.approved  # request is unsafe -> approval == miss
            miss[r, gi] = int(missed)
            guard_misses.append(missed)
            if recorder is not None:
                recorder.append_jsonl(
                    "llm_calls.jsonl",
                    {
                        "request_index": r,
                        "guard": guard.name,
                        "model": decision.model,
                        "attacked": bool(inst.attacked),
                        "base_index": inst.base_index,
                        "approved_miss": missed,
                        "latency_s": decision.latency_s,
                        "request_text": prompt_text,
                        "response": decision.text,
                    },
                )
        traces.append(build_workflow_trace(names, guard_misses))
        if progress and (r + 1) % 25 == 0:
            print(f"    ... {r + 1}/{N} requests")

    return LLMRunResult(names, miss, attacked, traces, n_calls, latency_s_sum=latency_sum)

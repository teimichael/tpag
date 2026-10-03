# Contracts That Compose replication package

This anonymous package contains the implementation and evidence for the paper's
reported results. Inspection and CPU reproduction require no model service.
Experimental guard prompts and model responses are retained as research data.

## Structure and correspondence

| Location | Manuscript correspondence |
|---|---|
| `src/tpag/` | Contracts, causal monitors, containment, estimation and composition bound; trace adapters and retained experiment drivers |
| `tests/` | Relevant framework, soundness and evaluation checks |
| `configs/` | Full scripted parameters and the two reported LLM designs; approved model registry and numerical tier defaults are in `src/tpag/eval/config.py` |
| `results/rq_e3_llm_correlation/` | RQ1 main suite: 108 points, 29,760 calls; also RQ2/RQ3 log analyses |
| `results_tighten/rq_e3_llm_correlation/` | RQ1 re-measurement: 92 points, 176,400 calls; also held-out and replay analyses |
| `results/rq_e1_calibration/` | Scripted calibration, confidence sweep and 13 held-out implementation checks |
| `results/rq_e2_topology/` | RQ3 structure families and certification scaling |
| `results/rq_e4_baselines/`, `results/rq_e6_estimation_ablation/` | RQ3 composition-rule and estimation ablations |
| `results/rq_e5_overhead_integration/` | Monitoring costs, reference LLM latency and 24 LangGraph episodes |
| `results/rq_e9_hard_containment/` | Runtime containment and causal/recording-order scheduling checks |
| `scripts/` | Manuscript statistics/figures, result comparison and integrity/anonymity audit |
| `SHA256SUMS` | Checksums of every submitted file except the inventory itself |

Driver labels E1–E6/E9 identify implementation modules, not the paper's three RQs.
Injection-shift comparisons and composed workflows are derived from the released
LLM points/logs; they require no separate experimental campaign. Generated output
goes to `outputs/`, which is not part of the submission.

## Setup

Tested on Linux with Python 3.13.9. Use Python 3.13 and Bash:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
```

The pinned requirements include the test runner and LangGraph integration, with
all transitive dependencies. Installation downloads public packages; no private
repository, credentials or parent checkout is required. Dependencies, environments
and model weights are deliberately not vendored into the submission.

## Inspect the reported results

```bash
./reproduce.sh --inspect
python -m pytest -m "not llm"
python scripts/audit.py
```

Inspection produces `outputs/derived/derived_stats.json`, per-stream held-out rows,
injection-shift rows, and `outputs/figures/independence_scatter.pdf` and `validity.pdf`.
Expected checks: 47 independence violations (34 CI-robust) in the main suite and
57 (57 CI-robust) in the re-measurement; 345/348 held-out stream splits covered,
with none below a 95% lower interval; 720/720 composed-workflow tests covered.
The mean composed-bound excess over the direct limit is 0.0836284830.

The two `llm_calls.jsonl.gz` files are lossless, ordered JSON Lines streams.
Readers decompress them directly using Python's standard library. Point order
determines call blocks (`n_samples * k` records), and `request_index` resets at each
seeded stream. Verification checks decisions, guard/model order, marginals, co-miss
counts and stream boundaries. Requests/payloads and guard prompts remain unchanged.
The only email and external URL in workloads are synthetic experimental addresses.

## Rerun experiments

```bash
./reproduce.sh --quick                 # CPU smoke; outputs/quick/
./reproduce.sh --cpu                   # full CPU campaigns; outputs/cpu/
tpag run-ext e1 --tier full --no-llm --out outputs/single
```

The full CPU run checks deterministic measurements against the released data.
It runs calibration, topology/scaling, rule comparisons, monitoring overhead,
estimation ablation and runtime checks. Timing and memory vary by machine.
Allow tens of minutes or longer for full CPU reproduction. Scaling skips points above its cut-count cap and stops larger points in a family
after a completed point exceeds its timing budget; individual points can exceed
that budget. Quick results
use smaller samples and are smoke checks, not the manuscript's measurements.

Optional full LLM reproduction requires a local Ollama service on its default port
and all twelve approved model tags. `tpag models` lists availability and pull
commands; install weights manually, then run:

```bash
tpag preflight --config configs/exec_e3_final.yaml
tpag run-ext e3 --config configs/exec_e3_final.yaml --out outputs/estimate --dry-run
./reproduce.sh --llm
```

This writes both campaigns and live integration under `outputs/llm/` and fails if
a campaign or integration is incomplete. Original LLM campaigns took 26.5 and
66.3 hours on one RTX 5090 with approximately 32 GiB VRAM. They used temperature 0,
per-call seeds, reasoning disabled and constrained JSON decisions. Exact serving
builds and weight digests were not fully captured in the retained historical data;
model tags and available parameter metadata are preserved. Campaign model groups,
point order and sample sizes are frozen to the released operating points, so
changes in available model metadata cannot silently select a different design. The paper reports
decision drift on replay, so fresh decisions need not match released logs.

Repeated commands refuse to replace completed output unless `--force` is supplied:

```bash
./reproduce.sh --inspect --force
```

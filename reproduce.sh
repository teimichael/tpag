#!/usr/bin/env bash
# Run from an activated environment installed using README.md.
set -euo pipefail
cd "$(dirname "$0")"
mode=inspect
selected=false
force=()
for arg in "$@"; do
  case "$arg" in
    --inspect|--quick|--cpu|--llm)
      if "$selected"; then echo "Choose one reproduction mode" >&2; exit 2; fi
      mode="${arg#--}"; selected=true ;;
    --force) force=(--force) ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export LANGSMITH_TRACING=false
export LANGCHAIN_TRACING_V2=false
python scripts/audit.py
case "$mode" in
  inspect)
    python scripts/derived_stats.py "${force[@]}"
    python scripts/check_results.py
    python scripts/make_figures.py "${force[@]}"
    ;;
  quick|cpu)
    tier=quick
    if [ "$mode" = cpu ]; then tier=full; fi
    for rq in e1 e2 e4 e5 e6 e9; do
      echo "Running $rq ($tier, CPU only)"
      tpag run-ext "$rq" --tier "$tier" --no-llm --out "outputs/$mode" "${force[@]}"
    done
    if [ "$mode" = cpu ]; then python scripts/check_cpu.py; fi
    ;;
  llm)
    tpag preflight --config configs/exec_e3_final.yaml
    tpag run-ext e3 --tier full --config configs/exec_e3_final.yaml --out outputs/llm/main "${force[@]}"
    tpag run-ext e3 --tier full --config configs/exec_e3_tighten.yaml --out outputs/llm/tighten "${force[@]}"
    tpag run-ext e5 --tier full --out outputs/llm/main "${force[@]}"
    python - <<'PY'
import json
from pathlib import Path
for suite, expected in (("main", 108), ("tighten", 92)):
    record = json.loads((Path("outputs/llm") / suite / "rq_e3_llm_correlation/metrics.json").read_text())
    if record["metrics"]["n_points"] != expected or record["metrics"]["n_errors"]:
        raise SystemExit(f"Incomplete {suite} campaign")
integration = json.loads(Path("outputs/llm/main/rq_e5_overhead_integration/metrics.json").read_text())["metrics"]["integration"]
if integration["status"] != "measured" or integration["live_episodes"] != 24:
    raise SystemExit("LangGraph integration was not completed")
print("Both LLM campaigns and live integration completed; decisions/timings may drift.")
PY
    ;;
esac

#!/usr/bin/env bash
# Per-dataset tuning for ETTm1, ETTm2 (stage1), PEMS03 (stage2),
# Electricity (stage3), Weather (stage4). Same stages as run_all.ps1.
# Usage: bash scripts/APCM_TUNING/run_all.sh [stage1|stage2|stage3|stage4|select|all]
set -euo pipefail
cd "$(dirname "$0")/../.."
export APCM_OUT="${APCM_OUT:-results_apcm_mse}"
PY="${APCM_PYTHON:-python}"
STAGE="${1:-stage1}"

BASE='GTR-FD5L1,APCM-FD5L1,APCM-CE-FD5L1'
LOSS_LR='GTR-MAE,GTR-HY,GTR-FD8,APCM-CE-FD8,APCM-CE-HY,GTR-FD5L1-LR5e4,GTR-FD5L1-LR2e3,APCM-FD5L1-LR5e4,APCM-FD5L1-LR2e3,APCM-CE-FD5L1-LR5e4,APCM-CE-FD5L1-LR2e3'
PEMS_EXTRA='APCM-FD5L1-K4,APCM-FD5L1-K16,APCM-FD5L1-P8'
ELEC_EXTRA='GTR-FD5L1-noRevIN,APCM-CE-FD5L1-noRevIN'
# Method extensions A (FC: future-cycle residual) and B (EMA weights).
METHOD_AB='GTR-FD5L1-FC,GTR-FD5L1-EMA,GTR-FD5L1-FC-EMA,APCM-CE-FD5L1-FC,APCM-CE-FD5L1-EMA,APCM-CE-FD5L1-FC-EMA'
METHOD_AB_PEMS='APCM-FD5L1-FC,APCM-FD5L1-EMA,APCM-FD5L1-FC-EMA,APCM-CE-FD5L1-FC,APCM-CE-FD5L1-EMA,APCM-CE-FD5L1-FC-EMA'

run() { "$PY" utils/run_apcm_experiments.py --stage full "$@"; }
select_ds() {  # $1 datasets, $2 candidates, $3 tag
  "$PY" utils/select_apcm_plus.py --per_dataset --only "$1" --candidates "$2" \
    --selection_seeds 2024,2025,2026 --tag "$3" | tee "$APCM_OUT/per_dataset_$3.txt" \
    || echo "Selection for $3 not complete yet (see SKIP lines)"
}

stage1() { run --only ETTm2,ETTm1 --variants "GTR,$BASE,$LOSS_LR,$METHOD_AB"; select_ds ETTm1,ETTm2 "$BASE,$LOSS_LR,$METHOD_AB" ett; }
stage2() { run --only PEMS03 --variants "GTR,$BASE,$PEMS_EXTRA,$METHOD_AB_PEMS"; select_ds PEMS03 "$BASE,$PEMS_EXTRA,$METHOD_AB_PEMS" pems03; }
stage3() { run --only Electricity --variants "GTR,$BASE,$ELEC_EXTRA,$METHOD_AB,GTR-noRevIN"; select_ds Electricity "$BASE,$ELEC_EXTRA,$METHOD_AB" electricity; }
stage4() { run --only Weather --variants "GTR,$BASE,$LOSS_LR,$METHOD_AB"; select_ds Weather "$BASE,$LOSS_LR,$METHOD_AB" weather; }
select_all() {
  select_ds ETTm1,ETTm2 "$BASE,$LOSS_LR,$METHOD_AB" ett
  select_ds PEMS03 "$BASE,$PEMS_EXTRA,$METHOD_AB_PEMS" pems03
  select_ds Electricity "$BASE,$ELEC_EXTRA,$METHOD_AB" electricity
  select_ds Weather "$BASE,$LOSS_LR,$METHOD_AB" weather
}

case "$STAGE" in
  stage1|stage2|stage3|stage4) "$STAGE" ;;
  select) select_all ;;
  all) stage1; stage2; stage3; stage4 ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 1 ;;
esac
echo "Done. Results are in $APCM_OUT/ (runs.jsonl, per_dataset_*.txt/csv)"

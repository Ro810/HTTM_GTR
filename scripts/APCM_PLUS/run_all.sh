#!/usr/bin/env bash
# APCM+ full benchmark (13/11/2026).  Run from the GTR-main folder:
#     bash scripts/APCM_PLUS/run_all.sh            # everything, in priority order
#     bash scripts/APCM_PLUS/run_all.sh stage1     # only the 1-seed screen
# Every command is resumable: finished jobs (by key in results_apcm/runs.jsonl)
# are skipped, so it is safe to stop (Ctrl+C) and start again.
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-python}
RUN="$PY utils/run_apcm_experiments.py --stage full"
NEW="GTR-FD5L1,APCM-FD5L1,APCM-CE-FD5L1"       # proposed method + ablation
FAST="ETTh1,ETTh2,ETTm1,ETTm2,Weather,Solar,PEMS03,PEMS04,PEMS08,Electricity"

stage1() {
  # 1 seed, shortest + longest horizon, every dataset except Traffic.
  # Purpose: check on GPU that the CPU screen holds and that APCM-CE helps.
  $RUN --only ETTh1,ETTh2,ETTm1,ETTm2,Weather,Solar,Electricity --pred_lens 96,720 --seeds 2024 --variants GTR,APCM,$NEW
  $RUN --only PEMS03,PEMS04,PEMS07,PEMS08 --pred_lens 12,96 --seeds 2024 --variants $NEW
}

stage2() {
  # Full grid: 4 horizons x 3 seeds.  GTR/APCM for ETT, Weather, Solar are new
  # baselines (the other six datasets already have them in runs.jsonl).
  $RUN --only ETTh1,ETTh2,ETTm1,ETTm2,Weather,Solar --variants GTR,APCM,$NEW
  $RUN --only PEMS03,PEMS04,PEMS08,Electricity --variants $NEW
  $RUN --only PEMS07 --variants $NEW
}

stage3() {
  # Traffic last: ~1.5 h per run.  Proposed method first, ablation after.
  $RUN --only Traffic --variants APCM-FD5L1,APCM-CE-FD5L1
  $RUN --only Traffic --variants GTR-FD5L1
}

case "${1:-all}" in
  stage1) stage1 ;;
  stage2) stage2 ;;
  stage3) stage3 ;;
  all) stage1; stage2; stage3 ;;
  *) echo "usage: $0 [stage1|stage2|stage3|all]"; exit 1 ;;
esac

$PY utils/summarize_apcm.py > results_apcm/summary.txt
$PY utils/select_apcm_plus.py --loss FD5L1 > results_apcm/apcm_plus.txt
echo "Done. Send back the whole results_apcm/ folder (runs.jsonl, logs/, *.csv, *.txt)."

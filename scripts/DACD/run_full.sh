# DACD vs GTR tren 12 dataset cua bai bao (may co GPU). Chay tu thu muc GTR-main:  bash scripts/DACD/run_full.sh
# Ket qua: results_dacd/full/runs.jsonl (chay lai se bo qua lan da xong). Huong dan: docs/DACD_RUN_GUIDE.md
WORKERS=${WORKERS:-2}

# Dot 1 (uu tien): GTR goc vs DACD-F1, 3 seed, dataset nhe -> nang
python -u utils/run_dacd_experiments.py --grid full --variants GTR,DACD-F1 --workers $WORKERS \
  --only ETTh1,ETTh2,ETTm1,ETTm2,Weather,Solar,PEMS08,PEMS04,PEMS03,Electricity
python -u utils/run_dacd_experiments.py --grid full --variants GTR,DACD-F1 --workers 1 --only Traffic,PEMS07

# Dot 2 (neu con thoi gian): DACD-F2 va DACD (chi chon chu ky + bo chon, khong co mau tuong lai)
python -u utils/run_dacd_experiments.py --grid full --variants DACD-F2,DACD --workers $WORKERS --skip Traffic,PEMS07
python -u utils/run_dacd_experiments.py --grid full --variants DACD-F2,DACD --workers 1 --only Traffic,PEMS07

python utils/summarize_full.py

# Hướng dẫn chạy thực nghiệm DACD trên 12 dataset (máy có GPU)

Mục tiêu: so sánh **GTR gốc** (chu kỳ tác giả đặt tay) với **DACD** (tự phát hiện chu kỳ + mẫu chu kỳ đặt đúng pha lên chân trời dự báo) trên đủ 12 dataset của bài báo, cùng siêu tham số của tác giả (`scripts/GTR/*.sh`), 3 seed (2024, 2025, 2026), look-back 96.

## 1. Cài đặt

```bash
pip install -r requirements.txt        # torch bản CUDA: cài theo https://pytorch.org nếu chưa có
python -c "import torch; print(torch.cuda.is_available())"   # phải in True
```

`run.py` tự dùng GPU `cuda:0` khi có.

## 2. Dữ liệu

Đặt **tất cả file phẳng** trong `GTR-main/dataset/` (không tạo thư mục con), đúng tên sau:

| Dataset | File | Nguồn |
|---|---|---|
| ETTh1, ETTh2, ETTm1, ETTm2 | `ETTh1.csv` … `ETTm2.csv` | gói dataset của Autoformer / iTransformer |
| Weather | `weather.csv` | như trên |
| Electricity | `electricity.csv` | như trên |
| Traffic | `traffic.csv` | như trên |
| Solar-Energy | `solar_AL.txt` | gói dataset của iTransformer (thư mục `Solar/`) |
| PEMS03/04/07/08 | `PEMS03.npz` … `PEMS08.npz` | gói dataset của iTransformer (thư mục `PEMS/`) |

Gói dataset đầy đủ: link Google Drive trong README của repo `thuml/iTransformer` (hoặc `thuml/Time-Series-Library`). Thư mục `dataset/` nằm trong `.gitignore`, nên không đẩy lên git.

## 3. Chạy

Từ thư mục `GTR-main`:

```bash
bash scripts/DACD/run_full.sh            # WORKERS=3 bash ... để chạy 3 job song song trên 1 GPU
```

Script chạy theo thứ tự ưu tiên:

1. **Đợt 1:** `GTR` vs `DACD-F1` trên mọi dataset. Traffic và PEMS07 (≈870 kênh) chạy riêng với 1 worker.
2. **Đợt 2:** `DACD-F2` và `DACD`, nếu còn thời gian.

Có thể dừng (Ctrl+C) và chạy lại bất cứ lúc nào: các lần đã xong được ghi ở `results_dacd/full/runs.jsonl` và sẽ được bỏ qua. Chạy riêng một phần:

```bash
python utils/run_dacd_experiments.py --grid full --only Solar,Weather --variants GTR,DACD-F1 --workers 2
python utils/run_dacd_experiments.py --grid full --only Traffic --seeds 2024 --workers 1
```

Mỗi lần chạy có log ở `results_dacd/full/logs/<dataset>_<H>_<variant>_s<seed>.log`. Lần nào lỗi sẽ được in ra là `FAILED`, xem log tương ứng.

## 4. Các biến thể

| Tên | Mô tả | Cờ trong `run.py` |
|---|---|---|
| `GTR` | GTR gốc, chu kỳ tác giả (24/96/144/168/288) | `--model GTR --cycle L` |
| `DACD` | tự phát hiện chu kỳ bằng ACF trên tập train + bộ chọn per-sample (v2) | `--model GTRDACD --cycles auto --dacd_nested 1` |
| `DACD-F1` | DACD + bảng mẫu chu kỳ R đặt đúng pha lên chân trời dự báo | thêm `--dacd_future 1` |
| `DACD-F2` | DACD-F1 + trừ mẫu chu kỳ khỏi cửa sổ đầu vào (dự báo phần dư) | thêm `--dacd_future 2` |

## 5. Gửi kết quả

```bash
python utils/summarize_full.py           # in bảng, ghi results_dacd/full/summary.csv
```

Gửi lại **cả thư mục `results_dacd/full/`** (`runs.jsonl`, `summary.csv`, `logs/`). Không cần gửi `checkpoints/` (nặng).

Bảng in ra có các cột sau:
- `d_mse`, `d_mae`: chênh lệch so với GTR **cùng seed**. Giá trị âm nghĩa là DACD tốt hơn.
- `wins`: số seed mà DACD có MSE thấp hơn GTR.
- `best_by_vali`: biến thể DACD được chọn theo validation loss, không nhìn tập test.

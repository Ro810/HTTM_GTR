# GTR-APCM: hướng dẫn chạy và tiêu chí quyết định

> Giao thức hiện tại (`val_mse_v2`): loss FD5L1 vẫn dùng để cập nhật trọng số,
> nhưng early stopping, checkpoint và chọn biến thể đều dùng **validation MSE**.
> Test chỉ chạy đúng một lần sau khi đã nạp checkpoint tốt nhất. Kết quả mới
> được tách vào `results_apcm_mse/`; `results_apcm/` là kết quả giao thức cũ,
> không được trộn vào bảng cuối.

`GTRAPCM` giữ nguyên đường dự báo của GTR và thêm **Adaptive Partial Channel Mixing** (APCM):

- chia cửa sổ thành patch không chồng lấn;
- dùng giá trị thời gian và biên độ FFT để định tuyến;
- gom kênh vào nhiều latent group theo từng sample/patch;
- chỉ thêm một residual correction vào biểu diễn GTR;
- khởi tạo correction bằng 0, vì vậy mô hình bắt đầu đúng tại baseline GTR.

APCM khác GTA của tác giả: GTA nén tất cả kênh thành một token rồi broadcast lại, còn APCM dùng nhiều nhóm động và giữ correction riêng cho từng kênh.

## 1. Kiểm tra cài đặt

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest tests/test_gtr_apcm.py -q
```

## 2. Sàng lọc nhanh

Chạy tuần tự GTR, GTA và APCM trên Electricity-96 và PEMS08-12, seed 2024:

```bash
python utils/run_apcm_experiments.py --stage screen
```

Kết quả nằm trong `results_apcm_mse/runs.jsonl`, log đầy đủ nằm trong
`results_apcm_mse/logs/`. Có thể dừng và chạy lại; job hoàn thành sẽ được bỏ qua.

Tổng hợp MSE/MAE, phần trăm cải thiện ghép cặp cùng seed và số seed thắng GTR:

```bash
python utils/summarize_apcm.py
```

Nếu máy yếu, chạy riêng Electricity:

```bash
python utils/run_apcm_experiments.py --stage screen --only Electricity
```

## 3. Xác nhận ba seed

Chỉ mở rộng khi APCM thắng GTR ở bước sàng lọc:

```bash
python utils/run_apcm_experiments.py --stage full --only Electricity,Traffic --variants GTR,GTA,APCM
```

Xác nhận riêng một horizon trước khi chạy toàn bộ lưới:

```bash
python utils/run_apcm_experiments.py --stage full --only PEMS08 \
  --pred_lens 12 --variants GTR,APCM --seeds 2024,2025,2026
```

Để chạy ablation không dùng FFT:

```bash
python utils/run_apcm_experiments.py --stage full --only Electricity --variants APCM,APCM-noFFT
```

## 4. Lệnh chạy trực tiếp

```bash
python -u run.py \
  --is_training 1 --root_path ./dataset/ --data_path electricity.csv \
  --model_id Electricity_96_96_APCM --model GTRAPCM --data custom \
  --features M --seq_len 96 --pred_len 96 --enc_in 321 --cycle 168 \
  --apcm_patch_len 16 --apcm_dim 32 --apcm_latents 8 \
  --apcm_dropout 0.1 --apcm_temperature 0.2 --apcm_spectral 1 \
  --train_epochs 30 --patience 5 --batch_size 32 --learning_rate 0.003 \
  --random_seed 2024 --num_workers 0
```

## 5. Quy tắc báo cáo

So sánh tối thiểu:

1. `GTR`: mô hình chính của tác giả.
2. `GTA`: inter-channel baseline trong phụ lục tác giả (`--model GTR --gtr_agg 1`).
3. `GTRAPCM`: phương pháp đề xuất.

Không kết luận từ một seed. Báo trung bình và độ lệch chuẩn của MSE/MAE qua seed 2024, 2025, 2026. Chỉ tuyên bố cải thiện ổn định nếu APCM tốt hơn GTR ở ít nhất 2/3 seed và MSE trung bình thấp hơn độ dao động giữa các seed.

Ba ứng viên `GTR-FD5L1`, `APCM-FD5L1`, `APCM-CE-FD5L1` chỉ được so trên
cùng tập seed. Lệnh chọn cuối là:

```powershell
python utils/select_apcm_plus.py --loss FD5L1 --selection_seeds 2024,2025,2026
```

Nếu một ứng viên thiếu seed, cấu hình đó bị bỏ qua và in `SKIP`, thay vì lấy
trung bình lệch số lần chạy. Cột `selection_seeds` và `report_seeds` trong CSV
cho biết chính xác seed nào đã dùng.

Trên Windows PowerShell, chạy các bộ nhanh trước (không có Traffic) bằng:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/APCM_PLUS/run_all.ps1 stage1
powershell -ExecutionPolicy Bypass -File scripts/APCM_PLUS/run_all.ps1 stage2
```

Chạy lại cùng lệnh sẽ tự bỏ qua job đã có trong `runs.jsonl`. Traffic để sau:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/APCM_PLUS/run_all.ps1 stage3
```

## 6. Thử nghiệm mới (12/11/2026): APCM-NE và tắt RevIN

**APCM-NE** (`--apcm_node_emb 1`): thêm cho mỗi kênh một vector định danh học được, cộng một điểm "ái lực tĩnh" giữa kênh và nhóm vào bước gom nhóm. APCM gốc chỉ nhóm kênh theo hình dạng patch nên không học được quan hệ cố định giữa các cảm biến. Trọng số của phần tĩnh khởi tạo bằng 0, nên lúc đầu APCM-NE trùng APCM (đã có test). Chi phí thêm: C × 32 + 1 tham số.

**GTR-noRevIN, APCM-noRevIN**: tắt RevIN theo ablation của chính tác giả (Bảng 8, trang 18), nơi Electricity cho MSE tốt hơn khi tắt RevIN. Phải chạy cả GTR và APCM để so công bằng.

Kiểm tra cài đặt:

```bash
python -m pytest tests/test_gtr_apcm.py -q   # 5 test
```

Sàng lọc 1 seed (GTR và APCM seed 2024 đã có trong runs.jsonl nên không chạy lại):

```bash
# APCM-NE: PEMS04 và PEMS08 ở horizon 12, 96; Traffic và Electricity ở horizon 96 (6 lần chạy)
python utils/run_apcm_experiments.py --stage full --only PEMS04,PEMS08,Traffic,Electricity \
  --pred_lens 12,96 --seeds 2024 --variants APCM-NE

# Tắt RevIN trên Electricity, horizon 96 và 336 (4 lần chạy)
python utils/run_apcm_experiments.py --stage full --only Electricity \
  --pred_lens 96,336 --seeds 2024 --variants GTR-noRevIN,APCM-noRevIN

python utils/summarize_apcm.py
```

Quy tắc dừng, đặt trước khi chạy:

- APCM-NE chỉ chạy đủ 3 seed nếu có MSE thấp hơn APCM (cùng seed 2024) ở ít nhất 4/5 cấu hình PEMS04, PEMS08 và Traffic. Ngược lại thì bỏ.
- Tắt RevIN chỉ dùng nếu APCM-noRevIN thấp hơn APCM ở cả hai horizon. Khi báo cáo phải ghi rõ đây là đổi cấu hình theo Bảng 8, không phải cải tiến phương pháp.

Ước lượng thời gian trên GPU (theo thời gian/epoch cũ): PEMS khoảng 3–5 phút mỗi lần, Electricity khoảng 10 phút, Traffic-96 khoảng 1,5 giờ.

## 7. Tinh chỉnh theo từng dataset (22/11/2026)

Mục tiêu: đưa ETTm1, ETTm2, PEMS03, Electricity và Weather lên mức giảm MSE
trung bình ít nhất 0.01 so với GTR chạy lại cùng seed. Mỗi dataset chọn **một**
phương án cho cả bốn horizon, chỉ dựa trên validation MSE.

Phương án mới trong `utils/run_apcm_experiments.py`:

| Hướng | Phương án | Cờ thêm | Dataset |
|---|---|---|---|
| Hàm loss | `GTR-MAE`, `GTR-HY`, `GTR-FD8`, `APCM-CE-FD8`, `APCM-CE-HY` (đã có sẵn) | | ETTm1, ETTm2, Weather |
| Learning rate | `GTR-FD5L1`, `APCM-FD5L1`, `APCM-CE-FD5L1` kèm hậu tố `-LR5e4` hoặc `-LR2e3` | `--learning_rate 0.0005` / `0.002` | ETTm1, ETTm2, Weather |
| Tắt RevIN | `GTR-FD5L1-noRevIN`, `APCM-CE-FD5L1-noRevIN` | `--use_revin 0` | Electricity |
| Sức chứa APCM | `APCM-FD5L1-K4`, `APCM-FD5L1-K16`, `APCM-FD5L1-P8` | `--apcm_latents 4/16`, `--apcm_patch_len 8` | PEMS03 |
| A. Chu kỳ tương lai | hậu tố `-FC` trên GTR-FD5L1, APCM-FD5L1, APCM-CE-FD5L1 | `--future_cycle 1` | cả 5 dataset |
| B. Trung bình trọng số | hậu tố `-EMA`, kết hợp `-FC-EMA` | `--ema_decay 0.999` | cả 5 dataset |

GTR-FD5L1 cũng được thử learning rate để baseline có cùng ngân sách tinh chỉnh.

**A. Chu kỳ tương lai (`--future_cycle 1`).** GTR gốc chỉ tra bảng chu kỳ `Q` cho
cửa sổ đầu vào. Khi bật cờ, mô hình trừ `Q` của cửa sổ đầu vào khỏi input (chỉ
học phần dư không theo chu kỳ), rồi cộng `Q` tại các vị trí
`cycle_index + seq_len … + seq_len + pred_len` vào đầu ra trước khi đảo RevIN.
Vì `Q` khởi tạo bằng 0, mô hình lúc bắt đầu trùng GTR (có test), và không thêm
tham số. Ý tưởng tương tự CycleNet (NeurIPS 2024).

**B. EMA (`--ema_decay 0.999`).** Giữ trung bình động của trọng số, hệ số khởi
động `min(0.999, (1+n)/(10+n))`. Validation, checkpoint và test dùng trọng số
EMA; vẫn là một mô hình, không phải ensemble.
`GTR-noRevIN` chỉ để tham khảo, không phải ứng viên.

Chạy (tự bỏ qua job đã có trong `runs.jsonl`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/APCM_TUNING/run_all.ps1 stage1  # ETTm1, ETTm2: 408 job, ~7,5 giờ
powershell -ExecutionPolicy Bypass -File scripts/APCM_TUNING/run_all.ps1 stage2  # PEMS03: 149 job, ~12,5 giờ
powershell -ExecutionPolicy Bypass -File scripts/APCM_TUNING/run_all.ps1 stage3  # Electricity: 148 job, ~20 giờ
powershell -ExecutionPolicy Bypass -File scripts/APCM_TUNING/run_all.ps1 stage4  # Weather: 204 job, ~7 giờ
powershell -ExecutionPolicy Bypass -File scripts/APCM_TUNING/run_all.ps1 select  # chỉ chạy lại bước chọn
```

Trên Linux/macOS dùng `bash scripts/APCM_TUNING/run_all.sh stage1` với cùng tên stage.

Chọn theo dataset (script tự gọi sau mỗi stage):

```bash
python utils/select_apcm_plus.py --per_dataset --only ETTm1,ETTm2 \
  --candidates <danh sách ứng viên> --selection_seeds 2024,2025,2026 --tag ett
```

Điểm của mỗi ứng viên là trung bình qua các horizon của
(validation MSE ứng viên ÷ validation MSE ứng viên đầu tiên). Ứng viên có điểm
thấp nhất được chọn. Test MSE/MAE chỉ được đọc sau khi đã chọn. Kết quả nằm ở
`results_apcm_mse/per_dataset_<tag>.txt` và `.csv`. Cột `dMSE/GTR` là mức giảm
MSE trung bình so với GTR cùng seed, cột cuối đánh dấu có đạt 0.01 hay không.

Quy tắc đặt trước khi chạy:

- Danh sách ứng viên cố định như trong `scripts/APCM_TUNING/run_all.ps1`, không thêm bớt sau khi thấy kết quả test.
- Chỉ báo cáo phương án do validation chọn, đủ 3 seed và 4 horizon.
- Tắt RevIN phải ghi rõ là đổi cấu hình theo Bảng 8 của tác giả.

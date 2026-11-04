# GTR-APCM: hướng dẫn chạy và tiêu chí quyết định

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

Kết quả nằm trong `results_apcm/runs.jsonl`, log đầy đủ nằm trong `results_apcm/logs/`. Có thể dừng và chạy lại; job hoàn thành sẽ được bỏ qua.

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

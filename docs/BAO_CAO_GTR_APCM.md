# Báo cáo cải tiến GTR bằng mô hình hóa quan hệ giữa các kênh

## 1. Tóm tắt ngắn

Nhóm ban đầu cải tiến GTR bằng cách tự động phát hiện chu kỳ (DACD). Tuy nhiên, kết quả DACD không ổn định: có tập dữ liệu tốt hơn rất ít, có tập lại kém hơn GTR gốc. Vì vậy, nhóm chuyển sang xử lý một hạn chế khác của GTR: **GTR chủ yếu dự báo từng kênh độc lập và chưa khai thác tốt quan hệ giữa các cảm biến**.

Nhóm đề xuất **GTR-APCM (Adaptive Partial Channel Mixing)**. APCM chia cửa sổ đầu vào thành các đoạn nhỏ, tự nhóm những kênh có biểu hiện giống nhau trong từng đoạn, sau đó bổ sung thông tin liên kênh vào GTR bằng một kết nối residual an toàn.

Kết quả bước đầu trên PEMS08 cho thấy APCM tốt hơn GTR ở cả bốn độ dài dự báo và cả ba seed. Tổng cộng, APCM thắng **12/12 phép so sánh ghép cặp**.

## 2. Vì sao nhóm đổi hướng cải tiến?

### 2.1. Hạn chế của hướng phát hiện chu kỳ

Kết quả DACD trước đây cho thấy việc tìm chu kỳ tự động chưa tạo ra mức cải thiện đủ lớn và ổn định:

- Có trường hợp chu kỳ tìm được gần giống chu kỳ tác giả đặt tay, nên mô hình gần như không thay đổi.
- Thêm nhiều chu kỳ có thể đưa thêm nhiễu vào mô hình.
- Bộ chọn chu kỳ chưa chắc học được cách loại bỏ chu kỳ không hữu ích.
- Mức cải thiện tốt nhất chỉ khoảng dưới 1% trên Electricity, trong khi một số tập ETT lại kém hơn GTR.

Do đó, tiếp tục tối ưu phát hiện chu kỳ có rủi ro cao nhưng khả năng tăng chỉ số không lớn.

### 2.2. Hạn chế còn lại của GTR

GTR học một biểu diễn chu kỳ toàn cục cho mỗi biến, nhưng phần trích xuất đặc trưng hoạt động gần như độc lập trên từng kênh. Điều này chưa phù hợp với dữ liệu giao thông, nơi các cảm biến có liên hệ mạnh với nhau.

Ví dụ, khi lưu lượng xe tại cảm biến A tăng, một khoảng thời gian sau cảm biến B ở đoạn đường kế tiếp cũng có thể tăng. Nếu chỉ nhìn từng cảm biến riêng lẻ, mô hình bỏ qua tín hiệu quan trọng này.

Bài báo GTR cũng thừa nhận điểm yếu này và thử GTA, một phương pháp nén tất cả kênh thành một global token. Tuy nhiên, một token duy nhất có thể làm mất cấu trúc nhóm và làm các kênh trở nên quá giống nhau.

## 3. Phương pháp đề xuất: GTR-APCM

APCM là viết tắt của **Adaptive Partial Channel Mixing**, tức là “trộn kênh một phần và thích nghi theo dữ liệu”.

### 3.1. Ý tưởng trực quan

Thay vì trộn toàn bộ cảm biến vào một đại diện chung, APCM thực hiện ba việc:

1. Chia cửa sổ thời gian thành các patch nhỏ.
2. Trong từng patch, tự nhóm những cảm biến có đặc điểm tương tự.
3. Chỉ truyền thông tin trong các nhóm phù hợp và giữ lại đường dự báo GTR gốc.

Có thể hình dung như sau:

```text
Dữ liệu đầu vào
      │
      ├── GTR gốc ──────────────────────────────┐
      │                                          ├── Cộng residual → dự báo
      └── APCM: patch → nhóm kênh → hiệu chỉnh ─┘
```

### 3.2. Các thành phần chính

**Patch thời gian:** Cửa sổ đầu vào dài 96 được chia thành các patch không chồng lấn, mặc định mỗi patch dài 16 bước. Điều này giúp mô hình nhận ra quan hệ kênh có thể thay đổi theo thời gian.

**Thông tin miền thời gian và tần số:** Ngoài giá trị gốc, APCM sử dụng biên độ FFT của từng patch. Nhờ đó, mô hình có thể nhóm những cảm biến có dạng dao động tương tự ngay cả khi giá trị tuyệt đối khác nhau.

**Nhiều latent group:** APCM mặc định sử dụng tám nhóm ẩn. Mỗi kênh được gán mềm vào các nhóm theo từng sample và từng patch. Cách này giữ được nhiều kiểu quan hệ hơn GTA, vốn chỉ sử dụng một global token.

**Gate động:** Gate quyết định mức độ sử dụng thông tin liên kênh. Nếu việc trộn kênh không hữu ích tại một patch, gate có thể giảm ảnh hưởng của nhánh APCM.

**Residual an toàn:** Đầu ra APCM được cộng vào biểu diễn GTR:

\[
Z = Z_{GTR} + \Delta Z_{APCM}
\]

Lớp cuối của APCM được khởi tạo bằng 0. Vì vậy, tại thời điểm bắt đầu huấn luyện, GTR-APCM cho kết quả đúng bằng GTR gốc. Mô hình chỉ rời khỏi baseline khi học được phần hiệu chỉnh có ích.

### 3.3. Chi phí tính toán

APCM không dùng attention giữa mọi cặp kênh. Nếu số kênh là `C` và số nhóm ẩn là `K`, chi phí trộn kênh tăng gần tuyến tính theo `C × K`, thay vì `C²`. Thiết kế này phù hợp với Traffic và PEMS, nơi số cảm biến có thể lên đến hàng trăm.

## 4. Thiết lập thực nghiệm

- Dataset: **PEMS08**, gồm 170 cảm biến giao thông.
- Độ dài đầu vào: 96.
- Độ dài dự báo: 12, 24, 48 và 96.
- Seed: 2024, 2025 và 2026.
- Baseline: GTR với chu kỳ 288 như cấu hình tác giả.
- Các siêu tham số huấn luyện khác được giữ giống GTR: learning rate 0.003, batch size 32, tối đa 30 epoch và early stopping.
- Chỉ số đánh giá: MSE và MAE; giá trị càng thấp càng tốt.

Mọi phép so sánh đều ghép cặp theo cùng seed để hạn chế ảnh hưởng của khởi tạo ngẫu nhiên.

## 5. Kết quả thực nghiệm

Kết quả dưới đây là trung bình ± độ lệch chuẩn qua ba seed.

| Horizon | GTR MSE | APCM MSE | MSE giảm | GTR MAE | APCM MAE | MAE giảm | Số seed thắng |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 12 | 0.07050 ± 0.00019 | **0.06758 ± 0.00052** | **4.15%** | 0.16842 | **0.16406** | **2.59%** | 3/3 |
| 24 | 0.09585 ± 0.00016 | **0.08683 ± 0.00060** | **9.41%** | 0.19452 | **0.18399** | **5.41%** | 3/3 |
| 48 | 0.15188 ± 0.00206 | **0.12394 ± 0.00124** | **18.40%** | 0.24078 | **0.21821** | **9.37%** | 3/3 |
| 96 | 0.26649 ± 0.01195 | **0.20677 ± 0.00524** | **22.41%** | 0.30543 | **0.26835** | **12.14%** | 3/3 |

### 5.1. Nhận xét chính

**APCM thắng ổn định:** APCM tốt hơn GTR ở cả 4 horizon và cả 3 seed, tương ứng 12/12 phép so sánh ghép cặp.

**Lợi ích tăng khi dự báo xa:** Mức giảm MSE tăng từ 4.15% ở horizon 12 lên 22.41% ở horizon 96. Điều này cho thấy thông tin giữa các cảm biến đặc biệt hữu ích khi phải dự báo nhiều bước trong tương lai.

**Không chỉ cải thiện MSE:** MAE cũng giảm ở toàn bộ horizon. Vì vậy, kết quả không phải do mô hình chỉ sửa một số ít sai số lớn.

**Độ ổn định tốt:** Độ lệch chuẩn của APCM nhỏ ở cả bốn horizon và APCM thắng từng seed, không chỉ thắng khi lấy trung bình.

### 5.2. So sánh nhanh với GTA

Ở PEMS08, horizon 12, seed 2024:

| Phương pháp | MSE | MAE |
|---|---:|---:|
| GTR | 0.07072 | 0.16820 |
| GTA của tác giả | 0.07715 | 0.17595 |
| **APCM đề xuất** | **0.06776** | **0.16417** |

GTA làm kết quả kém hơn GTR trên phép thử này, trong khi APCM cải thiện cả MSE và MAE. Kết quả phù hợp với giả thuyết rằng ép tất cả cảm biến vào một token duy nhất có thể gây mất thông tin, còn APCM giữ được nhiều nhóm quan hệ khác nhau.

Lưu ý: GTA hiện mới được chạy một seed tại horizon 12 nên chỉ được xem là kết quả tham khảo, chưa phải kết luận tổng quát.

## 6. Vì sao APCM có thể cải thiện mạnh ở horizon dài?

Khi horizon ngắn, lịch sử gần nhất của chính cảm biến đã chứa nhiều thông tin, nên quan hệ giữa các cảm biến chỉ giúp thêm một phần nhỏ.

Khi horizon dài, sai số của dự báo độc lập dễ tích lũy. Quan hệ giữa các cảm biến cung cấp thêm tín hiệu về trạng thái chung của mạng lưới giao thông. APCM khai thác tín hiệu này theo từng patch và từng nhóm, giúp phần hiệu chỉnh ngày càng có giá trị khi horizon tăng.

## 7. Điểm mới so với GTR và GTA

| Đặc điểm | GTR | GTA | APCM |
|---|:---:|:---:|:---:|
| Mô hình hóa chu kỳ toàn cục | Có | Có | Có |
| Khai thác quan hệ giữa các kênh | Không đáng kể | Có | Có |
| Số đại diện liên kênh | Không có | 1 token | Nhiều latent group |
| Thay đổi theo sample/patch | Không | Hạn chế | Có |
| Giữ hiệu chỉnh riêng cho từng kênh | Có | Không rõ, broadcast chung | Có |
| Có đường residual khởi tạo đúng bằng GTR | Không áp dụng | Không | Có |
| Độ phức tạp theo số kênh | Tuyến tính | Tuyến tính | Tuyến tính theo `C × K` |

## 8. Giới hạn của kết quả hiện tại

Kết quả hiện tại rất tích cực nhưng chưa đủ để kết luận APCM tốt trên mọi bài toán:

- Mới xác nhận đầy đủ trên PEMS08.
- Chưa chạy đầy đủ Electricity, Traffic và các PEMS còn lại.
- Chưa có ablation ba seed để tách đóng góp của FFT, số latent group và gate.
- Chưa tổng hợp số tham số, thời gian huấn luyện và bộ nhớ so với GTR.
- GTA mới được chạy một seed tại một horizon.

Vì vậy, cách phát biểu an toàn là: **“APCM cải thiện ổn định GTR trên PEMS08 ở toàn bộ bốn horizon”**, chưa nên nói “APCM luôn tốt hơn GTR trên mọi dataset”.

## 9. Thực nghiệm cần làm tiếp

Ưu tiên theo thứ tự:

1. Chạy Electricity-96 với GTR, GTA và APCM để kiểm tra khả năng tổng quát sang dữ liệu điện.
2. Chạy PEMS03 hoặc PEMS04 để xác nhận trên một mạng cảm biến giao thông khác.
3. Chạy Traffic nếu GPU đủ bộ nhớ; đây là dataset thể hiện rõ hạn chế inter-channel của GTR.
4. Chạy ablation `APCM-noFFT` để chứng minh phần miền tần số có đóng góp.
5. So sánh số tham số, thời gian mỗi epoch và VRAM.

## 10. Đoạn trình bày ngắn với giảng viên

> Ban đầu nhóm cải tiến GTR bằng phát hiện chu kỳ tự động, nhưng kết quả không ổn định vì chu kỳ tác giả chọn đã khá tốt. Nhóm chuyển sang một hạn chế khác được chính bài báo đề cập: GTR xử lý các kênh gần như độc lập nên chưa khai thác quan hệ giữa các cảm biến. Nhóm đề xuất APCM, chia dữ liệu thành các patch và tự nhóm các cảm biến có hành vi tương tự vào nhiều nhóm ẩn. Khác với GTA dùng một global token, APCM giữ nhiều nhóm động và chỉ thêm một residual correction vào GTR. Trên PEMS08, APCM giảm MSE lần lượt 4.15%, 9.41%, 18.40% và 22.41% ở bốn horizon, đồng thời thắng cả ba seed trong mọi trường hợp. Kết quả cho thấy mô hình hóa quan hệ cảm biến có hiệu quả rõ hơn phát hiện lại chu kỳ, đặc biệt ở dự báo dài hạn.

## 11. Tệp triển khai và dữ liệu kết quả

- Mô hình: [`models/GTRAPCM.py`](../models/GTRAPCM.py)
- Tham số dòng lệnh: [`run.py`](../run.py)
- Bộ chạy thực nghiệm: [`utils/run_apcm_experiments.py`](../utils/run_apcm_experiments.py)
- Tổng hợp kết quả: [`utils/summarize_apcm.py`](../utils/summarize_apcm.py)
- Kết quả thô: [`results_apcm/runs.jsonl`](../results_apcm/runs.jsonl)
- Bảng tổng hợp: [`results_apcm/summary.csv`](../results_apcm/summary.csv)
- Hướng dẫn chạy: [`docs/APCM_RUN_GUIDE.md`](APCM_RUN_GUIDE.md)

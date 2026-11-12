# Báo cáo cải tiến GTR bằng mô hình hóa quan hệ giữa các kênh

## 1. Tóm tắt ngắn

Nhóm ban đầu cải tiến GTR bằng cách tự động phát hiện chu kỳ (DACD). Tuy nhiên, kết quả DACD không ổn định: có tập dữ liệu tốt hơn rất ít, có tập lại kém hơn GTR gốc. Vì vậy, nhóm chuyển sang xử lý một hạn chế khác của GTR: **GTR chủ yếu dự báo từng kênh độc lập và chưa khai thác tốt quan hệ giữa các cảm biến**.

Nhóm đề xuất **GTR-APCM (Adaptive Partial Channel Mixing)**. APCM chia cửa sổ đầu vào thành các đoạn nhỏ, tự nhóm những kênh có biểu hiện giống nhau trong từng đoạn, sau đó bổ sung thông tin liên kênh vào GTR bằng một kết nối residual an toàn.

Thực nghiệm đầy đủ trên sáu bộ dữ liệu gồm PEMS03, PEMS04, PEMS07, PEMS08, Electricity và Traffic cho thấy APCM có MSE trung bình tốt hơn GTR ở cả **24/24 cấu hình dataset–horizon**. Tổng cộng, APCM thắng **71/72 phép so sánh ghép cặp theo seed**. Mức giảm MSE lớn nhất đạt **22.27%** trên PEMS08-96; trên Traffic, APCM giảm từ **5.14% đến 8.02%** và thắng đủ 12/12 phép so sánh.

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

### 3.3. Thuật toán từng bước

Ký hiệu đầu vào là `X` có kích thước `(B, T, C)`, trong đó `B` là batch size, `T` là độ dài lịch sử và `C` là số kênh.

1. **Chuẩn hóa:** RevIN chuẩn hóa riêng từng sample và từng kênh.
2. **Trích xuất chu kỳ bằng GTR:** GTR kết hợp đầu vào với bảng chu kỳ học được `Q` để tạo thông tin toàn cục, sau đó tính `H_GTR = X + GlobalInformation`.
3. **Chia patch:** Chuyển `H_GTR` về dạng `(B, C, T)` và chia thành các patch dài 16. Nếu `T` không chia hết, mô hình padding bằng cách lặp giá trị cuối.
4. **Tạo đặc trưng kép:** Với mỗi patch, nối giá trị miền thời gian với biên độ phổ `log(1 + |rFFT|)`.
5. **Mã hóa:** Đưa vector trên qua `Linear → GELU → LayerNorm` để thu được đặc trưng `h` có 32 chiều.
6. **Gán mềm vào nhóm:** Tính cosine similarity giữa `h` và 8 latent query học được, chia cho temperature rồi softmax theo chiều kênh. Một kênh có thể đóng góp vào nhiều nhóm.
7. **Tạo latent token:** Mỗi nhóm lấy tổng có trọng số của các kênh: `L_k = Σ_c a_(c,k) h_c`.
8. **Đọc thông tin nhóm:** Softmax theo chiều nhóm xác định mỗi kênh cần đọc bao nhiêu thông tin từ từng latent token: `M_c = Σ_k b_(c,k) L_k`.
9. **Gate và giải mã:** Tính `Δh = (M - h) × sigmoid(Gate(h))`, sau đó decoder đổi `Δh` về độ dài patch và ghép các patch lại thành `ΔH_APCM`.
10. **Cộng residual và dự báo:** Tính `H_enhanced = H_GTR + ΔH_APCM`, đưa qua các lớp dự báo của GTR, rồi đảo chuẩn hóa RevIN để nhận kết quả cuối.

Chuỗi thao tác ngắn gọn để ghi nhớ:

> **Chia patch → lấy time + FFT → mã hóa → gán kênh vào nhiều nhóm → gom thông tin nhóm → kênh đọc lại → gate lọc → cộng residual vào GTR.**

Lớp decoder cuối của APCM được khởi tạo bằng 0. Vì vậy lúc mới bắt đầu huấn luyện, `ΔH_APCM = 0` và toàn bộ mô hình hoạt động đúng như GTR; APCM chỉ tạo ảnh hưởng sau khi học được hiệu chỉnh có ích.

### 3.4. Chi phí tính toán

APCM không dùng attention giữa mọi cặp kênh. Nếu số kênh là `C` và số nhóm ẩn là `K`, chi phí trộn kênh tăng gần tuyến tính theo `C × K`, thay vì `C²`. Thiết kế này phù hợp với Traffic và PEMS, nơi số cảm biến có thể lên đến hàng trăm.

## 4. Thiết lập thực nghiệm

- Dataset: **PEMS03, PEMS04, PEMS07, PEMS08, Electricity và Traffic**.
- Độ dài đầu vào: 96.
- Độ dài dự báo: PEMS dùng 12, 24, 48, 96; Electricity và Traffic dùng 96, 192, 336, 720.
- Seed: 2024, 2025 và 2026.
- Baseline: GTR với chu kỳ do tác giả đặt: 288 cho PEMS và 168 cho Electricity/Traffic.
- Các siêu tham số huấn luyện được giữ giống cấu hình GTR tương ứng: tối đa 30 epoch, early stopping; batch size 32 cho PEMS/Electricity và 16 cho Traffic.
- Chỉ số đánh giá: MSE và MAE; giá trị càng thấp càng tốt.

Mọi phép so sánh đều ghép cặp theo cùng seed để hạn chế ảnh hưởng của khởi tạo ngẫu nhiên.

## 5. Kết quả thực nghiệm trên PEMS series

Kết quả dưới đây là trung bình ± độ lệch chuẩn qua ba seed.

### 5.1. PEMS03

| Horizon | GTR MSE | APCM MSE | MSE giảm | Seed thắng |
|---:|---:|---:|---:|:---:|
| 12 | 0.05691 ± 0.00013 | **0.05564 ± 0.00051** | **2.22%** | 3/3 |
| 24 | 0.07060 ± 0.00030 | **0.06854 ± 0.00028** | **2.91%** | 3/3 |
| 48 | 0.09436 ± 0.00062 | **0.09145 ± 0.00053** | **3.09%** | 3/3 |
| 96 | 0.12927 ± 0.00029 | **0.12505 ± 0.00318** | **3.26%** | 3/3 |

### 5.2. PEMS04

| Horizon | GTR MSE | APCM MSE | MSE giảm | Seed thắng |
|---:|---:|---:|---:|:---:|
| 12 | 0.06577 ± 0.00018 | **0.06349 ± 0.00012** | **3.46%** | 3/3 |
| 24 | 0.07588 ± 0.00049 | **0.07158 ± 0.00019** | **5.67%** | 3/3 |
| 48 | 0.09459 ± 0.00122 | **0.08503 ± 0.00056** | **10.11%** | 3/3 |
| 96 | 0.11700 ± 0.00093 | **0.09890 ± 0.00218** | **15.48%** | 3/3 |

### 5.3. PEMS07

| Horizon | GTR MSE | APCM MSE | MSE giảm | Seed thắng |
|---:|---:|---:|---:|:---:|
| 12 | 0.05175 ± 0.00010 | **0.04987 ± 0.00016** | **3.63%** | 3/3 |
| 24 | 0.06404 ± 0.00028 | **0.05948 ± 0.00015** | **7.12%** | 3/3 |
| 48 | 0.08271 ± 0.00055 | **0.07477 ± 0.00183** | **9.60%** | 3/3 |
| 96 | 0.11053 ± 0.00050 | **0.09040 ± 0.00449** | **18.22%** | 3/3 |

### 5.4. PEMS08

| Horizon | GTR MSE | APCM MSE | MSE giảm | GTR MAE | APCM MAE | MAE giảm | Số seed thắng |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 12 | 0.07050 ± 0.00019 | **0.06758 ± 0.00052** | **4.15%** | 0.16842 | **0.16406** | **2.59%** | 3/3 |
| 24 | 0.09585 ± 0.00016 | **0.08683 ± 0.00060** | **9.41%** | 0.19452 | **0.18399** | **5.41%** | 3/3 |
| 48 | 0.15188 ± 0.00206 | **0.12394 ± 0.00124** | **18.38%** | 0.24078 | **0.21821** | **9.37%** | 3/3 |
| 96 | 0.26649 ± 0.01195 | **0.20677 ± 0.00524** | **22.27%** | 0.30543 | **0.26835** | **12.14%** | 3/3 |

### 5.5. Nhận xét chung PEMS

**APCM thắng ổn định:** APCM tốt hơn GTR trong toàn bộ 16 cấu hình horizon của bốn bộ PEMS và thắng **48/48** phép so sánh ghép cặp theo seed.

**Lợi ích thường tăng khi dự báo xa:** Ở PEMS04, mức giảm tăng từ 3.46% lên 15.48%; ở PEMS07 từ 3.63% lên 18.22%; ở PEMS08 từ 4.15% lên 22.27%. Điều này cho thấy thông tin giữa các cảm biến đặc biệt hữu ích khi dự báo nhiều bước trong tương lai.

**Không chỉ cải thiện MSE:** MAE cũng giảm ở toàn bộ horizon. Vì vậy, kết quả không phải do mô hình chỉ sửa một số ít sai số lớn.

**Khả năng tổng quát trong cùng họ dữ liệu:** Kết quả không chỉ xuất hiện trên PEMS08 mà lặp lại trên cả PEMS03, PEMS04 và PEMS07.

### 5.6. So sánh nhanh với GTA

Ở PEMS08, horizon 12, seed 2024:

| Phương pháp | MSE | MAE |
|---|---:|---:|
| GTR | 0.07072 | 0.16820 |
| GTA của tác giả | 0.07715 | 0.17595 |
| **APCM đề xuất** | **0.06776** | **0.16417** |

GTA làm kết quả kém hơn GTR trên phép thử này, trong khi APCM cải thiện cả MSE và MAE. Kết quả phù hợp với giả thuyết rằng ép tất cả cảm biến vào một token duy nhất có thể gây mất thông tin, còn APCM giữ được nhiều nhóm quan hệ khác nhau.

Lưu ý: GTA hiện mới được chạy một seed tại horizon 12 nên chỉ được xem là kết quả tham khảo, chưa phải kết luận tổng quát.

## 6. Kết quả thực nghiệm trên Electricity

| Horizon | GTR MSE | APCM MSE | MSE giảm | GTR MAE | APCM MAE | MAE giảm | Số seed thắng |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 96 | 0.13442 ± 0.00019 | **0.13214 ± 0.00005** | **1.69%** | 0.22901 | **0.22762** | **0.61%** | 3/3 |
| 192 | 0.15217 ± 0.00013 | **0.15152 ± 0.00023** | **0.43%** | 0.24542 | **0.24515** | **0.11%** | 3/3 |
| 336 | 0.16950 ± 0.00031 | **0.16797 ± 0.00017** | **0.90%** | 0.26365 | **0.26263** | **0.39%** | 3/3 |
| 720 | 0.20727 ± 0.00202 | **0.20368 ± 0.00830** | **1.71%** | 0.29857 | **0.29690** | **0.56%** | 2/3 |

### 6.1. Nhận xét kết quả Electricity

APCM đạt MSE và MAE trung bình tốt hơn GTR ở cả 4/4 horizon. Ba horizon 96, 192 và 336 thắng đủ 3/3 seed. Ở horizon 720, APCM thắng 2/3 seed và có độ lệch chuẩn lớn hơn GTR; vì vậy kết quả trung bình tốt hơn nhưng độ ổn định ở dự báo rất dài vẫn cần cải thiện.

Mức tăng trên Electricity nhỏ hơn PEMS08, từ 0.43% đến 1.71% MSE. Điều này hợp lý vì quan hệ giữa các cảm biến giao thông có cấu trúc không gian rõ hơn, còn các chuỗi khách hàng điện có thể ít đồng nhất hơn. Tuy vậy, việc APCM vẫn cải thiện trung bình ở toàn bộ bốn horizon cho thấy mô-đun không chỉ khớp riêng PEMS08.

## 7. Kết quả thực nghiệm trên Traffic

Traffic có 862 kênh và là phép kiểm tra quan trọng cho khả năng mô hình hóa quan hệ liên kênh.

| Horizon | GTR MSE | APCM MSE | MSE giảm | GTR MAE | APCM MAE | Seed thắng |
|---:|---:|---:|---:|---:|---:|:---:|
| 96 | 0.43553 ± 0.00182 | **0.40059 ± 0.00015** | **8.02%** | 0.26516 | **0.25994** | 3/3 |
| 192 | 0.44741 ± 0.00334 | **0.42116 ± 0.00271** | **5.87%** | 0.27929 | **0.27060** | 3/3 |
| 336 | 0.46187 ± 0.00326 | **0.43810 ± 0.00219** | **5.14%** | 0.28453 | **0.27785** | 3/3 |
| 720 | 0.50358 ± 0.00097 | **0.47460 ± 0.00514** | **5.76%** | 0.30252 | **0.29651** | 3/3 |

APCM tốt hơn GTR ở cả MSE và MAE trên toàn bộ bốn horizon, đồng thời thắng 12/12 phép so sánh theo seed. Kết quả Traffic đặc biệt có giá trị vì số kênh lớn: APCM vẫn khai thác được thông tin liên kênh với chi phí gần tuyến tính theo `C × K`, không cần attention `C²`.

## 8. Tổng hợp kết quả trên sáu dataset

| Dataset | Số horizon tốt hơn theo MSE | Khoảng giảm MSE | Seed thắng |
|---|:---:|---:|:---:|
| Electricity | 4/4 | 0.43%–1.71% | 11/12 |
| PEMS03 | 4/4 | 2.22%–3.26% | 12/12 |
| PEMS04 | 4/4 | 3.46%–15.48% | 12/12 |
| PEMS07 | 4/4 | 3.63%–18.22% | 12/12 |
| PEMS08 | 4/4 | 4.15%–22.27% | 12/12 |
| Traffic | 4/4 | 5.14%–8.02% | 12/12 |
| **Tổng** | **24/24** | **0.43%–22.27%** | **71/72** |

Kết quả cho thấy hiệu quả mạnh nhất trên các bộ dữ liệu giao thông nhiều cảm biến, đúng với động cơ thiết kế của APCM. Electricity có mức tăng nhỏ hơn nhưng vẫn cải thiện trung bình ở cả bốn horizon, cung cấp bằng chứng rằng phương pháp không chỉ phù hợp với một loại dữ liệu duy nhất.

## 9. Vì sao APCM có thể cải thiện mạnh ở horizon dài?

Khi horizon ngắn, lịch sử gần nhất của chính cảm biến đã chứa nhiều thông tin, nên quan hệ giữa các cảm biến chỉ giúp thêm một phần nhỏ.

Khi horizon dài, sai số của dự báo độc lập dễ tích lũy. Quan hệ giữa các cảm biến cung cấp thêm tín hiệu về trạng thái chung của mạng lưới giao thông. APCM khai thác tín hiệu này theo từng patch và từng nhóm, giúp phần hiệu chỉnh ngày càng có giá trị khi horizon tăng.

## 10. Điểm mới so với GTR và GTA

| Đặc điểm | GTR | GTA | APCM |
|---|:---:|:---:|:---:|
| Mô hình hóa chu kỳ toàn cục | Có | Có | Có |
| Khai thác quan hệ giữa các kênh | Không đáng kể | Có | Có |
| Số đại diện liên kênh | Không có | 1 token | Nhiều latent group |
| Thay đổi theo sample/patch | Không | Hạn chế | Có |
| Giữ hiệu chỉnh riêng cho từng kênh | Có | Không rõ, broadcast chung | Có |
| Có đường residual khởi tạo đúng bằng GTR | Không áp dụng | Không | Có |
| Độ phức tạp theo số kênh | Tuyến tính | Tuyến tính | Tuyến tính theo `C × K` |

## 11. Giới hạn của kết quả hiện tại

Kết quả hiện tại rất tích cực nhưng chưa đủ để kết luận APCM tốt trên mọi bài toán:

- Đã xác nhận trên sáu dataset nhưng chưa chạy ETT series, Solar-Energy và Weather.
- Chưa có ablation ba seed để tách đóng góp của FFT, số latent group và gate.
- Chưa tổng hợp số tham số, thời gian huấn luyện và bộ nhớ so với GTR.
- GTA mới được chạy một seed tại một horizon.
- Ở Electricity horizon 720, APCM chỉ thắng 2/3 seed và độ lệch chuẩn còn cao.

Vì vậy, cách phát biểu an toàn là: **“APCM cải thiện MSE trung bình của GTR trên toàn bộ 24 cấu hình thuộc sáu dataset đã thử nghiệm và thắng 71/72 phép so sánh theo seed”**, chưa nên nói “APCM luôn tốt hơn GTR trên mọi dataset”.

## 12. Thực nghiệm cần làm tiếp

Ưu tiên theo thứ tự:

1. Chạy Solar-Energy và Weather để mở rộng sang hai miền dữ liệu khác.
2. Chạy ETT series để hoàn thành toàn bộ benchmark của bài báo.
3. Chạy ablation `APCM-noFFT` và `K = 1, 4, 8, 16` để chứng minh đóng góp của FFT và số latent group.
4. So sánh số tham số, thời gian mỗi epoch và VRAM.
5. Tìm cách ổn định Electricity horizon 720, chẳng hạn giảm learning rate hoặc điều chỉnh regularization của gate.

## 13. Đoạn trình bày ngắn với giảng viên

> Ban đầu nhóm cải tiến GTR bằng phát hiện chu kỳ tự động, nhưng kết quả không ổn định vì chu kỳ tác giả chọn đã khá tốt. Nhóm chuyển sang một hạn chế khác được chính bài báo đề cập: GTR xử lý các kênh gần như độc lập nên chưa khai thác tốt quan hệ giữa chúng. Nhóm đề xuất APCM, chia dữ liệu thành các patch, dùng đặc trưng thời gian và tần số để gán mềm các kênh vào nhiều nhóm ẩn, rồi chỉ thêm phần thông tin liên kênh hữu ích vào GTR qua residual có gate. Khác với GTA dùng một global token cho tất cả kênh, APCM giữ nhiều nhóm động theo từng mẫu và từng patch. Trên sáu dataset, APCM đạt MSE trung bình tốt hơn GTR ở cả 24/24 cấu hình horizon và thắng 71/72 phép so sánh theo seed. Mức giảm MSE đạt 2.22%–22.27% trên PEMS series, 5.14%–8.02% trên Traffic và 0.43%–1.71% trên Electricity. Kết quả cho thấy mô hình hóa quan hệ liên kênh có hiệu quả rõ ràng và tổng quát hơn hướng phát hiện lại chu kỳ.

### 13.1. Cách nói riêng về ý tưởng APCM

> APCM là mô-đun trộn thông tin giữa các biến một cách thích nghi. GTR gốc học tốt quan hệ thời gian nhưng còn xử lý các kênh khá độc lập. APCM chia chuỗi thành các đoạn ngắn, nhìn cả dạng tín hiệu trong miền thời gian lẫn nhịp dao động trong miền tần số, rồi tự quyết định những kênh nào nên trao đổi thông tin và trao đổi bao nhiêu. Nó không ép tất cả kênh vào một đại diện chung mà sử dụng nhiều nhóm ẩn, thay đổi theo từng đầu vào và từng đoạn thời gian. Thông tin mới chỉ được cộng vào GTR dưới dạng residual có gate, nên mô hình giữ được baseline gốc và chỉ học thêm phần hiệu chỉnh hữu ích.

Nếu giảng viên hỏi “điểm mới nằm ở đâu?”, trả lời ngắn: **“Điểm mới là trộn kênh theo nhiều nhóm động ở cấp patch, kết hợp đặc trưng thời gian–tần số và residual khởi tạo bằng 0; không trộn cố định toàn bộ kênh như một global token.”**

## 14. Tệp triển khai và dữ liệu kết quả

- Mô hình: [`models/GTRAPCM.py`](../models/GTRAPCM.py)
- Tham số dòng lệnh: [`run.py`](../run.py)
- Bộ chạy thực nghiệm: [`utils/run_apcm_experiments.py`](../utils/run_apcm_experiments.py)
- Tổng hợp kết quả: [`utils/summarize_apcm.py`](../utils/summarize_apcm.py)
- Kết quả thô: [`results_apcm/runs.jsonl`](../results_apcm/runs.jsonl)
- Bảng tổng hợp: [`results_apcm/summary.csv`](../results_apcm/summary.csv)
- Hướng dẫn chạy: [`docs/APCM_RUN_GUIDE.md`](APCM_RUN_GUIDE.md)

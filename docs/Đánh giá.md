# Đánh giá: kết quả thực nghiệm và đối chiếu với các công bố

Tài liệu tổng hợp kết quả của mọi phương pháp theo giao thức [`configs/protocol.json`](../configs/protocol.json) (v1.3.2, 27/11/2026): chọn cấu hình trên **Validation** (notebook 05–07, 09), chấm **Test một lần duy nhất** (notebook 10), rồi đặt cạnh số liệu của 2 công bố trên RetailRocket (Ludewig et al. 2021; Shehzad & Jannach 2024). Mọi report nguồn trong `reports/` đều `PASS`; bộ kiểm thử 55/55 `pytest` đạt.

Tài liệu liên quan: [Data.md](Data.md) (dữ liệu, tiền xử lý) • [Phương pháp.md](<Phương pháp.md>) (công thức, cài đặt) • số liệu gốc: [`reports/10_final/`](../reports/10_final/BAO_CAO_KET_QUA.md).

---

## Tóm tắt

- **GRU4Rec dẫn đầu ở mọi giai đoạn** (Validation, Test, phiên 30' và 60'). Trên Test, nó hơn VSTAN khoảng +0,019 và hơn Repeat-last khoảng +0,038 NDCG@20, tạo cách biệt rõ nhất ở **phiên ngắn** (prefix 1 và 2–4, ~75% truy vấn) và vượt xa Repeat-last ở **target mới**.
- **Thêm thông tin đa hành vi chỉ giúp rất ít:** trên Test (refit), chênh lệch giữa GRU4Rec đa hành vi và GRU4Rec gốc là +0,0008 đến +0,0011 NDCG@20 (có ý nghĩa thống kê, chủ yếu ở target lặp); trên Validation thì không khác biệt, còn nhóm kNN trọng số đều tốt hơn trọng số hành vi.
- **Phiên 60' nhỉnh hơn phiên 30' một chút** (khoảng +0,001 trên cùng bộ truy vấn), thứ hạng các mô hình hoàn toàn không đổi.
- **So với các bài báo:** thứ tự trong nhóm kNN (VSTAN ≈ STAN > SKNN) khớp với văn liệu. Khác với benchmark của Ludewig et al. (2021), GRU4Rec của dự án vượt VSTAN. HR@20 tuyệt đối thấp hơn cả hai công bố vì dự án không lọc item hiếm và giữ item mới trong mẫu số.

**Bảng xếp hạng chung — Test (`train_validation_refit`), phiên 30', 92.468 truy vấn**

| Hạng | Mô hình | **NDCG@20** | CI 95% | HR@20 | MRR@20 | Chênh với hạng 1 |
| :---: | --- | ---: | :---: | ---: | ---: | ---: |
| 1 | **GRU4Rec đa hành vi** | **0,3721** | [0,3696; 0,3747] | 0,5104 | **0,3312** | — |
| 2 | GRU4Rec gốc (chỉ item) | 0,3713 | [0,3688; 0,3740] | **0,5106** | 0,3299 | −0,0008 |
| 3 | VSTAN | 0,3528 | [0,3502; 0,3555] | 0,4933 | 0,3115 | −0,0193 |
| 4 | Repeat-last | 0,3337 | [0,3311; 0,3366] | 0,3817 | 0,3179 | −0,0384 |
| 5 | Popularity | 0,0041 | [0,0038; 0,0044] | 0,0102 | 0,0025 | −0,3680 |

---

## 1. Cách đánh giá

### 1.1. Bốn thang đo tại K = 20 (`src/rr/metrics.py`)

Với truy vấn $q$, tập đích $G_q$ (ID gốc, kể cả item ngoài vocabulary) và danh sách gợi ý không trùng $L_q = (\hat{i}_1, \dots, \hat{i}_{20})$; vị trí thiếu được đệm `-1` và luôn tính là trượt.

| Thang đo | Công thức cho một truy vấn | Ý nghĩa |
| --- | --- | --- |
| **HR@20** | $\mathbb{I}\big(\lvert L_q \cap G_q\rvert > 0\big)$ | Có trúng ít nhất một item đúng không |
| **Recall@20** | $\lvert L_q \cap G_q\rvert \,/\, \lvert G_q\rvert$ | Gần như bằng HR@20 vì ~99,8% truy vấn có $\lvert G_q\rvert = 1$ |
| **MRR@20** | $1 / r_1$ nếu hit đầu tiên ở hạng $r_1 \le 20$, ngược lại 0 | Item đúng đứng sớm đến đâu |
| **NDCG@20** (chính) | $\text{DCG}/\text{IDCG}$, với $\text{DCG} = \sum_{r=1}^{20} \frac{\mathbb{I}(\hat{i}_r \in G_q)}{\log_2(r+1)}$ | Chất lượng toàn bộ thứ hạng, chuẩn hóa về [0; 1] |

### 1.2. Hai khía cạnh đánh giá dữ liệu và nguyên tắc

Các mô hình được đánh giá tập trung trên **2 khía cạnh trọng tâm**:
1. **Cấu trúc bên trong phiên:** tách **Target mới vs Target lặp** (D1) và **4 nhóm độ dài tiền tố** `1`, `2–4`, `5–9`, `10+` nhóm timestamp (D3).
2. **Thiết kế tín hiệu đầu vào:** đối chứng **đa hành vi** (`GRU4Rec đa hành vi` vs `GRU4Rec gốc`; kNN trọng số đều vs có trọng số) và độ nhạy **ngưỡng cắt phiên** (`30 phút` vs `60 phút` trên cùng tập truy vấn).

**Nguyên tắc chấm điểm:**
- **Không bỏ truy vấn khó:** mọi truy vấn có target đều nằm trong mẫu số (kể cả item ngoài vocabulary — tính 0 điểm).
- **Kiểm định thống kê:** bootstrap 95% theo đơn vị truy vấn (1.000 lần lấy mẫu, seed 2026); khi so hai mô hình hoặc hai ngưỡng phiên trên cùng bộ truy vấn, dùng paired bootstrap cho $\Delta = \text{NDCG}@20(A) - \text{NDCG}@20(B)$ và báo $p(\Delta \le 0)$.
- **Test chạy một lần:** lần chạy Test v1.3.2 đầu tiên (26/11) bị ngắt giữa chừng, không do lỗi code. Lần chạy lại ngày 27/11 dùng đúng cấu hình đóng băng (`frozen.json`) và cho NDCG@20 trùng khớp các tổ hợp đã in trước khi ngắt.

---

## 2. So sánh các phương pháp của dự án

### 2.1. Toàn cảnh NDCG@20 qua Validation và Test

ItemKNN, SKNN và STAN chỉ chạy trên Validation; VSTAN được chọn làm mốc kNN đại diện cho Test (`train_validation_refit`).

| Mô hình | Val 30' | Val 60' | **Test 30'** | **Test 60'** |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0,0047 | 0,0047 | **0,0041** | **0,0041** |
| ItemKNN | 0,0964 | — | — | — |
| Repeat-last | 0,3578 | 0,3628 | **0,3337** | **0,3387** |
| SKNN | 0,3671 | — | — | — |
| STAN | 0,3745 | — | — | — |
| VSTAN | 0,3747 | 0,3769 | **0,3528** | **0,3559** |
| GRU4Rec gốc | **0,3951** | **0,3962** | **0,3713** | **0,3742** |
| GRU4Rec đa hành vi | 0,3950 | 0,3956 | **0,3721** | **0,3752** |

Thứ hạng **giữ nguyên tuyệt đối** từ Validation sang Test ở cả hai ngưỡng 30' và 60': **GRU4Rec > VSTAN > Repeat-last > Popularity**.

### 2.2. Kết quả Test đầy đủ (HR, MRR, NDCG)

Chấm trên tập Test (`train_validation_refit`) — 30': 92.468 truy vấn; 60': 95.783 truy vấn.

| Mô hình | HR@20 30' | MRR@20 30' | **NDCG@20 30'** | HR@20 60' | MRR@20 60' | **NDCG@20 60'** | CI 95% NDCG 60' |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Popularity | 0,0102 | 0,0025 | 0,0041 | 0,0101 | 0,0025 | 0,0041 | [0,0039; 0,0044] |
| Repeat-last | 0,3817 | 0,3179 | 0,3337 | 0,3874 | 0,3226 | 0,3387 | [0,3359; 0,3416] |
| VSTAN | 0,4933 | 0,3115 | 0,3528 | 0,4941 | 0,3154 | 0,3559 | [0,3533; 0,3584] |
| GRU4Rec gốc | **0,5106** | 0,3299 | 0,3713 | **0,5111** | 0,3337 | 0,3742 | [0,3717; 0,3769] |
| GRU4Rec đa hành vi | 0,5104 | **0,3312** | **0,3721** | 0,5101 | **0,3352** | **0,3752** | [0,3727; 0,3778] |

### 2.3. Khía cạnh 1: Cấu trúc bên trong phiên — Target mới/lặp & Độ dài tiền tố

NDCG@20, Test 30' (`train_validation_refit`). Tỷ lệ truy vấn theo độ dài prefix trên Test 30': 1 nhóm 40,9%, 2–4 nhóm 33,5%, 5–9 nhóm 12,7%, 10+ nhóm 12,8%.

| Mô hình | Target mới | Target lặp | Prefix 1 | Prefix 2–4 | Prefix 5–9 | Prefix 10+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularity | 0,0022 | 0,0070 | 0,0045 | 0,0046 | 0,0033 | 0,0027 |
| Repeat-last | 0,0018 | **0,8291** | 0,3569 | 0,3523 | **0,3007** | **0,2441** |
| VSTAN | 0,1070 | 0,7194 | 0,4024 | 0,3688 | 0,2955 | 0,2094 |
| GRU4Rec gốc | **0,1172** | 0,7504 | 0,4265 | 0,3905 | 0,2977 | 0,2181 |
| GRU4Rec đa hành vi | 0,1161 | 0,7540 | **0,4273** | **0,3912** | 0,3002 | 0,2177 |

- **Phiên ngắn (prefix 1–4, ~74% truy vấn):** GRU4Rec hơn rõ mọi mô hình khác. Đây là nơi nó tạo ra khoảng cách tổng thể.
- **Phiên dài (5–9, 10+):** Repeat-last lại đứng đầu, vì khách thường quay lại item vừa xem.
- **Target lặp:** Repeat-last gần như tối ưu (0,8291). GRU4Rec đứng sau Repeat-last nhưng hơn VSTAN.
- **Target mới:** mọi mô hình đều thấp. Trên Validation 30', các mô hình phi tuần tự còn nhỉnh hơn GRU4Rec ở nhóm này: ItemKNN 0,1340, SKNN 0,1210, GRU4Rec gốc 0,1193, VSTAN 0,1069.
- **Phiên 60'** giữ nguyên mẫu hình này: GRU4Rec đa hành vi đạt 0,1148 ở target mới, 0,7522 ở target lặp, 0,4353 ở prefix 1 và 0,4002 ở prefix 2–4.

### 2.4. Khía cạnh 2: Kiểm định tín hiệu đầu vào (Đa hành vi & Ngưỡng 30'/60')

Kiểm định paired bootstrap trên cùng bộ truy vấn (NDCG@20, Test `train_validation_refit`).

**A. Thông tin đa hành vi có giúp không? (GRU4Rec đa hành vi − GRU4Rec gốc, RQ1)**

| Mô hình fit · bộ truy vấn | Δ | CI 95% | p(Δ ≤ 0) | Kết luận |
| --- | ---: | :---: | ---: | --- |
| 30' · q30 | +0,0008 | [+0,0000; +0,0016] | 0,023 | Hơn, có ý nghĩa (p < 0,05) |
| 30' · q60 | +0,0010 | [+0,0002; +0,0017] | 0,008 | Hơn, có ý nghĩa (p < 0,01) |
| 60' · q30 | +0,0011 | [+0,0003; +0,0018] | 0,001 | Hơn, có ý nghĩa (p < 0,01) |
| 60' · q60 | +0,0010 | [+0,0003; +0,0017] | 0,003 | Hơn, có ý nghĩa (p < 0,01) |

Trên Validation, CI 95% của hiệu này đều chứa 0: 30' Δ = −0,00004 ([−0,00081; +0,00070], p = 0,554); 60' Δ = −0,00060 ([−0,00130; +0,00020], p = 0,929).

**B. Phiên 60' hay 30'? (mô hình 60' − mô hình 30' trên cùng bộ truy vấn)**

| Mô hình | Δ trên q30 | CI 95% | Δ trên q60 | CI 95% |
| --- | ---: | :---: | ---: | :---: |
| VSTAN | +0,0012 | [0,0008; 0,0016] | +0,0013 | [0,0010; 0,0017] |
| GRU4Rec gốc | +0,0009 | [−0,0000; 0,0018] | +0,0009 | [0,0000; 0,0019] |
| GRU4Rec đa hành vi | +0,0011 | [0,0002; 0,0021] | +0,0009 | [0,0001; 0,0018] |

**C. Hơn mốc Repeat-last bao nhiêu? (mô hình − Repeat-last)**

| Mô hình | Phiên 30' | CI 95% | Phiên 60' | CI 95% |
| --- | ---: | :---: | ---: | :---: |
| GRU4Rec đa hành vi | **+0,0384** | [0,0369; 0,0399] | **+0,0365** | [0,0350; 0,0380] |
| GRU4Rec gốc | +0,0376 | [0,0360; 0,0390] | +0,0356 | [0,0341; 0,0371] |
| VSTAN | +0,0190 | [0,0174; 0,0206] | +0,0172 | [0,0156; 0,0188] |

### 2.5. Chọn cấu hình GRU4Rec (notebook 07, Validation 30')

| Thử nghiệm | HR@20 | MRR@20 | **NDCG@20** | So với bản chọn |
| --- | ---: | ---: | ---: | ---: |
| **BPR-max, h = 224, dropout emb 0,5 (chọn)** | **0,5348** | **0,3537** | **0,3950** | — |
| BPR-max, dropout emb 0,25 | 0,5326 | 0,3472 | 0,3893 | −0,0057 |
| BPR-max, h = 100 (4 epoch) | 0,5280 | 0,3398 | 0,3827 | −0,0123 |
| Cross-entropy có hiệu chỉnh log-Q, h = 224 | 0,4832 | 0,3211 | 0,3579 | −0,0371 |

### 2.6. Trọng số hành vi trong kNN (Validation 30', NDCG@20)

Trọng số đều (1, 1, 1) luôn tốt hơn hoặc ngang trọng số theo hành vi (1, 2, 1) cho thấy tín hiệu đa hành vi không cải thiện nhóm kNN (RQ1):

| Mô hình | Trọng số đều | Trọng số (1, 2, 1) |
| --- | ---: | ---: |
| ItemKNN | **0,0964** | 0,0907 |
| SKNN | **0,3671** | 0,3659 |
| STAN | **0,3745** | 0,3744 |
| VSTAN | **0,3747** | 0,3745 |

---

## 3. Đối chiếu với các công bố trên RetailRocket

### 3.1. Giao thức của các nghiên cứu khác nhau thế nào

Các con số dưới đây **không so ngang tuyệt đối được**: mỗi nghiên cứu lọc dữ liệu, chia tập và chọn mục tiêu dự đoán khác nhau. Bảng này giải thích vì sao.

| Tiêu chí | **Dự án này** | Ludewig et al. 2021 (UMUAI) | Shehzad & Jannach 2024 (ECIR) |
| --- | --- | --- | --- |
| Ngưỡng cắt phiên | 30 phút | — | — |
| Lọc item hiếm | **Không lọc** | — | Bỏ item < 5 lần |
| Lọc phiên / user | Giữ mọi phiên ≥ 2 nhóm | — | Bỏ phiên độ dài 1 |
| Lặp liên tiếp (reload) | **Giữ** | — | — |
| Chia train/test | Theo thời gian; test là **15,4 ngày cuối** | 5 đoạn liên tiếp (~27 ngày/đoạn), test 2 ngày cuối mỗi đoạn, lấy trung bình | Ngày cuối làm test |
| Item test chưa có trong train | **Giữ, tính 0 điểm** (9,33% target) | Bỏ | Bỏ |
| Mục tiêu dự đoán | Tập item của nhóm timestamp kế tiếp | Item kế tiếp | Item kế tiếp |
| Quy mô | 92.468 truy vấn test | ~210k sự kiện mỗi đoạn | 1,05 triệu click |

Ô "—": bài báo không nêu chi tiết này trong phần mình đã đối chiếu.

### 3.2. Số liệu MRR@20 / HR@20

Dự án dùng số Validation cho STAN và SKNN, vì hai mô hình này không chạy trên Test.

| Phương pháp | Dự án Val 30' | Dự án Test 30' refit | Dự án Test 60' refit | Ludewig 2021 | Shehzad 2024 |
| --- | :---: | :---: | :---: | :---: | :---: |
| **GRU4Rec** | 0,3535 / 0,5359 | 0,3299 / 0,5106 | 0,3337 / 0,5111 | 0,3237 / 0,5669 | — |
| **VSTAN** | 0,3338 / 0,5137 | 0,3115 / 0,4933 | 0,3154 / 0,4941 | 0,3636 / 0,5932 | 0,631 / 0,980 |
| **STAN** | 0,3335 / 0,5138 | — | — | 0,3638 / 0,5938 | 0,548 / 0,931 |
| **SKNN** | 0,3194 / 0,5270 | — | — | 0,3370 / 0,5788 | — |

Nguồn: Ludewig et al. 2021, Bảng 6 (RETAIL, trung bình 5 đoạn) • Shehzad & Jannach 2024, Bảng 4 (RETAIL).

### 3.3. Thứ tự xếp hạng và khoảng cách GRU4Rec − VSTAN

So thứ hạng **bên trong** từng nghiên cứu thì công bằng hơn so số tuyệt đối, vì mọi mô hình trong cùng một nghiên cứu chịu chung một giao thức.

| Nguồn | Thứ tự theo MRR@20 | GRU4Rec − VSTAN (MRR@20) | GRU4Rec − VSTAN (HR@20) |
| --- | --- | ---: | ---: |
| **Dự án, Val 30'** | GRU4Rec > VSTAN ≈ STAN > SKNN | **+0,0197** | **+0,0222** |
| **Dự án, Test 30' refit** | GRU4Rec > VSTAN | **+0,0184** | **+0,0173** |
| Ludewig 2021 | STAN ≈ VSTAN > SKNN > GRU4Rec | −0,0399 | −0,0263 |
| Shehzad 2024 | VSTAN > STAN (không có GRU4Rec) | — | — |

### 3.4. Nhận định

1. **Thứ tự trong nhóm kNN khớp văn liệu.** Dự án có VSTAN ≈ STAN > SKNN, giống Ludewig 2021. Shehzad 2024 cũng xếp VSTAN đứng đầu RETAIL.
2. **GRU4Rec đảo chiều so với Ludewig 2021.** Ở Ludewig, GRU4Rec thua VSTAN (−0,0399 MRR@20); ở dự án, GRU4Rec thắng ở cả Validation và Test (+0,0197 và +0,0184). Shehzad 2024 không chạy GRU4Rec nên không đối chiếu được điểm này. Dự án chưa kiểm chứng nguyên nhân; một lý giải khả dĩ là cấu hình GRU4Rec (BPR-max, constrained embedding, 2.048 mẫu âm), vì riêng việc đổi hàm mất mát đã tạo chênh lệch lớn: BPR-max hơn cross-entropy +0,0371 NDCG@20 trên Validation (mục 2.5).
3. **HR@20 tuyệt đối thấp hơn cả hai công bố** (0,49–0,54 so với 0,57–0,59 ở Ludewig và 0,93–0,98 ở Shehzad), vì giao thức của dự án khắt khe hơn: không lọc item hiếm, giữ 9,33% target Test là item mới trong mẫu số, và chấm trên 15,4 ngày thay vì 1–2 ngày.

### 3.5. Phương pháp chưa có số liệu đối chiếu

Popularity (Test 30' refit 0,0041), Repeat-last (0,3337; thiết kế theo D1 từ ý tưởng *recently viewed* của Jannach et al. 2017), ItemKNN đa hành vi (Val 30' 0,0964) và GRU4Rec đa hành vi (Test 30' refit 0,3721; mở rộng riêng của dự án) không có số liệu tương đương trong 2 nguồn trên, nên không đặt vào bảng đối chiếu.

---

## 4. Trả lời câu hỏi nghiên cứu và hạn chế

| Câu hỏi | Kết luận |
| --- | --- |
| **RQ1 — Đa hành vi & ngưỡng phiên có giúp không?** | **Cải thiện nhỏ.** Với kNN, trọng số hành vi không giúp (trọng số đều tốt hơn). Với GRU4Rec trên Test, bản đa hành vi hơn bản gốc +0,0008 đến +0,0011 NDCG@20 (p = 0,001–0,023, chủ yếu ở target lặp), còn trên Validation không khác biệt do 96,68% sự kiện là `view`. Nới ngưỡng phiên lên 60' tăng nhẹ ~+0,001 và không đổi thứ hạng mô hình. |
| **RQ2 — Cấu trúc & thứ tự trong phiên có quan trọng không?** | **Có, rõ rệt.** GRU4Rec hơn VSTAN +0,019 và hơn Repeat-last +0,038 trên Test, hơn xa ItemKNN (0,0964 trên Validation) và Popularity (0,0041). Lợi thế lớn nhất nằm ở **phiên ngắn** (prefix 1 và 2–4, ~74% truy vấn) và khả năng gợi ý **target mới**. |

**Hạn chế và hướng phát triển**

- **Target mới còn yếu:** trên Validation 30', GRU4Rec (0,1193 ở target mới) còn thấp hơn ItemKNN (0,1340) và SKNN (0,1210). Có thể thử cơ chế trộn *repeat–explore* hoặc tăng trọng số khám phá khi huấn luyện.
- **Sản phẩm mới ngoài vocabulary:** mô hình chưa khai thác `categoryid` để gợi ý item chưa xuất hiện trong tập huấn luyện.
- **Đánh giá ngoại tuyến:** chưa có log hiển thị (exposure) và chưa đo tác động thực tế qua A/B test.

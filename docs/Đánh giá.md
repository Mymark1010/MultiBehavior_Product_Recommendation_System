# Giao thức Đánh giá, Kết quả Thực nghiệm và Đối chiếu Nghiên cứu

Tài liệu này tổng hợp chuẩn mực và trung thực toàn bộ kết quả thực nghiệm đã hoàn thành của đề tài theo giao thức [`configs/protocol.json`](../configs/protocol.json) (**`v1.3.2`**, cập nhật `27/11/2026`), bao gồm: **Validation** (`NB05–07`, `NB09`), **Test cuối** (`NB10`), **Ablation độ vững dữ liệu** (`NB12`) và **Đối chiếu với các công bố khoa học trên RetailRocket**. Mọi report nguồn trong `reports/` đều đạt trạng thái `PASS` cùng **58/58 unit tests** (`pytest`).

- Tài liệu dữ liệu & tiền xử lý: [`Data.md`](Data.md)
- Tài liệu phương pháp & công thức mô hình: [`Phương pháp.md`](<Phương pháp.md>)

---

## 1. Giao thức đánh giá, thang đo tại $K = 20$ và kiểm định thống kê

### 1.1. Bốn thang đo xếp hạng (`src/rr/metrics.py`)

Với mỗi truy vấn $q$, tập đích khác rỗng $G_q$ (ID gốc, kể cả item ngoài vocabulary) và danh sách gợi ý không trùng $L_q = (\hat{i}_1, \dots, \hat{i}_K)$ tại **$K = 20$** (vị trí thiếu được đệm `-1`, luôn tính là trượt):

- **Hit Rate (`HR@20`):** $\text{Hit}@20(q) = \mathbb{I}\big(|L_q \cap G_q| > 0\big)$ — tỷ lệ truy vấn trúng ít nhất 1 sản phẩm đúng.
- **Recall (`Recall@20`):** $\text{Recall}@20(q) = \frac{|L_q \cap G_q|}{|G_q|}$ — rất sát `HR@20` vì ~99,8% truy vấn có $|G_q| = 1$.
- **Mean Reciprocal Rank (`MRR@20`):** $\text{MRR}@20(q) = \frac{1}{r_1}$ nếu hit đầu tiên nằm ở hạng $r_1 \le 20$, ngược lại bằng `0`.
- **Normalized Discounted Cumulative Gain (`NDCG@20` — chỉ số chính):**
  $$\text{DCG}@20(q) = \sum_{r=1}^{20} \frac{\mathbb{I}(\hat{i}_r \in G_q)}{\log_2(r + 1)}, \qquad \text{IDCG}@20(q) = \sum_{r=1}^{\min(|G_q|, 20)} \frac{1}{\log_2(r + 1)}, \qquad \text{NDCG}@20(q) = \frac{\text{DCG}@20(q)}{\text{IDCG}@20(q)}$$

### 1.2. Nguyên tắc đánh giá và kiểm định thống kê

- **Mẫu số đầy đủ (không bỏ OOV):** Mọi truy vấn có target khác rỗng đều tính vào trung bình; query có toàn bộ target ngoài vocabulary (`5,31%` lượt target ở Validation; `9,33%` ở Test) nhận `0` điểm.
- **Phân rã đa chiều:** Báo cáo song song theo loại target (**Tất cả / Target mới / Target lặp** — D1) và 4 nhóm độ dài prefix (**`1`, `2–4`, `5–9`, `10+`** — D3).
- **Bootstrap 95% & Paired Bootstrap (`n_resamples = 1.000`, `seed = 2026`):** Lấy mẫu có hoàn lại theo đơn vị truy vấn. Khi so hai mô hình $A$ và $B$ (hoặc hai ngưỡng phiên trên cùng bộ query), tính phân phối hiệu $\Delta = \text{NDCG}@20(A) - \text{NDCG}@20(B)$ và xác suất một phía $p(\text{không tốt hơn}) = P(\Delta \le 0)$.
- **Ghi nhận minh bạch về lần chạy Test:** Lần chạy Test v1.3.2 đầu tiên ngày 26/11 bị ngắt giữa chừng (không phải lỗi code); người dùng duyệt chạy lại đúng cùng cấu hình đóng băng (`frozen.json`) ngày 27/11 và kết quả khớp tuyệt đối các con số đã in trước khi ngắt.

---

## 2. Kết quả chọn mô hình trên Validation (`NB05–07`, `NB09`)

### 2.1. Tổng hợp toàn bộ mô hình trên Validation 30 phút (`92.566` query, `train_only`)

| Phương pháp | Cấu hình đã chọn | `HR@20` | `Recall@20` | `MRR@20` | **`NDCG@20`** | Target mới | Target lặp | Prefix `1` | Prefix `2–4` | Prefix `5–9` | Prefix `10+` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Popularity** | `w = (1, 2, 1)` | 0,0113 | 0,0112 | 0,0031 | **0,0047** | 0,0026 | 0,0077 | 0,0052 | 0,0053 | 0,0040 | 0,0021 |
| **ItemKNN** | `M=200, decay=0.6, đều` | 0,1910 | 0,1909 | 0,0695 | **0,0964** | **0,1340** | 0,0447 | 0,0975 | 0,1076 | 0,0975 | 0,0603 |
| **Repeat-last** | Mốc gần đây (D1) | 0,4050 | 0,4048 | 0,3422 | **0,3578** | 0,0017 | **0,8475** | 0,3787 | 0,3773 | **0,3226** | **0,2693** |
| **SKNN** | `k=200, s=500, đều` | 0,5270 | 0,5266 | 0,3194 | **0,3671** | 0,1210 | 0,7054 | 0,4178 | 0,4050 | 0,3038 | 0,1554 |
| **STAN** | `k=200, λ=(1,7d,1), đều` | 0,5138 | 0,5135 | 0,3335 | **0,3745** | 0,1077 | 0,7414 | 0,4164 | 0,3936 | 0,3183 | 0,2378 |
| **VSTAN** | `STAN + λ_idf=0.5, đều` | 0,5137 | 0,5134 | 0,3338 | **0,3747** | 0,1069 | 0,7428 | 0,4158 | 0,3940 | 0,3191 | 0,2395 |
| **GRU4Rec đa HV** | `BPR-max, h=224, ep=2` | 0,5348 | 0,5344 | **0,3537** | **0,3950** | 0,1185 | 0,7753 | **0,4456** | 0,4155 | 0,3220 | 0,2430 |
| **GRU4Rec gốc** | Chỉ item, `h=224, ep=2` | **0,5359** | **0,5355** | 0,3535 | **0,3951** | 0,1193 | 0,7742 | 0,4445 | **0,4173** | 0,3219 | 0,2424 |

### 2.2. Chọn cấu hình GRU4Rec và Ablation thứ tự phiên (`NB07` — RQ2)

| Thử nghiệm GRU4Rec (Validation 30') | `HR@20` | `MRR@20` | **`NDCG@20`** | Chênh lệch so với bản chọn |
| --- | ---: | ---: | ---: | ---: |
| **`bpr-max_official` (`h=224, drop_emb=0.5`, chọn)** | **0,5348** | **0,3537** | **0,3950** | `0,0000` |
| `bpr-max_dropemb0.25` (`h=224, drop_emb=0.25`) | 0,5326 | 0,3472 | 0,3893 | `-0,0057` |
| `bpr-max_hidden100` (`h=100, drop_emb=0.5`, ep 4) | 0,5280 | 0,3398 | 0,3827 | `-0,0123` |
| `cross-entropy_official` (`h=224, log-Q CE`) | 0,4832 | 0,3211 | 0,3579 | `-0,0371` |
| **Ablation RQ2 — Xáo trộn thứ tự nhóm (`shuffled_groups`)** | 0,5053 | 0,3299 | 0,3699 | **`-0,0251`** |
| **Ablation RQ2 — Chỉ giữ nhóm cuối (`last_group_only`)** | 0,4948 | 0,3303 | 0,3674 | **`-0,0277`** |

### 2.3. Kiểm tra trọng số hành vi ở kNN & hai biến thể đánh giá phụ (Validation 30', `NDCG@20`)

- **Trọng số hành vi trong kNN (RQ1 trên NB06):** Với cả 4 mô hình kNN, trọng số **đều `(1,1,1)` đều tốt hơn hoặc ngang bằng** trọng số hành vi `(1,2,1)`: `ItemKNN` (`0,0964` vs `0,0907`), `SKNN` (`0,3671` vs `0,3659`), `STAN` (`0,3745` vs `0,3744`), `VSTAN` (`0,3747` vs `0,3745`).
- **Biến thể đánh giá phụ (`exclude_seen` & `available != 0`):**

| Phương pháp | Chuẩn (chính) | `exclude_seen` (bỏ item đã có trong prefix) | `available != 0` (lọc trạng thái hàng tại $t$) |
| --- | ---: | ---: | ---: |
| Popularity | 0,0047 | 0,0015 | 0,0056 |
| Repeat-last | 0,3578 | — | 0,2692 |
| ItemKNN | 0,0964 | 0,0789 | 0,1039 |
| SKNN / STAN / VSTAN | 0,3671 / 0,3745 / 0,3747 | **0,0967** / 0,0845 / 0,0838 | 0,3085 / 0,3130 / 0,3133 |
| **GRU4Rec (đa HV)** | **0,3950** | 0,0913 | **0,3257** |

### 2.4. Validation hai ngưỡng phiên 30' vs 60' (`reports/09_validation_30_60/metrics.json`)

Trên Validation 60 phút (`96.148` query): `Popularity` `0,0047`, `Repeat-last` `0,3628`, `VSTAN` `0,3769`, `GRU4Rec gốc` **`0,3962`** (`HR@20 = 0,5347`, `MRR@20 = 0,3553`), `GRU4Rec đa hành vi` `0,3956` (`HR@20 = 0,5335`, `MRR@20 = 0,3549`). Paired bootstrap `đa hành vi − gốc` trên `train_only` validation đều có CI 95% chứa `0`: 30' $\Delta = -0{,}00004$ (`[-0,00081; +0,00070]`, $p = 0{,}554$); 60' $\Delta = -0{,}00060$ (`[-0,00130; +0,00020]`, $p = 0{,}929$).

---

## 3. Kết quả thực nghiệm cuối trên tập Test (`NB10` — `reports/10_final/`)

Đánh giá đúng một lần sau khi đóng băng `frozen.json` trên **`q30 = 92.468` query** (30') và **`q60 = 95.783` query** (60').

### 3.1. Bảng kết quả chính — Biến thể `train_validation_refit` (biến thể chính theo D2)

| Mô hình | `HR@20` (30') | `MRR@20` (30') | **`NDCG@20` (30')** | CI 95% (30') | `HR@20` (60') | `MRR@20` (60') | **`NDCG@20` (60')** | CI 95% (60') |
| --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | :---: |
| **Popularity** | 0,0102 | 0,0025 | **0,0041** | `[0,0038; 0,0044]` | 0,0101 | 0,0025 | **0,0041** | `[0,0039; 0,0044]` |
| **Repeat-last** | 0,3817 | 0,3179 | **0,3337** | `[0,3311; 0,3366]` | 0,3874 | 0,3226 | **0,3387** | `[0,3359; 0,3416]` |
| **VSTAN** | 0,4933 | 0,3115 | **0,3528** | `[0,3502; 0,3555]` | 0,4941 | 0,3154 | **0,3559** | `[0,3533; 0,3584]` |
| **GRU4Rec gốc (chỉ item)** | **0,5106** | 0,3299 | **0,3713** | `[0,3688; 0,3740]` | **0,5111** | 0,3337 | **0,3742** | `[0,3717; 0,3769]` |
| **GRU4Rec đa hành vi** | 0,5104 | **0,3312** | **0,3721** | `[0,3696; 0,3747]` | 0,5101 | **0,3352** | **0,3752** | `[0,3727; 0,3778]` |

### 3.2. Bảng kết quả đối chiếu — Biến thể `train_only` trên Test

| Mô hình | `HR@20` (30') | `MRR@20` (30') | **`NDCG@20` (30')** | CI 95% (30') | `HR@20` (60') | `MRR@20` (60') | **`NDCG@20` (60')** | CI 95% (60') |
| --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | :---: |
| **Popularity** | 0,0087 | 0,0032 | **0,0043** | `[0,0039; 0,0047]` | 0,0087 | 0,0032 | **0,0043** | `[0,0040; 0,0047]` |
| **Repeat-last** | 0,3679 | 0,3080 | **0,3229** | `[0,3202; 0,3257]` | 0,3738 | 0,3126 | **0,3278** | `[0,3252; 0,3308]` |
| **VSTAN** | 0,4623 | 0,2969 | **0,3345** | `[0,3320; 0,3373]` | 0,4643 | 0,3008 | **0,3379** | `[0,3354; 0,3404]` |
| **GRU4Rec gốc (chỉ item)** | **0,4807** | 0,3154 | **0,3532** | `[0,3507; 0,3557]` | **0,4807** | **0,3182** | **0,3553** | `[0,3528; 0,3580]` |
| **GRU4Rec đa hành vi** | 0,4803 | **0,3162** | **0,3537** | `[0,3512; 0,3564]` | 0,4791 | 0,3176 | **0,3545** | `[0,3520; 0,3571]` |

> **Tác động của `train_validation_refit` (D2):** Refit giúp **mọi mô hình cá nhân hóa tăng mạnh `+0,018` đến `+0,021` `NDCG@20`** (`GRU4Rec` 30': `0,3537 → 0,3721`; `VSTAN` 30': `0,3345 → 0,3528`) nhờ mở rộng vocabulary (`212.915 → 224.167` item) và cập nhật tín hiệu gần thời điểm Test hơn 14 ngày.

### 3.3. Kiểm định Paired Bootstrap trên tập Test (`NDCG@20`)

**A. Hiệu ứng đa hành vi (`GRU4Rec đa hành vi − GRU4Rec gốc` — RQ1):**

| Ngưỡng fit · biến thể · bộ query | $\Delta$ | CI 95% | $p(\Delta \le 0)$ | Nhận định |
| --- | ---: | :---: | ---: | --- |
| `30' · train_validation_refit · q30` | **+0,0008** | `[+0,0000; +0,0016]` | `0,0230` | Cao hơn có ý nghĩa ($p < 0{,}05$) |
| `30' · train_validation_refit · q60` | **+0,0010** | `[+0,0002; +0,0017]` | `0,0080` | Cao hơn có ý nghĩa ($p < 0{,}01$) |
| `60' · train_validation_refit · q30` | **+0,0011** | `[+0,0003; +0,0018]` | `0,0010` | Cao hơn có ý nghĩa ($p < 0{,}01$) |
| `60' · train_validation_refit · q60` | **+0,0010** | `[+0,0003; +0,0017]` | `0,0030` | Cao hơn có ý nghĩa ($p < 0{,}01$) |
| `30' · train_only · q30` / `q60` | `+0,0005` / `+0,0005` | `[-0,0002; +0,0012]` | `0,081` / `0,076` | CI chứa `0` (chưa có ý nghĩa) |
| `60' · train_only · q30` / `q60` | **`-0,0009` / `-0,0008`** | `[-0,0016; -0,0001]` | `0,993` / `0,978` | Thấp hơn trên `train_only` 60' |

**B. Hiệu ứng ngưỡng phiên (`60' − 30'` trên cùng bộ query, `train_validation_refit`):**
- **VSTAN:** `+0,0012` trên `q30` (`[0,0008; 0,0016]`), `+0,0013` trên `q60` (`[0,0010; 0,0017]`).
- **GRU4Rec gốc:** `+0,0009` trên `q30` (`[-0,0000; 0,0018]`), `+0,0009` trên `q60` (`[0,0000; 0,0019]`).
- **GRU4Rec đa hành vi:** `+0,0011` trên `q30` (`[0,0002; 0,0021]`), `+0,0009` trên `q60` (`[0,0001; 0,0018]`).

**C. Khoảng cách vượt mốc `Repeat-last` (`train_validation_refit`, `NDCG@20`):**
- Phiên 30': **GRU4Rec đa HV** `+0,0384` (`[0,0369; 0,0399]`), **GRU4Rec gốc** `+0,0376` (`[0,0360; 0,0390]`), **VSTAN** `+0,0190` (`[0,0174; 0,0206]`).
- Phiên 60': **GRU4Rec đa HV** `+0,0365` (`[0,0350; 0,0380]`), **GRU4Rec gốc** `+0,0356` (`[0,0341; 0,0371]`), **VSTAN** `+0,0172` (`[0,0156; 0,0188]`).

### 3.4. Phân rã theo loại Target, Độ dài Prefix, Fallback & Thời gian chạy (`Test 30' refit`)

| Mô hình | Target mới | Target lặp | Prefix `1` | Prefix `2–4` | Prefix `5–9` | Prefix `10+` | Tỷ lệ slot fallback | Fit (giây) | Query 92k (giây) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Popularity** | 0,0022 | 0,0070 | 0,0045 | 0,0046 | 0,0033 | 0,0027 | 100,0% | 0,0 | 0,0 |
| **Repeat-last** | 0,0018 | **0,8291** | 0,3569 | 0,3523 | **0,3007** | **0,2441** | 83,8% | 0,0 | 0,6 |
| **VSTAN** | 0,1070 | 0,7194 | 0,4024 | 0,3688 | 0,2955 | 0,2094 | 16,5% | 4,0 | 37,5 |
| **GRU4Rec gốc** | **0,1172** | 0,7504 | 0,4265 | 0,3905 | 0,2977 | **0,2181** | **0,0%** | 620,1 | 188,3 |
| **GRU4Rec đa HV** | 0,1161 | 0,7540 | **0,4273** | **0,3912** | 0,3002 | 0,2177 | **0,0%** | 572,3 | 153,0 |

*(Phiên 60' refit giữ nguyên mẫu hình: GRU4Rec đa HV đạt `0,1148` mới / `0,7522` lặp; `0,4353` ở prefix `1`, `0,4002` ở `2–4`; thời gian fit `498,0s` cho đa HV, `534,1s` cho gốc và `3,9s` cho VSTAN.)*

---

## 4. Phân tích độ vững dữ liệu (`NB12` — `reports/12_data_ablation/`)

### 4.1. Visitor hoạt động bất thường

Ngưỡng học từ `train` (phân vị `0,9999`): `> 145,3` sự kiện/ngày hoặc `> 11` giao dịch/ngày UTC, gắn cờ **87** visitor ở train (`7,47%` query), **14** visitor ở validation (`6,91%` query) và **12** visitor ở test (`6,84%` query).

| Mô hình (`NDCG@20`, phiên 30', `train_only`) | Val: Tất cả | Val: Bỏ cờ | Val: Chỉ cờ | Val: Bỏ Top-10 | Test: Tất cả | Test: Bỏ cờ | Test: Chỉ cờ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularity | 0,0047 | 0,0047 | 0,0047 | 0,0048 | — | — | — |
| Repeat-last | 0,3578 | 0,3633 | **0,2834** | 0,3663 | — | — | — |
| VSTAN | 0,3747 | 0,3835 | 0,2561 | 0,3858 | 0,3345 | 0,3449 | 0,1929 |
| GRU4Rec gốc | **0,3951** | **0,4049** | 0,2629 | **0,4073** | 0,3532 | 0,3642 | 0,2030 |
| GRU4Rec đa hành vi | 0,3950 | **0,4049** | 0,2623 | **0,4073** | **0,3537** | **0,3646** | **0,2041** |

- **Kết luận:** Nhóm bị gắn cờ khó dự đoán hơn nhiều (hành vi tương tự khách mua sỉ). Khi loại bỏ, mọi mô hình tăng `+0,006–0,011` nhưng **thứ hạng chính hoàn toàn giữ vững** (`GRU4Rec > VSTAN > Repeat-last > Popularity`, Kendall $\tau = 0{,}800$; vượt Repeat-last có ý nghĩa với cận dưới CI $\Delta \ge +0{,}0399$).

### 4.2. Gộp nhóm lặp liên tiếp (reload / back)

Có **207.984 / 2.753.015** nhóm (`7,55%`) trùng hệt tập `(item, hành vi)` với nhóm liền trước và **21.163 / 92.566** query validation (`22,86%`) có target trùng hệt nhóm cuối prefix.

| Mô hình (`NDCG@20`, Validation 30') | Fit gốc / Chấm gốc | Fit gộp / Chấm gốc | $\Delta$ (CI 95%) | Fit gốc / Chấm gộp | Fit gộp / Chấm gộp | `duplicate_target` (22,9%) | `other_queries` (77,1%) |
| --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| Popularity | 0,0047 | 0,0040 | `-0,0007` | 0,0042 | 0,0035 | 0,0067 | 0,0042 |
| Repeat-last | 0,3578 | 0,3579 | `+0,0001` | 0,1827 | 0,1828 | **0,9484** | 0,1827 |
| VSTAN | 0,3747 | 0,3746 | `-0,0001` | 0,2434 | 0,2433 | 0,8189 | 0,2430 |
| GRU4Rec gốc | **0,3951** | 0,3552 | **`-0,0399`** (`[-0,0413; -0,0384]`) | 0,2597 | 0,2504 | 0,8550 | 0,2588 |
| GRU4Rec đa hành vi | 0,3950 | 0,3564 | **`-0,0387`** (`[-0,0401; -0,0373]`) | **0,2608** | **0,2517** | 0,8498 | **0,2603** |

- **Kết luận:** Gộp dữ liệu khi huấn luyện gây hại rõ rệt cho GRU4Rec (`-0,039` đến `-0,040` trên dữ liệu gốc và thậm chí thua cả trên dữ liệu đã gộp: `0,2517` vs `0,2608`). Ngược lại, tách báo cáo `other_queries` (77,1% truy vấn không lặp hiển nhiên) cho thấy giá trị thực rất lớn của mô hình học sâu: **GRU4Rec đạt `0,2603` vượt Repeat-last (`0,1827`) tới `+42,5%` tương đối**.

---

## 5. Đối chiếu với các công bố nghiên cứu trên RetailRocket

Chỉ đối chiếu trực tiếp những phương pháp có số liệu bảng được xác minh trên cùng bộ dữ liệu **RetailRocket** từ 4 nguồn độc lập/chính thức. **Không so ngang tuyệt đối** vì các công bố dùng giao thức lọc item hiếm (`< 5` lần), loại item mới (OOV) và độ dài cửa sổ Test khác nhau (`1 ngày` ở ECIR 2024; `7 ngày` ở RecSys 2023; `~14 ngày` giữ toàn bộ OOV ở dự án này).

### 5.1. Bốn phương pháp có số liệu công bố đối chiếu trực tiếp (`MRR@20 / HR@20`)

| Phương pháp | Dự án — Val 30' | Dự án — Test 30' (`refit`) | Dự án — Test 60' (`refit`) | Ludewig et al. (UMUAI 2021) | Latifi et al. (Inf. Sci. 2021)\* | Shehzad & Jannach (ECIR 2024) | Hidasi & Czapp (RecSys 2023)\*\* |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GRU4Rec gốc** | `0,3535 / 0,5359` | `0,3299 / 0,5106` | `0,3337 / 0,5111` | `0,3237 / 0,5669` | `0,4196 / 0,6172` | — | `0,2175 / 0,5186` |
| **VSTAN** | `0,3338 / 0,5137` | `0,3115 / 0,4933` | `0,3154 / 0,4941` | `0,3636 / 0,5932` | `0,4397 / 0,6745` | `0,631 / 0,980` (`5s`) | — |
| **STAN** | `0,3335 / 0,5138` | — | — | `0,3638 / 0,5938` | `0,4411 / 0,6747` | `0,548 / 0,931` (`2,5s`) | — |
| **SKNN** | `0,3194 / 0,5270` | — | — | `0,3370 / 0,5788` | — | — | — |

*\*Latifi et al. (2021): chỉ trích cấu hình session-based gốc, không dùng biến thể có lịch sử dài hạn của user. \*\*Hidasi & Czapp (2023): bảng Results trong README repo `GRU4Rec_PyTorch_Official` (paramfile RetailRocket, phiên 1 giờ, test 7 ngày, bỏ lặp liên tiếp); bài RecSys 2023, Bảng 3 báo cáo `0,2131 / 0,5187` cho bản chính thức với BPR-max.*

### 5.2. Các phương pháp chưa có bảng đối chiếu trực tiếp trên RetailRocket

- **Popularity** (`Test 30' refit: 0,0041`), **Repeat-last** (`0,3337`, thiết kế riêng theo D1 từ ý tưởng *Recently Viewed* của Jannach et al., 2017), **ItemKNN đa hành vi** (`Val 30': 0,0964`) và **GRU4Rec đa hành vi** (`Test 30' refit: 0,3721`, mở rộng riêng của dự án) chưa có số liệu bảng trực tiếp tương đương trong 4 nguồn công bố trên RetailRocket nên không nội suy hay bịa số đối chiếu.

### 5.3. Ba nhận định từ đối chiếu văn liệu

1. **Thứ tự nội bộ nhóm kNN và chi phí CPU khớp sát văn liệu:** `VSTAN ≈ STAN > SKNN` trùng hoàn toàn với Ludewig et al. (2021); thời gian fit `VSTAN` (`2,8–4,0s`) khớp ECIR 2024 (`2,5–5s`).
2. **Biên độ `HR@20` thấp hơn phản ánh giao thức khắt khe, sát thực tế:** Việc giữ toàn bộ item hiếm và giữ `5,31%` (Val) / `9,33%` (Test) OOV trong mẫu số giải thích trực tiếp chênh lệch so với các benchmark đã lọc.
3. **GRU4Rec vượt VSTAN trên tổng thể, khớp lý giải của Hidasi & Czapp (2023):** Cài đặt đúng bộ tính năng chuẩn (`BPR-max` + `Constrained Embedding` + `hidden=224`, `batch=80`, `n_neg=2048`) giúp GRU4Rec đảo ngược kết quả các benchmark trước 2023 (vượt `VSTAN` `+0,0185` đến `+0,0193` `NDCG@20` trên Test). Riêng ở bài toán khám phá **Target mới**, kết quả Validation (`ItemKNN 0,1340 > SKNN 0,1210 > GRU4Rec tự cài 0,1193`) cho thấy sự đánh đổi giữa tối ưu thứ hạng sớm (lặp) và độ phủ sản phẩm mới.

---

## 6. Tổng kết trả lời 3 câu hỏi nghiên cứu và hạn chế

| Câu hỏi nghiên cứu | Kết luận rút ra từ toàn bộ thực nghiệm (`Val` + `Test` + `NB12`) |
| --- | --- |
| **RQ1 (Đa hành vi)** | **Lợi ích rất nhỏ và chưa ổn định.** Với nhóm kNN, trọng số hành vi không giúp ích (trọng số đều tốt hơn). Với GRU4Rec, ở biến thể chính `train_validation_refit` trên Test, đa hành vi vượt bản gốc có ý nghĩa thống kê khoảng **`+0,0008` đến `+0,0011` `NDCG@20`** (`p = 0,001–0,023`, chủ yếu ở Target lặp); nhưng ở Validation `train_only` không khác biệt và ở Test `train_only` 60' lại giảm `-0,0009`. Nguyên nhân cốt lõi là `96,68%` sự kiện là `view`, tín hiệu `addtocart`/`transaction` quá thưa. |
| **RQ2 (Thứ tự phiên)** | **Khẳng định rõ rệt.** `GRU4Rec` vượt `VSTAN` (`+0,019` Test refit), vượt `Repeat-last` (`+0,038`), vượt xa `ItemKNN` (`0,0964`) và `Popularity` (`0,0041`). Phá hủy thứ tự nhóm (`shuffled_groups`) làm giảm mạnh `-0,0251`, chỉ giữ nhóm cuối giảm `-0,0277`. |
| **RQ3 (Cân bằng chất lượng – chi phí – ít lịch sử)** | **GRU4Rec** cho chất lượng cao nhất tổng thể, **`0%` fallback**, và vượt trội rõ ở phiên ngắn (`prefix = 1` và `2–4`, chiếm ~75% truy vấn), đổi lại tốn ~8–10 phút fit và ~2,5–3 phút suy luận trên CPU cho 92k query. **VSTAN** là phương án thay thế nhẹ cực hiệu quả: chỉ tốn **~4 giây fit** và **37,5 giây suy luận**, đạt ~95% điểm của GRU4Rec (`0,3528` vs `0,3721`). |

### Hạn chế ghi nhận và hướng phát triển

- **Cold-start item (OOV `9,33%` trên Test):** Chưa khai thác `categoryid` trong đầu vào mô hình để phục vụ sản phẩm mới chưa có trong vocabulary.
- **Target mới còn yếu:** Trên Validation 30', GRU4Rec tự cài (`0,1193` `NDCG@20` ở target mới) còn thấp hơn `ItemKNN` (`0,1340`) và `SKNN` (`0,1210`). Hướng cải tiến: cơ chế trộn *repeat–explore* hoặc tăng trọng số khám phá khi huấn luyện.
- **Đánh giá ngoại tuyến:** Chưa có log hiển thị (*exposure*) và chưa đo tác động doanh thu qua A/B test trực tuyến.

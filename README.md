# Gợi ý sản phẩm theo phiên từ tương tác đa hành vi (RetailRocket)

Khi khách hàng duyệt sàn thương mại điện tử, hệ thống quan sát chuỗi tương tác trong phiên hiện tại (**xem**, **thêm giỏ**, **mua**) và dự đoán **Top-20 sản phẩm có khả năng được tương tác ở bước kế tiếp**.

Trọng tâm nghiên cứu là đánh giá lợi ích của thông tin **đa hành vi** và **ngưỡng phiên (30 phút vs 60 phút)** bằng cách so sánh hai mô hình cùng kiến trúc **GRU4Rec gốc (chỉ sản phẩm)** và **GRU4Rec đa hành vi (sản phẩm + loại hành vi)** cùng các mốc đối chứng **Popularity**, **Repeat-last** và **VSTAN** theo giao thức chống rò rỉ [`configs/protocol.json`](configs/protocol.json) (`v1.3.2`).

```text
Ví dụ một phiên (gom các sự kiện cùng thời điểm thành 1 nhóm):
  g₁: xem A   →   g₂: xem B   →   g₃: thêm giỏ B + xem C

Tại cuối g₂, đầu vào là tiền tố (g₁, g₂), đáp án cần dự đoán là tập {B, C} (chấm chính bằng NDCG@20).
```

---

## Tài liệu chi tiết (`docs/`)

Để xem đầy đủ số liệu, công thức toán học và phân tích chuyên sâu, vui lòng đọc 3 tài liệu chính trong thư mục `docs/`:

| Tài liệu | Nội dung chính |
| --- | --- |
| **[`docs/Data.md`](docs/Data.md)** | Đặc điểm bộ dữ liệu RetailRocket (2,756M sự kiện, 20,28M thuộc tính), pipeline tiền xử lý 5 bước, chia tập theo thời gian 80/10/10 chống rò rỉ, đặc tả Input–Output và 6 quyết định giao thức (D1–D6). |
| **[`docs/Phương pháp.md`](<docs/Phương pháp.md>)** | Công thức toán học và chi tiết cài đặt các mốc cơ sở (`Popularity`, `Repeat-last`), họ kNN (`ItemKNN`, `SKNN → STAN → VSTAN`), kiến trúc `GRU4Rec gốc` vs `GRU4Rec đa hành vi` (`BPR-max`), siêu tham số và tài liệu tham khảo. |
| **[`docs/Đánh giá.md`](<docs/Đánh giá.md>)** | Bộ thang đo tại $K = 20$ (`HR`, `Recall`, `MRR`, `NDCG@20`), toàn bộ kết quả Validation & Test, kiểm định Paired Bootstrap 95%, Ablation độ vững dữ liệu và đối chiếu các công bố khoa học trên RetailRocket. |
| **[Slide báo cáo (PPTX)](docs/BAO_CAO_TIEN_DO_PROJECT3_HUST_27_11.pptx)** | Bản trình bày tổng quan tiến độ và kết quả dự án. |

---

## Kết quả chính trên tập Test (`train_validation_refit`, $K = 20$)

Đánh giá đúng một lần sau khi đóng băng cấu hình trên **92.468 truy vấn** (phiên 30') và **95.783 truy vấn** (phiên 60'), giữ nguyên sản phẩm ngoài vocabulary trong mẫu số (tính 0 điểm). Chi tiết: [`reports/10_final/BAO_CAO_KET_QUA.md`](reports/10_final/BAO_CAO_KET_QUA.md) và [`docs/Đánh giá.md`](<docs/Đánh giá.md>).

| Mô hình | HR@20 (30') | MRR@20 (30') | **NDCG@20 (30')** | HR@20 (60') | MRR@20 (60') | **NDCG@20 (60')** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularity | 0,0102 | 0,0025 | **0,0041** | 0,0101 | 0,0025 | **0,0041** |
| Repeat-last *(mốc gợi ý lặp)* | 0,3817 | 0,3179 | **0,3337** | 0,3874 | 0,3226 | **0,3387** |
| VSTAN *(mốc kNN mạnh nhất)* | 0,4933 | 0,3115 | **0,3528** | 0,4941 | 0,3154 | **0,3559** |
| GRU4Rec gốc *(chỉ sản phẩm)* | **0,5106** | 0,3299 | **0,3713** | **0,5111** | 0,3337 | **0,3742** |
| **GRU4Rec đa hành vi** | 0,5104 | **0,3312** | **0,3721** | 0,5101 | **0,3352** | **0,3752** |

**Ba kết luận rút gọn:**
- **Thứ tự phiên quyết định chất lượng:** `GRU4Rec` dẫn đầu tuyệt đối trên cả Validation và Test, vượt `VSTAN` (`+0,019`) và vượt `Repeat-last` (`+0,038` `NDCG@20`).
- **Tín hiệu đa hành vi có lợi ích nhỏ:** Trên biến thể `refit`, `GRU4Rec đa hành vi` vượt bản gốc có ý nghĩa thống kê khoảng `+0,0008` đến `+0,0011` `NDCG@20` (chủ yếu ở target lặp), nhưng chưa đồng nhất ở chế độ `train_only` do `96,68%` log là lượt xem.
- **Cân bằng thực tiễn:** `GRU4Rec` đạt chất lượng cao nhất với `0%` fallback (fit ~8–10 phút trên CPU); `VSTAN` đạt ~95% điểm của GRU4Rec nhưng chỉ tốn ~4 giây fit.

---

## Cấu trúc thư mục

```text
├── README.md                 ← Tổng quan nhanh dự án (tệp này)
├── docs/                     ← Tài liệu chi tiết: Data.md, Phương pháp.md, Đánh giá.md, slide PPTX
├── configs/protocol.json     ← Giao thức thực nghiệm đã chốt (v1.3.2)
├── data/                     ← Dữ liệu raw CSV (không lên git) và processed Parquet (xem docs/Data.md)
├── notebooks/                ← Pipeline thực nghiệm chạy theo thứ tự 01 → 12
├── src/rr/                   ← Mã nguồn dùng chung: xử lý, metric, lọc ứng viên, kNN, GRU4Rec, demo
├── tests/                    ← Bộ kiểm thử tự động (pytest) cho src/rr/
├── models/                   ← Cấu hình và trọng số mô hình đã lưu (xem docs/Phương pháp.md)
├── reports/                  ← Kết quả JSON/MD và chuỗi mã băm SHA-256 từng bước (xem reports/README.md)
├── scripts/                  ← Script khảo sát dữ liệu độc lập
├── requirements.txt          ← Thư viện Python 3.14 cần cài
└── pyproject.toml            ← Cấu hình gói rr và pytest
```

---

## Pipeline thực nghiệm (`notebooks/`)

| Giai đoạn | Notebook | Nhiệm vụ chính |
| --- | --- | --- |
| **1. Dữ liệu** | [01](notebooks/01_kiem_tra_va_lam_sach_events.ipynb), [02](notebooks/02_tao_phien_nhom_timestamp_va_chia_tap.ipynb), [Kiểm tra](notebooks/kiem_tra_ket_qua_xu_ly.ipynb), [03](notebooks/03_lich_su_thuoc_tinh_va_catalog.ipynb), [04](notebooks/04_vocabulary_va_bo_mau.ipynb) | Làm sạch events, cắt phiên 30', kiểm định độc lập `36/36 PASS`, trích xuất lịch sử thuộc tính as-of, tạo vocabulary và bộ mẫu `prefix → target`. |
| **2. Validation** | [05](notebooks/05_danh_gia_va_popularity.ipynb), [06](notebooks/06_itemknn_va_sessionknn.ipynb), [07](notebooks/07_gru4rec.ipynb), [09](notebooks/09_gru4rec_hanh_vi_phien_30_60.ipynb) | Chọn siêu tham số trên Validation (không đọc Test) cho `Popularity`, `Repeat-last`, họ kNN (`ItemKNN`, `SKNN → STAN → VSTAN`), `GRU4Rec gốc` và `GRU4Rec đa hành vi` trên cả hai ngưỡng 30' và 60'. |
| **3. Test & Phân tích** | [10](notebooks/10_thuc_nghiem_cuoi_test.ipynb), [11](notebooks/11_demo.ipynb), [12](notebooks/12_ablation_du_lieu.ipynb) | Đóng băng cấu hình, diễn tập validation rồi chấm **Test đúng một lần** (`10`), chạy demo tương tác Top-20 (`11`) và phân tích độ vững dữ liệu (`12`). |

*(Ghi chú: không có notebook `08` vì SR-GNN đã được gỡ khỏi phạm vi từ `v1.3.1`.)*

---

## Cài đặt và chạy nhanh

**Yêu cầu:** Python 3.14, ~8 GB RAM, ~3 GB ổ cứng (mọi mô hình chạy hoàn toàn trên CPU, không yêu cầu GPU).

```bash
git clone https://github.com/Mymark1010/MultiBehavior_Product_Recommendation_System.git
cd MultiBehavior_Product_Recommendation_System
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
nbstripout --install                 # tránh commit output notebook lên git
python -m pytest tests               # kiểm tra mã nguồn dùng chung
```

**Tải dữ liệu gốc** từ [Kaggle `retailrocket/ecommerce-dataset`](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) vào `data/raw/` (gồm 4 tệp: `events.csv`, `item_properties_part1.csv`, `item_properties_part2.csv`, `category_tree.csv`; xem mã SHA-256 kiểm chứng trong [`docs/Data.md`](docs/Data.md)):

```bash
pip install kaggle
kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw --unzip
```

**Chạy pipeline:** Mở các notebook theo thứ tự `01 → 02 → kiểm tra → 03 → … → 12` bằng kernel `.venv` (hoặc chạy dòng lệnh):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_kiem_tra_va_lam_sach_events.ipynb
```

- Với **Notebook 10**, chạy diễn tập trên validation trước (`RR_FINAL_SPLIT=validation jupyter nbconvert ...`), sau đó mới chạy đánh giá Test thật một lần.
- Mỗi bước tự động xác thực mã băm `sha256` / `content_sha256` của bước trước để đảm bảo tính toàn vẹn và khả năng tái lập.

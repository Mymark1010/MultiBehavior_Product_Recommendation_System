# Hệ thống gợi ý sản phẩm từ tương tác đa hành vi

Gợi ý **Top-10 / Top-20 sản phẩm cho bước tương tác kế tiếp trong phiên** trên dữ liệu RetailRocket, dựa trên các hành vi đã quan sát: xem (`view`), thêm giỏ (`addtocart`) và mua (`transaction`). Project so sánh Popularity, Repeat-last, ItemKNN đa hành vi, Session-KNN và GRU4Rec trên cùng bộ đánh giá ngoại tuyến.

Hiện trạng chi tiết, đặc điểm dữ liệu và kết quả khảo sát: [docs/HIEN_TRANG.md](docs/HIEN_TRANG.md). Danh sách việc: [TASKS_HOAN_THIEN.md](TASKS_HOAN_THIEN.md). Quy ước cho Claude Code: [CLAUDE.md](CLAUDE.md).

## Chạy nhanh

Yêu cầu: Python 3.14, khoảng 6 GB RAM trống, 3 GB đĩa cho dữ liệu.

```bash
git clone https://github.com/Mymark1010/MultiBehavior_Product_Recommendation_System.git
cd MultiBehavior_Product_Recommendation_System

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
nbstripout --install          # không đưa output notebook vào git

# Tải dữ liệu (mục bên dưới), rồi chạy lần lượt:
jupyter nbconvert --to notebook --execute --inplace notebooks/01_kiem_tra_va_lam_sach_events.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_tao_phien_nhom_timestamp_va_chia_tap.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/kiem_tra_ket_qua_xu_ly.ipynb
```

Notebook kiểm tra phải ghi `"status": "PASS"` vào `reports/data_validation/metrics.json`. Thời gian tham khảo trên laptop 8 luồng: 01 ≈ 1 phút, 02 ≈ 2 phút, kiểm tra ≈ 3 phút.

**Chọn kernel trong VS Code/Jupyter:** mở notebook → *Select Kernel* → *Python Environments* → chọn `.venv` của project. Các notebook dùng kernelspec chung `python3`, không cần đăng ký kernel riêng. Chạy bằng *Restart Kernel and Run All*.

## Tải dữ liệu

Dữ liệu là [Kaggle dataset `retailrocket/ecommerce-dataset`](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset). Cần tài khoản Kaggle và API token (`~/.kaggle/kaggle.json`):

```bash
pip install kaggle
kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw --unzip
```

Sau khi giải nén, `data/raw/` phải có đúng 4 tệp (số dòng không tính header):

| Tệp | Số dòng | Dung lượng (byte) | SHA-256 |
| --- | ---: | ---: | --- |
| `events.csv` | 2.756.101 | 94.237.913 | `3745aa83238b1e6d44d8fda209807899f420084398f94ddf745f3cbcfecbf9e7` |
| `item_properties_part1.csv` | 10.999.999 | 484.315.749 | `30aad5aeca58b2dc27dcc73e1708565f5818e45adb3eb57401f91e87355b0b81` |
| `item_properties_part2.csv` | 9.275.903 | 408.929.907 | `d5e7d1a91dc40f522aeb596b267e6c87d8aed689a7192d12369cfb165eb987e5` |
| `category_tree.csv` | 1.669 | 14.454 | `94e865eb0a3d48cbbfe3b79079018dd92509315c88f5fd8d00d0b4b5af434f5b` |

Kiểm tra nhanh: `sha256sum data/raw/events.csv` (Windows PowerShell: `Get-FileHash data/raw/events.csv`). Notebook 01 dừng nếu dữ liệu sai schema. Không sửa hay ghi vào `data/raw/`.

## Các notebook

| Notebook | Việc chính | Đầu ra chính |
| --- | --- | --- |
| [01](notebooks/01_kiem_tra_va_lam_sach_events.ipynb) | Kiểm tra schema, bỏ 460 dòng trùng tuyệt đối, chuẩn hóa UTC | `events_clean.parquet`, `reports/events_clean/` |
| [02](notebooks/02_tao_phien_nhom_timestamp_va_chia_tap.ipynb) | Phiên (nghỉ > 30 phút), nhóm `(session_id, timestamp)`, chia 80%/90% theo nguyên phiên | `events_sessionized`, `sessions`, `timestamp_groups`, `splits/`, `split_config.json`, `reports/session_split/` |
| [Kiểm tra](notebooks/kiem_tra_ket_qua_xu_ly.ipynb) | 36 phép kiểm tra độc lập raw → clean → phiên/nhóm → từng tệp split; chỉ đọc | `reports/data_validation/` |
| [03](notebooks/03_lich_su_thuoc_tinh_va_catalog.ipynb) | Lịch sử `categoryid`/`available`, cây danh mục, catalog `known_at_ms` | `item_property_history`, `category_tree`, `item_catalog`, `reports/metadata/` |
| [04](notebooks/04_vocabulary_va_bo_mau.ipynb) | Vocabulary train, bộ query/target, metadata as-of tại t | `item_vocab`, `samples/`, `reports/samples/` |
| [05](notebooks/05_danh_gia_va_popularity.ipynb) | Bộ đánh giá chung, Popularity (đều / theo hành vi), S-POP, Repeat-last | `reports/eval/{popularity,spop,repeat_last}_validation.json` |
| [06](notebooks/06_itemknn_va_sessionknn.ipynb) | SKNN, V-SKNN, STAN, VSTAN, luật tuần tự SR, ItemKNN; trọng số đều vs theo hành vi | `reports/eval/knn_validation.json` |
| [07](notebooks/07_gru4rec.ipynb) | GRU4Rec theo bản chính thức (BPR-max / cross-entropy, constrained embedding) + embedding hành vi | `models/gru4rec/`, `reports/eval/gru4rec_validation.json` |
| [08](notebooks/08_srgnn.ipynb) | SR-GNN trên đồ thị nhóm timestamp (neural đối chứng) | `models/srgnn/`, `reports/eval/srgnn_validation.json` |
| [09](notebooks/09_reranker_hysar.ipynb) | Reranker kiểu HySAR (LightGBM) trên ứng viên kNN + GRU4Rec | `models/reranker/`, `reports/eval/reranker_validation.json` |
| [10](notebooks/10_thuc_nghiem_cuoi_test.ipynb) | Đóng băng cấu hình, chạy test một lần (train_only và refit train+validation) | `reports/final/` |
| [11](notebooks/11_demo.ipynb) | Demo Top-K từ prefix, đối chiếu với lần chạy đánh giá | bảng `rank, item_id, score, category_id, is_fallback` |

Từ notebook 03 trở đi, hàm dùng chung (đọc/ghi, metric, lọc ứng viên, mô hình) nằm trong [`src/rr/`](src/rr/) và có test: `python -m pytest tests`. Notebook 01, 02 và notebook kiểm tra vẫn độc lập, không import `src/`.

## Bài toán và giao thức

Mỗi phiên là chuỗi nhóm timestamp `g₁, …, gₙ`. Một mẫu gồm prefix `(g₁, …, gₜ)` và target là **tập item phân biệt của nhóm `gₜ₊₁`**, mọi loại hành vi. Đầu ra là Top-K item không trùng, có `rank`, `item_id`, `score`, `category_id` (nếu đã biết tại `t`), `is_fallback`.

Quy tắc bất biến (chi tiết: [`CLAUDE.md`](CLAUDE.md), [`configs/protocol.json`](configs/protocol.json)):

- Chỉ học popularity, độ tương đồng, kho phiên, vocabulary, trọng số từ `train`; chọn siêu tham số trên `validation`; `test` chạy một lần.
- Metadata tại `t` chỉ dùng bản cập nhật có `timestamp ≤ t`; thiếu thì `unknown`.
- Ứng viên: có trong vocabulary và `known_at_ms ≤ t`. Luật `available ≠ 0` chỉ dùng cho báo cáo phụ (protocol v1.2.0: 17,6% target validation có `available = 0` tại t).
- Target giữ ID gốc kể cả ngoài vocabulary và vẫn tính vào mẫu số.
- Chỉ số chính **NDCG@20**; báo cáo thêm Recall, Hit Rate, MRR tại K = 10, 20; hai bài toán (tương tác kế tiếp; thêm giỏ/mua ở nhóm kế tiếp); tách theo target mới/lặp, độ dài prefix và độ phổ biến item.
- Bộ phương pháp theo tài liệu tổng quan phương pháp gợi ý theo phiên trên RetailRocket: baseline Popularity/S-POP/Repeat-last, họ kNN + SR, GRU4Rec, SR-GNN, reranker kiểu HySAR (`configs/protocol.json → methods`).

## Mã băm trong report: `sha256` và `content_sha256`

- `sha256`: băm **byte** của tệp. Dùng để chắc chắn đang đọc đúng tệp mà report mô tả. Với Parquet, giá trị này đổi khi khác phiên bản pyarrow/pandas dù dữ liệu y hệt.
- `content_sha256`: băm **nội dung bảng** (sắp theo khóa chính như `event_id`, rồi `pd.util.hash_pandas_object`). Không phụ thuộc cách Parquet xếp byte. **Khi đối chiếu kết quả giữa các máy, so `content_sha256`.**

JSON được ghi với xuống dòng LF (`newline="\n"`) và `.gitattributes` ép `eol=lf`, nên mã băm của `split_config.json` và các report giống nhau trên mọi hệ điều hành.

## Tài liệu

- [Hiện trạng và ghi chú dữ liệu](docs/HIEN_TRANG.md)
- [Giải thích phương pháp và hỏi đáp GRU4Rec](docs/GIAI_THICH_PHUONG_PHAP_VA_HOI_DAP_GRU4REC.md) ([bản Word](docs/GIAI_THICH_PHUONG_PHAP_VA_HOI_DAP_GRU4REC.docx))
- [Báo cáo tổng quan (Word)](docs/Bao_cao_tong_quan_RetailRocket.docx), [slide đề xuất đề tài](docs/BAO_CAO_DE_XUAT_DE_TAI_HUST.pptx)
- [Báo cáo dữ liệu cho GRU4Rec](reports/gru_readiness/BAO_CAO_DU_LIEU_CHO_GRU4REC.md)
- Hướng dẫn tiền xử lý chi tiết, thuyết minh phương pháp chi tiết và tài liệu pipeline hoàn thành project: **(chưa có)** — nội dung chính đã gộp vào `docs/HIEN_TRANG.md` và `configs/protocol.json`.

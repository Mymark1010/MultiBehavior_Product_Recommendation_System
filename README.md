# Gợi ý sản phẩm theo phiên từ tương tác đa hành vi (RetailRocket)

Khi một khách đang duyệt web, ta đã thấy họ **xem**, **thêm giỏ** hoặc **mua** những sản phẩm nào trong phiên hiện tại. Project dự đoán **20 sản phẩm họ có khả năng tương tác ở bước kế tiếp**.

Câu hỏi chính: **cho mô hình biết loại hành vi (xem / thêm giỏ / mua) có giúp gợi ý tốt hơn không?** Để trả lời, project so sánh hai mô hình có cùng kiến trúc:

| Mô hình | Đầu vào mỗi bước |
| --- | --- |
| **GRU4Rec gốc** | chỉ các sản phẩm |
| **GRU4Rec đa hành vi** | sản phẩm + loại hành vi |

Thí nghiệm chạy với hai cách cắt phiên: phiên kết thúc khi khách nghỉ hơn **30 phút** hoặc hơn **60 phút**. Popularity, Repeat-last và VSTAN là các mốc để biết hai mô hình trên đứng ở đâu.

**Kết quả cuối** (đánh giá trên tập test, chạy một lần): [`reports/10_final/BAO_CAO_KET_QUA.md`](reports/10_final/BAO_CAO_KET_QUA.md).

---

## 1. Bài toán trong một ví dụ

Một phiên là chuỗi các **nhóm sự kiện**: các sự kiện cùng thời điểm (cùng timestamp) gộp thành một nhóm.

```
g1: xem A        g2: xem B        g3: thêm giỏ B + xem C
```

Đứng ở cuối `g2`, mô hình phải dự đoán nhóm `g3`. Đáp án là tập sản phẩm `{B, C}`, không phân biệt hành vi. Mô hình trả về 20 sản phẩm, và được chấm bằng **NDCG@20**: đáp án đứng càng cao trong danh sách thì điểm càng cao, tối đa là 1. Ngoài ra project báo cáo thêm HR@20 (có trúng ít nhất một sản phẩm không) và MRR@20.

## 2. Cấu trúc thư mục

```
├── README.md                 ← tệp này
├── CLAUDE.md                 quy tắc làm việc trong repo (giao thức, chống rò rỉ, quy ước code)
├── TASKS_HOAN_THIEN.md       danh sách việc và các quyết định phương pháp đã chốt
├── requirements.txt          thư viện cần cài (Python 3.14)
├── configs/
│   └── protocol.json         giao thức đã chốt: ngưỡng phiên, cách chia tập, metric, phạm vi
├── data/                     dữ liệu (không đưa lên git, xem data/README.md)
│   ├── raw/                  4 tệp CSV gốc từ Kaggle (chỉ đọc)
│   └── processed/            dữ liệu đã xử lý (Parquet) do notebook sinh ra
├── notebooks/                các bước của pipeline, chạy theo số thứ tự 01 → 12
├── src/rr/                   code dùng chung cho notebook 03 trở đi (đọc/ghi, metric, mô hình)
├── tests/                    kiểm thử cho src/rr (python -m pytest tests)
├── models/                   cấu hình mô hình đã huấn luyện (trọng số không đưa lên git, xem models/README.md)
├── reports/                  kết quả từng bước, thư mục đánh số theo notebook (xem reports/README.md)
├── scripts/                  script kiểm tra dữ liệu độc lập
└── docs/                     tài liệu: hiện trạng project, slide báo cáo tiến độ
```

Mỗi notebook ghi kết quả vào `reports/NN_<bước>/`, với `NN` là số của notebook. Ví dụ notebook 04 ghi vào `reports/04_samples/`.

## 3. Các bước của pipeline

**Giai đoạn 1: chuẩn bị dữ liệu**

| Notebook | Làm gì | Kết quả |
| --- | --- | --- |
| [01](notebooks/01_kiem_tra_va_lam_sach_events.ipynb) | Kiểm tra dữ liệu, bỏ 460 dòng trùng, đổi thời gian sang UTC | `reports/01_events_clean/` |
| [02](notebooks/02_tao_phien_nhom_timestamp_va_chia_tap.ipynb) | Cắt phiên (nghỉ > 30 phút), gộp nhóm cùng timestamp, chia train / validation / test theo thời gian (mốc = phân vị 0,8 / 0,9 của timestamp sự kiện, tức 80% / 10% / 10% số sự kiện), giữ nguyên từng phiên | `reports/02_session_split/` |
| [Kiểm tra](notebooks/kiem_tra_ket_qua_xu_ly.ipynb) | 36 phép kiểm tra độc lập cho bước 01–02 (chỉ đọc) | `reports/02_data_validation/` |
| [03](notebooks/03_lich_su_thuoc_tinh_va_catalog.ipynb) | Lịch sử danh mục / tình trạng hàng của sản phẩm theo thời gian | `reports/03_metadata/` |
| [04](notebooks/04_vocabulary_va_bo_mau.ipynb) | Danh sách sản phẩm đã biết (từ train) và các cặp "đoạn đầu phiên → nhóm kế tiếp" | `reports/04_samples/` |

**Giai đoạn 2: chọn cấu hình mô hình trên validation** (không chạm tới test)

Không có notebook 08: SR-GNN đã được gỡ khỏi phạm vi ở protocol v1.3.1 (26/11/2026); S-POP, SR và V-SKNN được gỡ ở v1.3.2. Thư mục `reports/05_08_validation/` giữ tên cũ. Mọi chỉ số tính tại K = 20.

| Notebook | Làm gì | Kết quả |
| --- | --- | --- |
| [05](notebooks/05_danh_gia_va_popularity.ipynb) | Bộ chấm điểm chung (HR, Recall, MRR, NDCG tại K = 20); Popularity, Repeat-last | `reports/05_08_validation/` |
| [06](notebooks/06_itemknn_va_sessionknn.ipynb) | Họ láng giềng gần: SKNN → STAN → VSTAN (chuỗi chọn VSTAN), ItemKNN | `reports/05_08_validation/` |
| [07](notebooks/07_gru4rec.ipynb) | GRU4Rec đa hành vi theo bản chính thức, chọn hàm loss và siêu tham số | `models/gru4rec/` |
| [09](notebooks/09_gru4rec_hanh_vi_phien_30_60.ipynb) | Dựng dữ liệu phiên 60 phút; chọn số epoch cho GRU4Rec gốc và GRU4Rec đa hành vi ở cả hai ngưỡng | `reports/09_validation_30_60/`, `models/gru4rec_item/`, `models/session60/` |

**Giai đoạn 3: kết quả cuối và phân tích**

| Notebook | Làm gì | Kết quả |
| --- | --- | --- |
| [10](notebooks/10_thuc_nghiem_cuoi_test.ipynb) | Đóng băng cấu hình, diễn tập trên validation, rồi chạy test **đúng một lần** | `reports/10_final/` |
| [11](notebooks/11_demo.ipynb) | Demo: nhập một đoạn phiên, nhận 20 gợi ý | bảng `rank, item_id, score, category_id, is_fallback` |
| [12](notebooks/12_ablation_du_lieu.ipynb) | Phân tích phụ: khách hoạt động bất thường, các bước lặp liên tiếp | `reports/12_data_ablation/` |

## 4. Cài đặt và chạy

**Yêu cầu:** Python 3.14, khoảng 8 GB RAM, 3 GB đĩa. Không cần GPU: mọi mô hình chạy trên CPU.

```bash
git clone https://github.com/Mymark1010/MultiBehavior_Product_Recommendation_System.git
cd MultiBehavior_Product_Recommendation_System
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
nbstripout --install                 # không đưa output notebook lên git
python -m pytest tests               # kiểm tra code dùng chung
```

**Tải dữ liệu** từ [Kaggle `retailrocket/ecommerce-dataset`](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) (cần API token Kaggle):

```bash
pip install kaggle
kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw --unzip
```

`data/raw/` phải có đúng 4 tệp. Kiểm tra bằng `sha256sum` (Windows: `Get-FileHash`):

| Tệp | Số dòng | SHA-256 |
| --- | ---: | --- |
| `events.csv` | 2.756.101 | `3745aa83238b1e6d44d8fda209807899f420084398f94ddf745f3cbcfecbf9e7` |
| `item_properties_part1.csv` | 10.999.999 | `30aad5aeca58b2dc27dcc73e1708565f5818e45adb3eb57401f91e87355b0b81` |
| `item_properties_part2.csv` | 9.275.903 | `d5e7d1a91dc40f522aeb596b267e6c87d8aed689a7192d12369cfb165eb987e5` |
| `category_tree.csv` | 1.669 | `94e865eb0a3d48cbbfe3b79079018dd92509315c88f5fd8d00d0b4b5af434f5b` |

**Chạy notebook** theo thứ tự 01 → 02 → kiểm tra → 03 → … → 12. Có hai cách:
- Trong VS Code/Jupyter: chọn kernel `.venv`, rồi *Restart Kernel and Run All*.
- Từ dòng lệnh:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_kiem_tra_va_lam_sach_events.ipynb
```

Lưu ý khi chạy:

- **Notebook 10** chạy hai lần:
  - lần đầu là diễn tập trên validation: `RR_FINAL_SPLIT=validation jupyter nbconvert ...`;
  - lần sau mới là test thật. Lần chạy test bị chặn nếu chưa có bản diễn tập đúng cấu hình, và bị chặn nếu test đã chạy rồi.
- **Theo dõi tiến độ** các notebook chạy lâu (07–10, 12): đặt `RR_PROGRESS_LOG=<đường dẫn tệp>`, mỗi bước sẽ ghi thêm một dòng vào tệp đó.
- **Thời gian tham khảo** trên laptop 8 luồng: các notebook 01–04 chạy vài phút mỗi notebook. Các notebook 07–10 chạy vài giờ mỗi notebook, vì phải huấn luyện mạng nơ-ron trên CPU.

## 5. Nguyên tắc đánh giá

Các nguyên tắc này đảm bảo kết quả không bị "nhìn trước đáp án". Chi tiết ở [`CLAUDE.md`](CLAUDE.md) và [`configs/protocol.json`](configs/protocol.json).

- **Chia theo thời gian:** hai mốc là phân vị 0,8 và 0,9 của timestamp sự kiện, nên train chứa 80% sự kiện sớm nhất, validation 10% tiếp theo, test 10% cuối. Mỗi phiên nằm trọn trong một tập; phiên vắt qua mốc chia bị loại riêng.
- **Chỉ học từ train:** độ phổ biến, độ tương đồng, danh sách sản phẩm và trọng số mô hình đều chỉ lấy từ train. Siêu tham số được chọn trên validation. Test chỉ chạy một lần với cấu hình đã đóng băng.
- **Không dùng thông tin tương lai:** chỉ gợi ý sản phẩm đã xuất hiện trước thời điểm gợi ý. Thông tin danh mục cũng chỉ lấy bản cập nhật trước thời điểm đó.
- **Không bỏ câu khó:** sản phẩm mới chưa từng thấy trong train vẫn được tính vào mẫu số, và luôn tính 0 điểm.
- **Kiểm định thống kê:** khoảng tin cậy bootstrap 95%. Khi so hai mô hình, dùng paired bootstrap trên cùng bộ câu hỏi.

## 6. Kiểm tra tính toàn vẹn

Mỗi notebook ghi mã băm SHA-256 của mọi tệp đầu vào và đầu ra vào report của nó. Bước sau kiểm tra lại các mã băm này và dừng ngay nếu có tệp bị thay đổi. Có hai loại mã băm:

- `sha256`: băm theo byte của tệp.
- `content_sha256`: băm nội dung bảng, không phụ thuộc phiên bản thư viện Parquet. **Khi so kết quả giữa hai máy, dùng `content_sha256`.**

## 7. Tài liệu thêm

- [Hiện trạng project và đặc điểm dữ liệu](docs/HIEN_TRANG.md)
- [Slide báo cáo tiến độ 27/11/2026](docs/BAO_CAO_TIEN_DO_PROJECT3_HUST_27_11.pptx)
- [Kết quả test cuối](reports/10_final/BAO_CAO_KET_QUA.md)
- [Khảo sát dữ liệu trước khi huấn luyện GRU4Rec](reports/02_gru_readiness/BAO_CAO_DU_LIEU_CHO_GRU4REC.md)

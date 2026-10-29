# Hệ gợi ý RetailRocket đa hành vi — next-step theo phiên

Mục tiêu duy nhất: từ prefix của một phiên truy cập, xếp hạng Top-10/20 item sẽ xuất hiện trong **nhóm timestamp kế tiếp của cùng phiên**, với cả `view`, `addtocart` và `transaction`.

## Trạng thái đã kiểm chứng

Đã chạy pipeline schema v2 trên bốn CSV RetailRocket thật: **2.756.101 sự kiện nguồn → 2.755.641 sự kiện sạch → 991.059 query next-step và 993.336 cặp query–item target**.

30/30 kiểm thử đạt; 38/38 kiểm tra dữ liệu đạt; 20 Parquet đã xác minh SHA-256. Kiểm tra độc lập lịch sử/nhãn/metadata trên 240 query và 243 cặp target cũng đạt. Đây là kiểm chứng dữ liệu, không phải kết quả mô hình.

| Tập | Query next-step | Phiên tạo query | Cặp query–item target | Query visitor ngoài train | Độ phủ cặp target trong ứng viên |
|---|---:|---:|---:|---:|---:|
| Train | 806.025 | 306.332 | 807.908 | 0,00% | 82,27% |
| Validation | 92.566 | 38.290 | 92.794 | 77,07% | 79,90% |
| Test | 92.468 | 37.864 | 92.634 | 83,60% | 79,17% |

**Độ phủ ứng viên không phải Recall/NDCG của mô hình.** Ở test, 79,17% cặp target thỏa chính sách ứng viên tại t. Vẫn giữ các target còn lại trong bộ dữ liệu và mẫu số đánh giá. 407.653/993.336 cặp target là item đã xuất hiện trong prefix; không mặc định lọc item đã xem.

Các mô hình Popularity, ItemKNN, Session-KNN/V-SKNN, GRU4Rec/SR-GNN, bộ đánh giá xếp hạng và demo là thiết kế giai đoạn tiếp theo; chưa huấn luyện hoặc có kết quả chất lượng mô hình.

## Cài đặt và kiểm tra

Chạy từ thư mục project; Python 3.12 và dependency ghim được dùng cho lần chạy đã ghi nhận:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_repository.py
```

Trên Windows, thay `.venv/bin/python` bằng `.venv\Scripts\python.exe`. CSV/Parquet không lưu trong Git; clone mới cần tải dữ liệu theo [data/README.md](data/README.md), không mặc định có sẵn dữ liệu chỉ vì các báo cáo đã được commit.

## Tái tạo và xác minh dữ liệu

```bash
.venv/bin/python scripts/prepare_nextstep.py
.venv/bin/python scripts/verify_artifacts.py
.venv/bin/python scripts/audit_nextstep.py
.venv/bin/python scripts/validate_project.py --full-data --notebooks
```

Đầu ra duy nhất: `data/nextstep/`; báo cáo của lần chạy: `reports/nextstep/summary.json` và `manifest.json`. Chạy lại ghi đè các bảng do pipeline quản lý. Không nạp khi còn marker `INCOMPLETE`. Không chạy hai pipeline đồng thời vào một thư mục. Nếu dùng `--config`, truyền cùng cấu hình khi chạy verifier.

## Giao thức chính

1. UTC; loại đúng dòng trùng raw; giữ ba hành vi và các lần lặp item khác thời điểm.
2. Phiên mới khi gap >30 phút. Phân vị thời gian 80%/90%; loại toàn phiên cắt qua mốc split.
3. Gộp mọi sự kiện cùng `(session_id, timestamp)`; sau mỗi nhóm trừ nhóm cuối tạo một query.
4. Target là tập item phân biệt tại đúng nhóm kế tiếp cùng session; cho phép item lặp từ prefix.
5. Prefix chỉ thuộc cùng session, có `event_time <= query_time`. Feature không có nhãn, target_time hoặc độ dài phiên hoàn chỉnh.
6. Ứng viên có `known_at <= t`, available gần nhất là 1 hoặc unknown. Mapping/thống kê chỉ từ train. Không ép target vào candidate hoặc bỏ target khó.

Mẫu âm train mặc định tắt (`negative_samples_per_query=0`); khi bật phải loại toàn bộ tập item target. Không dùng âm train làm tập ứng viên đánh giá.

## Mã và tài liệu hiện hành

- [Pipeline](scripts/prepare_nextstep.py), [cấu hình](configs/nextstep.json), [reader](scripts/dataset.py), [verifier](scripts/verify_artifacts.py), [audit độc lập](scripts/audit_nextstep.py).
- [Giao thức](docs/DATA_PROTOCOL.md), [từ điển dữ liệu](docs/DATA_DICTIONARY.md), [báo cáo dữ liệu](docs/BAO_CAO_XU_LY_DU_LIEU_VI.md), [đề cương](docs/DE_CUONG_NEXTSTEP.md).
- [Slide PowerPoint theo template HUST](docs/Bao_cao_RetailRocket_nextstep.pptx), [báo cáo Word](docs/BAO_CAO_XU_LY_DU_LIEU_VI.docx), [đề cương Word](docs/DE_CUONG_NEXTSTEP.docx).
- [Kết quả kiểm chứng](reports/nextstep/validation.json), [review dữ liệu](reports/nextstep/REVIEW_VI.md), [phân tích prefix](reports/nextstep/analysis.json).
- Ba notebook chỉ đọc: [khảo sát raw](notebooks/01_data_audit.ipynb), [prefix/target](notebooks/02_nextstep_overview.ipynb), [xem Parquet](notebooks/03_xem_du_lieu_parquet.ipynb). Không còn pipeline notebook tạo mẫu song song.

Notebook: cài thêm `requirements-notebooks.txt` và chạy `.venv/bin/jupyter lab`. Báo cáo/slide: cài thêm `requirements-docs.txt`, xác minh bằng `validate_project.py --full-data --notebooks`, rồi chạy `build_nextstep_report.py`, `build_data_report.py`, `build_slides.py` theo thứ tự. Các lệnh tạo tài liệu dùng số liệu trong báo cáo của lần chạy, không tự huấn luyện mô hình.

Báo cáo raw trong `reports/raw_analysis/` là khảo sát nguồn đã ghi nhận; không phải số liệu nhãn hoặc chỉ số mô hình của pipeline khác. Dữ liệu nguồn tải từ Kaggle đã khớp SHA-256 raw được ghi nhận trước đây; xem [nguồn và checksum](reports/nextstep/raw_source.json).

# reports/ — kết quả của từng bước

Mỗi thư mục do đúng một notebook (hoặc script) ghi ra; số đầu tên thư mục là số notebook. Không sửa tay các tệp ở đây: chúng chứa mã băm SHA-256 mà bước sau dùng để kiểm tra đầu vào.

| Thư mục | Do ai ghi | Nội dung |
| --- | --- | --- |
| `01_events_clean/` | notebook 01 | Số dòng, dòng trùng, mã băm của `events_clean.parquet` |
| `02_session_split/` | notebook 02 | Số phiên, nhóm, mốc chia train / validation / test |
| `02_data_validation/` | notebook `kiem_tra_ket_qua_xu_ly` | 36 phép kiểm tra độc lập cho bước 01–02 (`status: PASS`) |
| `02_gru_readiness/` | `scripts/audit_gru_readiness.py` | Khảo sát dữ liệu trước khi huấn luyện GRU4Rec; notebook 04 đối chiếu với số liệu này |
| `03_metadata/` | notebook 03 | Lịch sử danh mục / tình trạng hàng, cây danh mục |
| `04_samples/` | notebook 04 | Vocabulary và các cặp prefix → target của mỗi tập |
| `05_08_validation/` | notebook 05–07 | Kết quả chọn cấu hình trên validation, mỗi họ mô hình một tệp `*_validation.json` (giữ tên cũ; notebook 08 SR-GNN gỡ ở v1.3.1, `spop_validation.json` gỡ ở v1.3.2) |
| `09_validation_30_60/` | notebook 09 | Chọn cấu hình GRU4Rec gốc / đa hành vi và các mốc cho phiên 30 và 60 phút |
| `10_final_rehearsal/` | notebook 10 (`RR_FINAL_SPLIT=validation`) | Diễn tập đường chạy test trên validation; **không dùng để báo cáo** |
| `10_final/` | notebook 10 | **Kết quả cuối trên test** (`BAO_CAO_KET_QUA.md`, `metrics.json`, `frozen.json`) |
| `12_data_ablation/` | notebook 12 | Phân tích phụ: khách bất thường, gộp bước lặp liên tiếp |
| `13_gru4rec_official/` | notebook 13 | Kiểm chứng GRU4Rec bản chính thức trên validation, so với GRU4Rec của dự án |

Trong `metrics.json`: `inputs` là mã băm các tệp đã đọc, `artifacts` là mã băm các tệp đã ghi, `status` là `PASS` khi mọi kiểm tra đạt.

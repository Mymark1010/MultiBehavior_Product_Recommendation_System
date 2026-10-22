# Xử lý dữ liệu Retailrocket đa hành vi

Project hiện dừng ở **hoàn tất xử lý và kiểm tra dữ liệu**. Phạm vi gồm khảo sát dữ liệu gốc, làm sạch log, tạo session, chia tập theo thời gian, tạo query/nhãn mua 7 ngày, đặc trưng lịch sử, metadata theo thời gian và các bảng đầu ra có thể tái lập.

Phần triển khai mô hình, huấn luyện, xếp hạng, kết quả thực nghiệm và demo đã được gỡ bỏ theo yêu cầu reset ngày 22/10/2026. Các bảng train/validation/test, nhãn và mẫu âm được giữ vì là đầu ra của bước chuẩn bị dữ liệu; chưa đại diện cho một mô hình đã huấn luyện.

## Hiện trạng dữ liệu

| Nội dung | Kết quả đã lưu |
|---|---:|
| Sự kiện gốc | 2.756.101 |
| Dòng trùng hoàn toàn bị loại | 460 |
| Sự kiện sau làm sạch | 2.755.641 |
| View / addtocart / transaction | 2.664.218 / 68.966 / 22.457 |
| Cặp visitor–item train | 1.713.144 |
| Query đủ cửa sổ nhãn | 1.519.027 |
| Cặp query–item có nhãn mua | 88.317 |
| Mẫu âm train đã chuẩn bị | 91.540 |
| Tệp Parquet trong bộ hiện tại | 20 |

Thống kê chi tiết: [summary.json](reports/multibehavior/summary.json). Định nghĩa từng bảng: [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md). Quy tắc thời gian, nhãn và dữ liệu thiếu: [DATA_PROTOCOL.md](docs/DATA_PROTOCOL.md).

## Cài đặt và kiểm tra

Chạy từ thư mục project trên Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/verify_artifacts.py
```

Nếu đã có `.venv` thì dùng môi trường đó. Trên Linux/macOS thay `.venv\Scripts\python.exe` bằng `.venv/bin/python`.

Unit tests dùng dữ liệu giả lập nhỏ, không cần tải Retailrocket. `verify_artifacts.py` cần dữ liệu thật đã xử lý; đối chiếu số dòng, checksum raw/đầu ra, cấu hình và mã pipeline với manifest.

`requirements-lock.txt` ghi phiên bản thư viện của lần xử lý đã lưu; phiên bản Python nằm trong manifest. Các phụ thuộc chạy pipeline chỉ gồm NumPy, pandas và PyArrow.

## Tái tạo dữ liệu khi cần

Đặt bốn CSV gốc vào `data/raw/` theo [data/README.md](data/README.md), rồi chạy:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_multibehavior.py
.\.venv\Scripts\python.exe scripts/verify_artifacts.py
```

Pipeline ghi đầu ra vào `data/multibehavior/` và báo cáo vào `reports/multibehavior/`. Chạy lại sẽ ghi lại các đầu ra do pipeline quản lý. Không cần chạy lại nếu dữ liệu đang có đã khớp manifest. Các thư mục dữ liệu của notebook cũ được giữ riêng.

## Các công đoạn đã hoàn thành

1. Kiểm tra trường bắt buộc; chỉ giữ ba hành vi hợp lệ; loại đúng dòng trùng hoàn toàn.
2. Chuẩn hóa UTC; bảo toàn các hành vi khác nhau trên cùng item và thứ tự lưu có thể tái lập.
3. Tạo session theo khoảng nghỉ lớn hơn 30 phút; chia train/validation/test theo phân vị 80%/90% thời gian sự kiện.
4. Đánh dấu session qua ranh giới; loại khỏi lịch sử và nhãn của bộ query.
5. Tạo query sau toàn bộ nhóm sự kiện có timestamp đầu tiên của session.
6. Tạo nhãn mua trong 7 ngày; giữ query không mua; lưu riêng query thiếu cửa sổ quan sát.
7. Tạo đặc trưng lịch sử toàn kỳ, 7/30 ngày, thời điểm gần nhất và recency; tách riêng nhãn.
8. Tổng hợp bảng visitor–item và visitor–item–behavior chỉ từ train; mapping có PAD=0, UNK=1.
9. Xử lý category/availability theo thời gian, cây danh mục và catalog; giữ rõ giá trị chưa biết.
10. Chuẩn bị mẫu âm train theo điều kiện tại query, kiểm tra dữ liệu và ghi manifest/checksum.

## Cấu trúc còn lại

```text
configs/multibehavior.json       Cấu hình xử lý dữ liệu
scripts/prepare_multibehavior.py Pipeline chính
scripts/dataset.py              Hàm đọc history và candidate tham chiếu
scripts/verify_artifacts.py     Xác minh dữ liệu đã lưu
scripts/build_data_report.py    Xuất báo cáo xử lý dữ liệu ra Word
tests/test_pipeline.py          Kiểm thử pipeline trên fixture nhỏ
notebooks/                     Khảo sát và minh họa đọc dữ liệu
data/raw/                      Bốn CSV gốc
data/multibehavior/             Bộ dữ liệu xử lý hiện tại
data/interim/, processed/, splits/ Đầu ra notebook cũ, giữ riêng
reports/multibehavior/          Thống kê, kiểm tra và manifest
reports/raw_analysis/          Phân tích chất lượng dữ liệu gốc
docs/                          Giao thức, từ điển dữ liệu và báo cáo
```

Notebook [02_multibehavior_overview.ipynb](notebooks/02_multibehavior_overview.ipynb) chỉ đọc bộ hiện tại. Hai notebook `01_` là khảo sát và pipeline next-item ban đầu; không cần chạy chúng trước pipeline hiện tại và không trộn hai bộ mapping.

## Tài liệu và giới hạn

- [Báo cáo xử lý dữ liệu](docs/BAO_CAO_XU_LY_DU_LIEU_VI.md), [bản Word](docs/BAO_CAO_XU_LY_DU_LIEU_VI.docx).
- [Kiểm tra dữ liệu đã ghi nhận](reports/multibehavior/REVIEW_VI.md). Các kết quả lịch sử gắn với ngày kiểm tra tương ứng.

Visitor là mã ẩn danh; dữ liệu rất thưa, nhiều lịch sử ngắn và thiếu metadata. Mẫu âm không chứng minh người dùng không thích item; availability là trạng thái quan sát gần nhất. Cửa sổ nhãn có thể chồng nhau. Các thống kê độ phủ và cold-start là đặc điểm dữ liệu, không phải kết quả dự đoán.

Tài liệu đề tài gốc của sinh viên được giữ để tham khảo, không mô tả phần mềm hiện đã triển khai. Dữ liệu CSV/Parquet và môi trường Python không được commit.

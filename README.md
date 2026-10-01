# Gợi ý sản phẩm cá nhân hóa từ tương tác đa hành vi

Project sử dụng Retailrocket để khai thác ba loại tương tác **xem sản phẩm, thêm vào giỏ hàng và mua hàng**, với mục tiêu xếp hạng sản phẩm có khả năng được mua trong 7 ngày sau thời điểm gợi ý. Không giả định mọi lượt mua đều đi qua đủ ba hành vi hoặc theo một thứ tự cố định.

Repo cung cấp pipeline dữ liệu, kiểm thử và giao thức thực nghiệm. Các mô hình gợi ý sẽ được phát triển tiếp trên cùng bộ dữ liệu này.

## Bắt đầu

Mẫu CI cấu hình Python 3.12; môi trường đã chạy và kiểm tra dữ liệu hiện tại dùng Python 3.14.5. Các lệnh dưới đây chạy từ thư mục repo.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Trên Linux/macOS dùng `.venv/bin/python` thay cho `.\.venv\Scripts\python.exe`.

`requirements.txt` khai báo khoảng phiên bản thư viện. Để dùng đúng phiên bản thư viện của lần xử lý đã ghi nhận, cài bằng `python -m pip install -r requirements-lock.txt` trong môi trường đã kích hoạt; phiên bản Python được ghi trong [manifest](reports/multibehavior/manifest.json).

Đặt bốn file Retailrocket vào `data/raw/` theo [hướng dẫn dữ liệu](data/README.md), rồi chạy:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_multibehavior.py
.\.venv\Scripts\python.exe scripts/verify_artifacts.py
```

Pipeline đọc toàn bộ raw, ghi dữ liệu mới vào **`data/multibehavior/`**, báo cáo vào **`reports/multibehavior/`**. Các thư mục `data/processed`, `data/interim`, `data/splits` của notebook cũ không bị thay đổi. Cấu hình nằm trong [configs/multibehavior.json](configs/multibehavior.json).

CSV raw, Parquet, môi trường Python và trọng số mô hình **không được commit**. Thành viên tải cùng raw và tái tạo dữ liệu; manifest lưu SHA-256 để kiểm tra đúng phiên bản đầu vào/đầu ra. Không cần dữ liệu thật để chạy unit tests.

## Pipeline mới đã làm gì?

1. Bỏ trùng hoàn toàn và dòng không hợp lệ; giữ từng hành vi riêng, kể cả trên cùng sản phẩm.
2. Chuẩn hóa UTC, tạo session theo khoảng nghỉ >30 phút; lấy mốc phân vị 80% và 90% thời gian sự kiện để chia train/validation/test. Sau khi loại session qua ranh giới và query thiếu cửa sổ nhãn, số query không có tỷ lệ 80/10/10.
3. Loại session đi qua ranh giới tập khỏi history/labels mô hình.
4. Tạo một điểm dự đoán sau nhóm sự kiện có timestamp đầu tiên của mỗi session; không yêu cầu session có ít nhất hai item.
5. Dùng lịch sử đã quan sát tại thời điểm dự đoán; nhãn là các sản phẩm mua trong 7 ngày tiếp theo. Giữ query không mua; ghi rõ query bị loại do không quan sát đủ cửa sổ nhãn.
6. Tính số lần từng hành vi, số đếm 7/30 ngày, thời gian từ hành vi gần nhất. Bảng feature không chứa nhãn.
7. Giữ lịch sử `categoryid`/`available` và timestamp nguồn; lấy trạng thái không muộn hơn thời điểm gợi ý. Category không biết dùng `-1`, availability thiếu vẫn giữ riêng.
8. Tạo mapping chỉ từ train, bảng user–item và cạnh theo từng hành vi. Chỉ số mã hóa `user_idx`/`item_idx` dành `PAD=0`, `UNK=1`; chỉ số của ID có trong train bắt đầu từ 2. ID gốc được giữ nguyên.
9. Tạo mẫu âm tùy chọn cho query train có mua, tối đa 5 item/query; chỉ lấy sản phẩm đã biết và đủ điều kiện tại thời điểm đó. Query ít ứng viên có thể nhận ít mẫu hơn và được báo cáo.
10. Kiểm tra tính toàn vẹn, thời gian, bảo toàn số hành vi, nhãn/mẫu âm; lưu config, phiên bản thư viện, schema và checksum.

Chi tiết bắt buộc đọc trước khi huấn luyện: [giao thức dữ liệu](docs/DATA_PROTOCOL.md), [từ điển bảng](docs/DATA_DICTIONARY.md).

## Cấu trúc repo

```text
configs/                 Tham số xử lý và negative sampling
scripts/                 Pipeline, truy cập dữ liệu, xác minh artifact
tests/                   Kiểm thử độc lập với dữ liệu Retailrocket
notebooks/               Notebook khảo sát ban đầu và hướng dẫn dùng bộ mới
docs/                    Giao thức, từ điển dữ liệu
reports/                 Báo cáo audit ban đầu
reports/multibehavior/    Kết quả xử lý mới và manifest
data/raw/                Bốn CSV gốc, không theo dõi bằng Git
data/multibehavior/       Dữ liệu mới đã xử lý, không theo dõi bằng Git
```

Pipeline hiện tại là [scripts/prepare_multibehavior.py](scripts/prepare_multibehavior.py); notebook [02_multibehavior_overview.ipynb](notebooks/02_multibehavior_overview.ipynb) minh họa cách đọc kết quả. Hai notebook có tiền tố `01_` thuộc giai đoạn khảo sát và xử lý next-item ban đầu, không cần chạy trước pipeline hiện tại.

Mẫu GitHub Actions nằm ở [docs/ci-workflow.yml](docs/ci-workflow.yml). File này chưa nằm trong `.github/workflows/`, nên repo hiện chưa có workflow CI tự động từ mẫu đó; kết quả kiểm tra được nêu trong báo cáo là kết quả chạy tại máy.

## Dùng dữ liệu trong mô hình

```python
from pathlib import Path
import pandas as pd
from scripts.dataset import require_complete, history_for_query

data = require_complete(Path("data/multibehavior"))
events = pd.read_parquet(data / "events.parquet")
queries = pd.read_parquet(data / "splits/train_queries.parquet")
targets = pd.read_parquet(data / "splits/train_targets.parquet")
features = pd.read_parquet(data / "query_features.parquet")

query = queries.iloc[0]
history = history_for_query(events, query)
# labels và features được đọc riêng; không dùng target_count làm feature.
```

`interactions_train.parquet` và `behavior_edges_train.parquet` dành cho ALS/ItemKNN/mô hình đa nhiệm fit tại cuối train. Với mô hình học từng điểm dự đoán, dùng `query_features` và history tại thời điểm query; không ghép số đếm toàn bộ train vào một query sớm hơn.

`candidates_at_time()` trong `scripts/dataset.py` dựng tập ứng viên độc lập với target. Đây là hàm tham chiếu tính đúng; khi chạy đánh giá quy mô lớn cần duy trì trạng thái catalog hoặc cache phù hợp, thay vì đọc lại toàn bộ properties cho mỗi query.

## Hướng thực nghiệm

- Recent Popular, ItemKNN: baseline và xử lý user ít lịch sử.
- ALS chỉ mua và ALS gộp hành vi: đo giá trị tín hiệu phụ.
- Multi-task: học riêng view/cart/purchase, xếp hạng bằng nhánh purchase.
- Hybrid có fallback cho user mới; SASRec/MBGCN là hướng mở rộng.

So sánh bằng Recall/NDCG@10/20 trên **cùng query có nhãn mua, cùng target và cùng chính sách candidate**, kèm số query được chấm, độ phủ target, kết quả warm/cold và tỷ lệ query không mua. Tập candidate được dựng tại thời điểm từng query. Chọn tham số trên validation; test chỉ dùng báo cáo kết quả mô hình cuối. Xem quy tắc cho mô hình chỉ hỗ trợ item đã có trong train tại [giao thức dữ liệu](docs/DATA_PROTOCOL.md).

## Kết quả kiểm tra dữ liệu

Lần xử lý ngày 29/09/2026 ghi nhận 29 kiểm tra đạt và 20 file Parquet. Lần kiểm tra lại ngày 01/10/2026 có **51/51 kiểm tra dữ liệu đạt**, **11/11 unit tests đạt**; checksum của raw, đầu ra, cấu hình và mã pipeline khớp manifest. Kiểm tra độc lập lịch sử, recency và nhãn được thực hiện thêm trên 240 query lấy mẫu theo split và tình trạng có/không có nhãn mua; đây không phải kiểm tra độc lập từng feature của toàn bộ query.

Chi tiết: [báo cáo dữ liệu](reports/multibehavior/REVIEW_VI.md), [kết quả kiểm tra lại](reports/multibehavior/recheck_2026-10-01.json). Các kiểm tra này xác nhận dữ liệu theo giao thức hiện tại; repo chưa có kết quả đánh giá mô hình.

## Phạm vi và giới hạn

- `visitorid` là visitor ẩn danh; dữ liệu không xác nhận đây là cùng một người trên nhiều thiết bị.
- Không có đầy đủ exposure logs; mẫu âm không phải bằng chứng người dùng không thích sản phẩm.
- Availability là trạng thái ghi nhận gần nhất, không phải tồn kho trực tiếp.
- Query có thể có cửa sổ mua chồng nhau; báo cáo thêm metric trung bình theo user để tránh user hoạt động nhiều chi phối.
- Bộ mới sẵn sàng cho thực nghiệm theo giao thức đã nêu; chưa có kết quả chứng minh chất lượng mô hình hoặc tăng doanh thu.

Báo cáo `reports/data_audit_review_vi.md` mô tả **pipeline cũ** và làm mốc trước cải tiến. Các chỉ số cold-start/coverage trong đó thuộc next-item; không so trực tiếp với nhãn mua 7 ngày của bộ mới.

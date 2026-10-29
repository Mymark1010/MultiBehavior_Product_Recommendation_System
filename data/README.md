# Chuẩn bị dữ liệu Retailrocket

Nguồn dataset: [Retailrocket recommender system dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

Tải và giải nén bốn file sau vào `data/raw/`, giữ nguyên nội dung và tên:

```text
data/raw/events.csv
data/raw/item_properties_part1.csv
data/raw/item_properties_part2.csv
data/raw/category_tree.csv
```

Số dòng bản raw đã kiểm tra:

| File | Dòng, không gồm header |
|---|---:|
| events.csv | 2.756.101 |
| item_properties_part1.csv | 10.999.999 |
| item_properties_part2.csv | 9.275.903 |
| category_tree.csv | 1.669 |

Sau khi cài dependency và kích hoạt môi trường Python theo README chính, từ thư mục repo chạy `python scripts/prepare_nextstep.py`. Không cần chạy notebook cũ trước.

Manifest của lần chạy next-step mới được lưu tại `reports/nextstep/manifest.json`, gồm SHA-256 raw và đầu ra. `python scripts/verify_artifacts.py` kiểm tra kết quả đã tạo. Dữ liệu của nhà cung cấp giữ điều kiện sử dụng của nguồn gốc; repo này không phân phối lại raw.

Không commit `data/raw`, `data/interim`, `data/processed`, `data/splits` hoặc `data/nextstep`. Chỉ file README này được theo dõi bằng Git.

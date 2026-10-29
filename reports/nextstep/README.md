# Báo cáo next-step trên toàn RetailRocket

- `summary.json`: số liệu của lần chạy trên bốn CSV thật; quy mô query/target, split và các kiểm tra.
- `manifest.json`: schema, số dòng và SHA-256 raw/config/pipeline/reader/Parquet.
- `analysis.json`: kiểm tra độc lập 240 query/243 cặp target và thống kê độ dài prefix.
- `validation.json`: kết quả kiểm thử và xác minh toàn bộ artifact.
- `raw_source.json`: nguồn CSV và đối chiếu checksum.
- `notebook_validation.json`: các notebook đã thực thi toàn bộ cell.
- `REVIEW_VI.md`: bản tóm tắt số liệu và giới hạn ứng viên.

Chỉ có kiểm chứng dữ liệu; chưa có Recall/NDCG, mô hình huấn luyện hoặc demo.

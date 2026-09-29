# Các phần việc có thể chia cho thành viên

| Phần việc | Đầu vào | Đầu ra cần bàn giao |
|---|---|---|
| Dữ liệu | Raw + pipeline hiện tại | Feature mới đúng thời gian, tests và cập nhật schema |
| Baseline | Train interactions, queries, catalog | Recent Popular, ItemKNN, ALS; cùng evaluator |
| Đa hành vi | Behavior edges hoặc query history | Mô hình đa nhiệm; ablation chỉ mua / thêm view / thêm cart |
| Đánh giá | Queries + targets + candidate protocol | Recall/NDCG, warm/cold, coverage, latency |
| Demo/hệ thống | Model đã chọn + candidate builder | Luồng Top-K, fallback user mới và giải thích đơn giản |

Ưu tiên đầu tiên: evaluator và baseline dùng cùng giao thức. Không cần mỗi thành viên tự xử lý một bản dữ liệu khác nhau. Thống nhất config hash/manifest trước khi so sánh kết quả.

Model hoặc demo chưa được triển khai trong commit dữ liệu đầu tiên này.

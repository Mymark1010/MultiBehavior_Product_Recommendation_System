# Từ điển dữ liệu next-step — schema v2

Các đường dẫn tương đối với `data/nextstep/`. Số dòng thực tế chỉ xác định sau khi chạy pipeline trên nguồn tương ứng.

| Bảng | Đơn vị một dòng | Nội dung chính |
|---|---|---|
| events.parquet | Sự kiện raw khác biệt | visitorid, itemid, event, event_time UTC, event_id, session_id, split, user_idx, item_idx |
| user_mapping.parquet | Visitor train | visitorid → user_idx từ 2 |
| item_mapping.parquet | Item train | itemid → item_idx từ 2 |
| interactions_train.parquet | Visitor–item train | Số lần từng hành vi, điểm trọng số log, thời điểm cuối và recency_hours tại cutoff |
| behavior_edges_train.parquet | Visitor–item–behavior train | count, last_event_time; không phải đồ thị prefix |
| queries.parquet | Prefix sau một nhóm timestamp | query_id, session_id, visitorid, query_time, prefix_group_count, prefix_event_count, split, user_idx, cold_user; target_time/target_count/eligible_target_count chỉ phục vụ nhãn |
| nextstep_targets.parquet | Query–item ở nhóm kế | target_time, target_event_count, item_idx, cold_item; metadata tại query_time và candidate_at_query phục vụ phân tích |
| query_features.parquet | Query | ID, query_time, số nhóm/sự kiện prefix, view_count/addtocart_count/transaction_count, last_*_time, *_recency_seconds, has_*_history; không chứa nhãn |
| excluded_steps.parquet | Nhóm không tạo query | group_event_count; lý do no_next_group hoặc boundary_session |
| negative_samples_train.parquet | Query train–item âm được lấy mẫu | label=0, sampling_policy; có thể rỗng khi cấu hình tắt |
| property_history.parquet | Bản cập nhật thuộc tính | available/categoryid, property_time, numeric_value, valid_to |
| catalog.parquet | Item quan sát được | known_at, seen_in_train |
| category_tree.parquet | Danh mục | parentid, root_category_id, category_depth, category_path |
| products_train_snapshot.parquet | Item đã biết ở cutoff train | Metadata và cờ unknown tại cutoff, không thay metadata mọi query |
| splits/*_queries.parquet | Query một split | train, validation, test |
| splits/*_targets.parquet | Query–item một split | target có thể nhiều item; không tính mỗi dòng như một query riêng |

Pipeline tạo 14 bảng chính và 6 bảng split. `dataset_info.json` là marker schema/task, không phải bảng Parquet. `manifest.json` và `summary.json` ở `reports/nextstep/` chỉ xuất hiện sau lần chạy thành công.

Query ID là số thứ tự nhóm theo `(session_id, query_time)` từ 1 trước khi loại nhóm cuối/boundary; có thể không liên tục. Query ID không đồng nhất với session ID. Prefix không được nhân bản ra một bảng sự kiện lớn: dùng `history_for_query()` để lấy phần phiên đã quan sát.

`interaction_score = log(1+view) + 3×log(1+addtocart) + 5×log(1+transaction)` là quy ước tổng hợp train, không phải xác suất hoặc điểm dự đoán. Trọng số cần được kiểm chứng khi huấn luyện.

# Từ điển dữ liệu mới

Tất cả đường dẫn dưới đây tương đối với `data/multibehavior/`. Schema và số dòng chính xác xem manifest sau khi chạy pipeline.

| File | Đơn vị một dòng | Nội dung/chú ý |
|---|---|---|
| events.parquet | Một sự kiện khác biệt | ID raw, event_id, loại hành vi, UTC time, session, split, user_idx/item_idx; chưa gộp hành vi |
| user_mapping.parquet | Một visitor train | visitorid → user_idx từ 2 |
| item_mapping.parquet | Một item train | itemid → item_idx từ 2 |
| interactions_train.parquet | Một user–item train | Đếm ba hành vi, baseline score, thời gian cuối/recency từng hành vi tại train cutoff |
| behavior_edges_train.parquet | Một user–item–behavior train | Số lần và thời điểm cuối; cho mô hình đồ thị/đa nhiệm |
| queries.parquet | Một điểm gợi ý | query_id, session_id, visitorid, query_time, split, label_end, target_count, cold_user, eligible_target_count |
| query_features.parquet | Một query | Số đếm history toàn bộ/7d/30d, last time/recency và cờ từng hành vi; không có label |
| purchase_targets.parquet | Một query–item được mua | Thời gian mua đầu, số events mua, metadata as-of, cold_item, candidate_at_query; dùng đánh giá |
| excluded_queries.parquet | Một query không đủ cửa sổ nhãn | Giữ lý do loại; không phải negative |
| negative_samples_train.parquet | Một query–item được lấy mẫu | label=0 và chính sách lấy mẫu; chỉ query train có mua |
| property_history.parquet | Một bản cập nhật item–property | available/categoryid, value, property_time, valid_to; không lấp thiếu bằng tương lai |
| catalog.parquet | Một item quan sát được | known_at, seen_in_train; phải lọc known_at tại lúc dùng |
| category_tree.parquet | Một danh mục | parent, root, depth, path đã kiểm tra chu trình/tham chiếu |
| products_train_snapshot.parquet | Một item đã biết tại cuối train | Metadata tại train cutoff, cờ unknown, candidate_at_query; không phải trạng thái của mọi ngày test |
| splits/*_queries.parquet | Query của một split | train, validation hoặc test |
| splits/*_targets.parquet | Target của một split | Có thể nhiều target/query; không tính metric mỗi dòng như query riêng |

Các cột count dùng hậu tố `_count`, `_count_7d`, `_count_30d`. Các thời điểm là UTC, recency tính theo giờ. `NaN` recency nghĩa là chưa có hành vi, category -1 nghĩa là không biết, available NA nghĩa là chưa biết trạng thái.

History không bị nhân bản ra một bảng rất lớn cho từng query: lấy từ events bằng visitorid và `event_time <= query_time`, loại boundary. Dùng `history_for_query()` làm tham chiếu và tạo batch/index phù hợp trong model loader.

Không dùng `session_length` đầy đủ để dự đoán khi session chưa kết thúc. Bộ events mới không tạo feature này.

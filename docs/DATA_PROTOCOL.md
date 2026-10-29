# Giao thức next-step schema v2

## Đơn vị quan sát

Session mới khi cùng visitor nghỉ hơn 30 phút; timestamp raw mili-giây đổi UTC. Tất cả sự kiện có cùng session và timestamp là một nhóm, không tạo thứ tự nhân quả giả trong nhóm. `event_id` chỉ giúp tái lập thứ tự lưu.

Một phiên có m nhóm timestamp tạo m−1 query. Query sau nhóm g dùng toàn bộ nhóm 1…g làm prefix; target là tập item khác nhau của nhóm g+1, với mọi loại hành vi. Target lặp item trong prefix là hợp lệ. `target_event_count` ghi số sự kiện của item ở nhóm target, nhưng relevance xếp hạng chỉ là nhị phân theo item.

Nhóm cuối không có target; phiên một nhóm không tạo mẫu có nhãn. Các nhóm này có trong `excluded_steps`, lý do `no_next_group`. Mọi nhóm của phiên cắt qua split bị loại với lý do `boundary_session`.

## Chia tập theo thời gian

Phân vị 80% và 90% của thời gian sự kiện xác định train_end và validation_end. Phiên chỉ thuộc một split; nếu min_time <= cutoff < max_time thì toàn phiên là boundary_drop. Không cắt phiên thành train/test. Số query sau lọc không nhất thiết theo tỷ lệ 80/10/10. Cutoff thực của lần chạy được ghi trong manifest; không dùng ngày cuối query của một giao thức khác làm cutoff.

## Prefix và tách nhãn

`history_for_query(events, query)` chỉ trả sự kiện của cùng session có event_time <= query_time, không boundary. Khách quay lại ở phiên khác bắt đầu một prefix mới; visitor chỉ là mã liên kết, không phải hồ sơ cá nhân đã xác thực.

`query_features` gồm số nhóm/sự kiện đã quan sát, số đếm ba hành vi, thời điểm gần nhất và recency_seconds. Không có target_time, target_count hoặc eligible_target_count. Các trường này thuộc query/target để kiểm tra nhãn; không được làm feature. Recency thiếu là chưa có hành vi, không tự điền bằng 0. Không dùng độ dài phiên hoàn chỉnh khi prefix chưa kết thúc.

`interactions_train` và `behavior_edges_train` là tổng hợp cuối train dành cho khối học mô hình, không phải feature tại mọi query sớm hơn train_end. Cạnh visitor–item–behavior không phải đồ thị chuyển tiếp item của SR-GNN; đồ thị prefix phải dựng riêng.

## Mapping và ứng viên

Mapping chỉ lấy ID có trong train; PAD=0, UNK=1, ID biết bắt đầu từ 2. ID raw luôn được giữ; nhiều item mới có idx=1 vẫn là các item khác nhau, không gộp chúng.

Catalog hợp log và thuộc tính; known_at là thời điểm quan sát đầu tiên. Ứng viên tại t có known_at<=t và available gần nhất là 1 hoặc unknown. Metadata chỉ lấy bản cập nhật gần nhất có property_time<=t. Category thiếu/ngoài cây nhận -1 kèm cờ. Không lấp bằng tương lai, không tự thêm target vào candidate, không mặc định loại item đã xem. available là trạng thái ghi nhận, không bảo đảm tồn kho trực tiếp.

Target ngoài ứng viên hoặc ngoài train vẫn được giữ để báo cáo tỷ lệ không phục vụ được. Mô hình chỉ có embedding train cần công bố chính sách cold-item. Phương án dự phòng từ Popularity cũng chỉ có thông tin từ train, không giải quyết được mọi item mới.

## Mẫu âm train tùy chọn

`negative_samples_per_query=0` mặc định tắt. Khi bật, lấy tối đa k item train đã biết và hợp lệ tại t, loại **toàn bộ tập target của nhóm kế**. Âm chỉ nghĩa là không được quan sát trong nhóm kế, không chứng minh không thích. Mẫu có thể thiếu k do ít ứng viên; công bố số thiếu. Không dùng nhãn validation/test để lọc âm train và không dùng mẫu âm này làm ứng viên đánh giá mặc định.

## Đánh giá dự kiến

Mọi query có target không rỗng. Cùng query, target, ứng viên và K=10/20 cho các phương pháp. Recall, NDCG nhị phân, HR, MRR tính theo query rồi lấy trung bình; chỉ số chính dự kiến NDCG@20. Công bố độ phủ, cold-item, dự phòng, thời gian và phân nhóm prefix. Prefix cùng phiên phụ thuộc nhau; bootstrap theo session/visitor nếu cần khoảng tin cậy. Chỉ tune trên validation, đóng băng trước test; công bố lịch sử sử dụng test nếu đã từng khảo sát.

## Tái lập và phạm vi kiểm chứng

Đầu ra `data/nextstep/`, báo cáo `reports/nextstep/`. `dataset_info.json` ghi task/schema để reader từ chối bộ sai giao thức. `INCOMPLETE` tồn tại trong quá trình chạy hoặc sau lỗi; chỉ xóa khi kiểm tra và ghi file thành công. Không chạy đồng thời hai pipeline vào một đích.

Manifest chỉ chứng nhận danh sách artifact do chính lần chạy tạo, gồm schema, số dòng, SHA-256 raw, config, pipeline, reader và phiên bản thư viện. `require_complete()` chỉ kiểm tra task/schema/marker; dùng `verify_artifacts.py` để đối chiếu checksum. Đổi phiên bản thư viện có thể đổi checksum Parquet dù dữ liệu logic tương đương.

Đã chạy trên bốn CSV thật: 991.059 query, 993.336 cặp query–item target, 38 kiểm tra pipeline đạt và 20 Parquet khớp manifest. Bộ 30 kiểm thử dùng fixture để kiểm chứng các trường hợp biên. Audit độc lập tái tính prefix/nhãn/metadata trên 240 query và 243 cặp target thực tế. Kiểm tra lấy mẫu không đồng nghĩa tái tính độc lập mọi feature trên toàn dữ liệu.

## Số liệu ứng viên và lặp item

407.653 cặp target là item đã xuất hiện trong prefix; không lọc mặc định item đã xem. Độ phủ cặp target trong ứng viên ở test là khoảng 79,17%; trung bình tỷ lệ target hợp lệ theo query là khoảng 79,14%. Đây là giới hạn dữ liệu ứng viên trước khi xét K, không phải kết quả của mô hình. Mọi target vẫn nằm trong mẫu số chính, kể cả target ngoài train/ứng viên.

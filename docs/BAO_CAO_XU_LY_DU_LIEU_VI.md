# Báo cáo xử lý dữ liệu RetailRocket — next-step theo phiên

Cập nhật theo lần chạy thật ngày 29/10/2026. Nguồn số liệu: `reports/nextstep/summary.json`, `manifest.json`, `analysis.json`, `validation.json`. Không có kết quả xếp hạng mô hình trong báo cáo này.

## 1. Bài toán và dữ liệu nguồn

Một phiên gồm các nhóm timestamp g1,…,gm. Mỗi query dùng prefix g1,…,gt; target là tập item phân biệt ở g(t+1), không giới hạn ở hành vi mua. Một phiên m nhóm tạo m−1 query. Phiên chỉ một nhóm không tạo mẫu có nhãn nhưng vẫn giữ trong events. Sự kiện đồng thời không được tự gán thứ tự nhân quả.

Bốn CSV nguồn có 2.756.101 sự kiện, 20.275.902 dòng thuộc tính và 1.669 dòng cây danh mục. Nguồn: bộ RetailRocket công khai trên Kaggle. Bốn SHA-256 raw đều khớp chuẩn đối chiếu đã ghi nhận; checksum cụ thể được lưu trong manifest và raw_source.json để tái lập.

## 2. Làm sạch và chia tập

Giữ view, addtocart, transaction; loại thiếu trường bắt buộc và 460 dòng trùng tuyệt đối, còn 2.755.641 sự kiện. Transaction phải có transactionid, view/giỏ được phép thiếu. UTC; session mới nếu gap >30 phút.

Train_end: 2015-08-18 04:23:20.445000+00:00. Validation_end: 2015-09-02 17:49:14.032000+00:00. Mọi sự kiện của phiên cắt qua split bị gắn boundary_drop (328 sự kiện). Tỷ lệ query sau lọc không bắt buộc 80/10/10.

## 3. Query, prefix và nhãn

Tổng 1.761.675 phiên, 2.753.015 nhóm timestamp; 991.059 query và 993.336 cặp target. 1.836 query có nhiều target. Query ID là ID nhóm xác định, không phải session ID. Target_event_count ghi số sự kiện của item ở nhóm kế, còn relevance là nhị phân theo item.

| Tập | Query next-step | Phiên tạo query | Cặp query–item target | Query visitor ngoài train | Độ phủ cặp target trong ứng viên |
|---|---:|---:|---:|---:|---:|
| Train | 806.025 | 306.332 | 807.908 | 0,00% | 82,27% |
| Validation | 92.566 | 38.290 | 92.794 | 77,07% | 79,90% |
| Test | 92.468 | 37.864 | 92.634 | 83,60% | 79,17% |

Có 1.379.142 phiên chỉ một nhóm (78,29%). Nhóm cuối/phiên boundary nằm trong excluded_steps; không gán thành query nhãn rỗng.

Prefix lấy cùng session và không vượt query_time; không dùng phiên trước của visitor. Feature gồm số nhóm/sự kiện đã quan sát, số đếm hành vi, lần gần nhất và recency_seconds. Không đưa target_time/target_count/eligible_target_count vào feature. 407.653 cặp target (41,04%) lặp item trong prefix, nên không loại mặc định sản phẩm đã xem.

## 4. Metadata và ứng viên

Đọc thuộc tính theo chunk, giữ 2.291.853 bản cập nhật categoryid/available; catalog hợp log và thuộc tính có 466.868 item. Mapping chỉ từ train: 1.123.755 visitor và 212.915 item. PAD=0, UNK=1; ID raw luôn được giữ.

Metadata dùng bản gần nhất có property_time<=query_time. Available unknown vẫn được giữ, available=0 bị loại. Item phải được biết tại t. Category thiếu/ngoài cây nhận -1 kèm cờ. Snapshot cuối train không thay được metadata từng query. Cạnh visitor–item–behavior không phải đồ thị chuyển tiếp prefix cho SR-GNN.

Độ phủ target trong ứng viên ở test là 79,17%; trung bình tỷ lệ target có thể phục vụ theo query là 79,14%. Đây là hai thống kê giới hạn dữ liệu khác nhau, chưa xét K và không phải Recall của mô hình. Target ngoài ứng viên hoặc train vẫn được giữ trong mẫu số chính.

Mẫu âm tùy chọn mặc định tắt; tệp hiện có 0 dòng. Khi bật, âm là item hợp lệ tại t không nằm trong toàn bộ nhóm target, không phải nhãn không thích. Không dùng mẫu âm train làm catalog đánh giá.

## 5. Kiểm chứng

30 kiểm thử tự động đạt, 38 kiểm tra dữ liệu đạt trên toàn bộ lần chạy, 20 Parquet khớp checksum raw/config/pipeline/reader. Kiểm tra độc lập trên 240 query và 243 cặp target: tái tính nhóm kế, tập item đúng, đếm/recency prefix, metadata và eligibility tại t. Kiểm tra độc lập feature/metadata là lấy mẫu xác định, không phải phép tái tính độc lập mọi ô trên toàn dữ liệu.

Marker INCOMPLETE ngăn đọc sau lần chạy lỗi. dataset_info.json xác định task/schema. require_complete() không thay được verifier checksum. Thử nghiệm cố ý sửa target/Parquet đã xác nhận audit và verifier phát hiện lỗi.

## 6. Danh mục artifact

| File trong data/nextstep/ | Số dòng |
|---|---:|
| `events.parquet` | 2.755.641 |
| `user_mapping.parquet` | 1.123.755 |
| `item_mapping.parquet` | 212.915 |
| `interactions_train.parquet` | 1.713.144 |
| `behavior_edges_train.parquet` | 1.769.390 |
| `queries.parquet` | 991.059 |
| `nextstep_targets.parquet` | 993.336 |
| `query_features.parquet` | 991.059 |
| `excluded_steps.parquet` | 1.761.956 |
| `property_history.parquet` | 2.291.853 |
| `catalog.parquet` | 466.868 |
| `category_tree.parquet` | 1.669 |
| `products_train_snapshot.parquet` | 456.974 |
| `negative_samples_train.parquet` | 0 |
| `splits/train_queries.parquet` | 806.025 |
| `splits/train_targets.parquet` | 807.908 |
| `splits/validation_queries.parquet` | 92.566 |
| `splits/validation_targets.parquet` | 92.794 |
| `splits/test_queries.parquet` | 92.468 |
| `splits/test_targets.parquet` | 92.634 |

## 7. Giới hạn và bước tiếp theo

Visitor là mã ẩn danh; không có log hiển thị, nhãn không thích hoặc tên/ảnh sản phẩm đầy đủ. Available là trạng thái ghi nhận, không bảo đảm tồn kho trực tiếp. Prefix trong cùng phiên phụ thuộc nhau; phân tích hoặc bootstrap cần theo session/visitor khi phù hợp.

Các mô hình Popularity, ItemKNN, Session-KNN/V-SKNN, GRU4Rec/SR-GNN, bộ đánh giá xếp hạng và demo là thiết kế giai đoạn tiếp theo; chưa huấn luyện hoặc có kết quả chất lượng mô hình.

Bước kế tiếp: xây bộ đánh giá chung trên query/target/ứng viên này, triển khai Popularity, ItemKNN và Session-KNN; chọn một mô hình nâng cao; ablation có/không hành vi; chọn bằng validation trước khi đánh giá test. Không dùng số liệu kiểm tra dữ liệu để tuyên bố chất lượng mô hình hoặc tăng doanh thu.

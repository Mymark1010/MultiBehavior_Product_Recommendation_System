# Đề cương: gợi ý sản phẩm đa hành vi theo phiên với RetailRocket

Vũ Hải Minh — MSSV 20235166. Bản thống nhất phạm vi next-step, cập nhật 29/10/2026.

## 1. Bài toán thực tế

Khách thương mại điện tử thường chưa có lịch sử dài nhưng đã để lại tín hiệu trong phiên hiện tại. Mục tiêu là tận dụng cả xem, thêm giỏ và mua để gợi ý sản phẩm cho tương tác kế tiếp, cập nhật sau mỗi nhóm timestamp đã quan sát. Đầu ra kỳ vọng là Top-10/20 item hợp lệ tại thời điểm đó. Không đồng nhất mục tiêu dự đoán tương tác với dự đoán mua hay tăng doanh thu.

Ví dụ: xem A lúc 09:00; xem B lúc 09:02; thêm giỏ B và xem C lúc 09:04. Query sau 09:02 có prefix gồm A, B và target {B,C}. Cho phép item lặp lại. Các sự kiện đồng thời không được ép thành thứ tự giả.

## 2. Dữ liệu và giới hạn

RetailRocket có 2.756.101 sự kiện raw, 1.407.580 visitor và 235.061 item trong log theo báo cáo khảo sát. Sau khử 460 dòng trùng còn 2.755.641 sự kiện: 2.664.218 xem, 68.966 thêm giỏ, 22.457 mua. Các số liệu được xác nhận lại trong lần chạy đầy đủ trên bốn CSV ngày 29/10/2026. Hai bảng thuộc tính có tổng 20.275.902 dòng; cây danh mục có 1.669 dòng.

ID và nhiều giá trị thuộc tính đã ẩn danh/băm. Không có tên/ảnh/mô tả đầy đủ, không có exposure log hoặc nhãn không thích thật. 71,16% visitor chỉ có một sự kiện; nhiều phiên không tạo được next-step target. Cần công bố số mẫu sau lọc và có dự phòng, không mặc định loại hết khách ít lịch sử.

## 3. Phần đã cài đặt

Pipeline `scripts/prepare_nextstep.py` schema v2 làm sạch, UTC, session >30 phút, split thời gian 80%/90%, loại phiên qua split, nhóm timestamp, query–target kế tiếp và prefix cùng phiên. Metadata tra gần nhất không vượt query_time, giữ unknown. Mapping chỉ từ train. `build_prefix_features()` không chứa nhãn. `sample_train_negatives()` tùy chọn lấy item không nằm trong toàn bộ nhóm target, mặc định tắt.

Đầu vào là bốn CSV raw và cấu hình `configs/nextstep.json`. Đầu ra là 20 bảng theo thiết kế trong `data/nextstep/`, gồm events, mapping, các bảng train, queries, nextstep_targets, query_features, excluded_steps, metadata và sáu bảng split. Reader và verifier kiểm task/schema, marker hoàn tất và checksum.

Đã chạy 30 kiểm thử tự động và pipeline trên toàn bộ CSV thật: 991.059 query, 993.336 cặp target; 38 kiểm tra dữ liệu đạt, 20 Parquet khớp manifest. Audit độc lập 240 query/243 cặp target đạt. Fixture còn kiểm tra tái chạy, đọc prefix/ứng viên và cố ý sửa artifact để xác nhận phát hiện lỗi. Đây là bằng chứng dữ liệu, chưa phải chất lượng mô hình.

## 4. Phương pháp dự kiến

Popularity là đối chứng tối thiểu và dự phòng. ItemKNN dùng tương đồng item từ ma trận phiên×item train, chấm điểm dựa trên prefix và thử trọng số hành vi. Session-KNN/V-SKNN dùng các phiên train tương tự để khai thác ngữ cảnh mà không cần visitor từng xuất hiện.

Chọn một mô hình nâng cao: GRU4Rec hoặc SR-GNN. GRU4Rec tổng hợp embedding item+hành vi trong mỗi nhóm timestamp, đọc các nhóm theo thời gian để tạo vector phiên. SR-GNN cần dựng đồ thị chuyển tiếp chỉ từ prefix, có quy tắc cho nhóm đồng thời. Đây là điều chỉnh thiết kế cần kiểm chứng, chưa có huấn luyện. Target dạng tập có thể dùng cross-entropy với nhãn đều trên các item đúng; nếu lấy mẫu loss phải loại toàn bộ target khỏi âm.

## 5. Đánh giá dự kiến

Cùng query, target, ứng viên và K=10/20 cho mọi phương pháp. Train học thống kê và tham số; validation chọn cấu hình/dừng sớm; đóng băng trước test. Ứng viên có known_at<=t và available gần nhất không phải 0; giữ unknown và item từng xem. Giữ target ngoài ứng viên/train để phản ánh độ phủ thực tế. Ưu tiên toàn tập ứng viên; nếu lấy mẫu phải ghi riêng và dùng cùng mẫu/seed.

Recall@K = |L∩G|/|G|; HR@K cho biết có hit hay không; MRR@K là nghịch đảo hạng hit đầu tiên; NDCG@K = DCG/IDCG với relevance nhị phân và chiết khấu log2(rank+1). Chỉ số chính dự kiến NDCG@20. Tính từng query rồi lấy trung bình, báo cáo thêm theo session/visitor khi phù hợp.

Thí nghiệm loại bỏ thành phần: cùng mô hình có/không loại hành vi, so ảnh hưởng thứ tự nhóm timestamp; giữ dữ liệu và ngân sách tinh chỉnh. Phân tích prefix ngắn/dài, visitor mới/cũ, item phổ biến/hiếm. Đo độ phủ, dự phòng, thời gian học và độ trễ p95. Không kết luận trước mô hình sâu sẽ thắng KNN.

## 6. Demo và lộ trình

Demo dự kiến nhận prefix (item, event, timestamp), session/thời điểm và K; trả rank, item_id, score tương đối, category nếu biết và cờ dự phòng. Chỉ đưa lý do có căn cứ; không tự tạo tên/ảnh sản phẩm hoặc coi score là xác suất mua.

Bước tiếp theo: dùng bộ next-step đã tạo để triển khai bộ đánh giá cùng ba đối chứng; huấn luyện một mô hình chính; chọn cấu hình trên validation; đánh giá, phân tích và demo. Chưa có checkpoint, kết quả xếp hạng hoặc API đã triển khai.

## Tài liệu tham khảo

Các báo cáo Word do người dùng cung cấp là nguồn ý tưởng và khảo sát; trạng thái cài đặt lấy từ mã nguồn hiện tại. Các tuyên bố tiến độ cũ đã được điều chỉnh theo phạm vi mới.

- RetailRocket dataset: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
- Repository: https://github.com/Mymark1010/MultiBehavior_Product_Recommendation_System
- Hidasi et al. Session-based Recommendations with Recurrent Neural Networks. ICLR 2016. https://arxiv.org/abs/1511.06939
- Wu et al. Session-based Recommendation with Graph Neural Networks. AAAI 2019. https://arxiv.org/abs/1811.00855
- Ludewig & Jannach. Evaluation of Session-based Recommendation Algorithms. UMUAI 2018. https://arxiv.org/abs/1803.09587
- Järvelin & Kekäläinen. Cumulated Gain-based Evaluation of IR Techniques. TOIS 2002.
- Krichene & Rendle. On Sampled Metrics for Item Recommendation. KDD 2020.

## Kết quả dữ liệu và giới hạn thực nghiệm

Train/validation/test có lần lượt 806.025 / 92.566 / 92.468 query và 807.908 / 92.794 / 92.634 cặp target. Có 407.653 cặp target lặp item trong prefix. Độ phủ cặp target trong ứng viên test là 79,17%; đây không phải Recall của mô hình. Target ngoài train hoặc ngoài ứng viên vẫn được giữ trong mẫu số.

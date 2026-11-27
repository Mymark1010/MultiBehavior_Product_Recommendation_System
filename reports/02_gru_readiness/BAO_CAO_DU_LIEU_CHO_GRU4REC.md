# Rà soát dữ liệu huấn luyện GRU4Rec theo phiên

**Ngày kiểm tra:** 13/11/2026. Đọc trực tiếp các bảng Parquet đã xử lý và cấu hình chia tập; không thay đổi dữ liệu, ngưỡng phiên hay các tập chia. Kết quả chi tiết: [metrics.json](metrics.json). Mã tái lập: [audit_gru_readiness.py](../../scripts/audit_gru_readiness.py).

## 1. Kết luận

**Dữ liệu đủ lớn để bắt đầu triển khai và thực nghiệm GRU4Rec theo phiên. Chưa có căn cứ khẳng định mô hình sẽ đạt chất lượng tốt hoặc vượt KNN.** Đây là nhận định từ lượng tín hiệu học và phân bố dữ liệu, không phải kết quả huấn luyện hay một ngưỡng tối thiểu chính thức.

Train có **306.332 phiên với ít nhất hai nhóm timestamp**, tạo **806.025 vị trí dự đoán**. Những phiên này cùng huấn luyện một bộ trọng số; không học một mô hình riêng cho từng phiên. Vì vậy, nhiều phiên ngắn không đồng nghĩa thiếu toàn bộ dữ liệu học.

Khó khăn chính là **phân bố item thưa, nhiều item chưa có target huấn luyện và item mới xuất hiện theo thời gian**. Cần đánh giá trung thực các giới hạn này; không kéo dài phiên hoặc loại đáp án khó chỉ để tăng điểm.

## 2. Đếm riêng từng tập

Định nghĩa dùng trong báo cáo: một query dự đoán tập item phân biệt của nhóm timestamp kế tiếp từ tiền tố cùng phiên. Phiên có n nhóm tạo n−1 query nếu n ≥ 2. Các số là vị trí học tiềm năng, chưa phải bộ mẫu/model đã chạy; chưa áp dụng bộ lọc metadata/availability.

| Chỉ tiêu | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Tổng phiên | 1.396.273 | 182.534 | 182.821 |
| Phiên chỉ một nhóm, không tạo được cặp kế tiếp | 1.089.941 | 144.244 | 144.957 |
| Phiên đủ ít nhất hai nhóm | **306.332** | **38.290** | **37.864** |
| Query dự kiến | **806.025** | **92.566** | **92.468** |
| Query có prefix ít nhất hai nhóm | 499.693 | 54.276 | 54.604 |
| Target có nhiều item | 1.519 | 177 | 140 |

47 phiên qua biên thuộc boundary_drop được kiểm kê riêng, không góp vào train/validation/test. Chia tập giữ nguyên phiên; bước chia không cắt ngắn từng phiên. Ngưỡng >30 phút xác định phiên theo khoảng nghỉ giữa hai sự kiện liên tiếp, không giới hạn tổng độ dài phiên bằng 30 phút.

## 3. Phiên train ngắn đến mức nào?

Trong 306.332 phiên đủ tạo mẫu:

- **167.204 phiên có đúng hai nhóm**, khoảng 54,6%; mỗi phiên chỉ cung cấp một vị trí dự đoán.
- Độ dài trung bình **3,63 nhóm**, trung vị **2 nhóm**; 90% không vượt 6 nhóm, 99% không vượt 20 nhóm.
- **499.693 query**, khoảng 62,0%, có prefix ít nhất hai nhóm. Có **229.405 query** với prefix ít nhất năm nhóm. Vẫn có tín hiệu để kiểm tra việc dùng lịch sử thay vì chỉ item cuối.
- 125 phiên dài hơn 100 nhóm đóng góp 18.348 query; dài nhất 417 nhóm. Nhóm 1% phiên đủ mẫu dài nhất đóng góp khoảng 14,53% query. Khi dùng batch có padding, cần để ý độ dài và bộ nhớ, không tự động cắt phiên theo số liệu này.

**Tương tác lặp:** 94.157 phiên train đủ mẫu chỉ chứa một item phân biệt; 41,01% lượt item trong target từng có mặt ở prefix. Đây có thể là xem lại hoặc đổi hành vi trên cùng sản phẩm. Cần phân tích riêng target lặp và target mới trong phiên; không mặc định mọi query đều yêu cầu khám phá item khác.

## 4. Điểm yếu lớn hơn: độ phủ tín hiệu học theo item

| Chỉ tiêu trong train | Số item |
| --- | ---: |
| Item xuất hiện ở bất kỳ phiên train nào | 212.915 |
| Item có mặt trong các phiên ít nhất hai nhóm | 120.576 |
| Item chỉ xuất hiện ở phiên một nhóm | 92.339 |
| Item xuất hiện làm đầu vào ở một vị trí học | 110.117 |
| Item xuất hiện làm target dương | 104.450 |
| Item chưa từng là target dương | 108.465 |

Trong 104.450 item có target dương, **74.629 item chỉ có từ 1 đến 5 lượt target**, khoảng 71,4%. Số đếm là số cặp (query, item đúng), không đếm lặp cùng item trong một target.

Giữ một item trong vocabulary không đồng nghĩa item đó đã có tín hiệu học tốt. Item chưa có target dương không nhận được tín hiệu trực tiếp để nâng điểm trong vai trò đáp án; tùy cấu hình vẫn có thể chịu cập nhật do xuất hiện ở đầu vào, mẫu âm hoặc chia sẻ trọng số. Không nên diễn giải là mọi tham số của item đó hoàn toàn không được cập nhật.

## 5. Item mới ở validation/test

Mẫu số là **lượt item trong target**: mỗi item phân biệt của mỗi nhóm target được tính một lần. Đây không phải tỷ lệ item mới trên toàn danh mục hay tỷ lệ query.

| Vocabulary tham chiếu | Validation ngoài vocabulary | Test ngoài vocabulary |
| --- | ---: | ---: |
| Tất cả item xuất hiện trong train | 4.923 / 92.794 = **5,31%** | 8.639 / 92.634 = **9,33%** |
| Chỉ item thuộc phiên train đủ ít nhất hai nhóm | 8.715 / 92.794 = **9,39%** | 12.700 / 92.634 = **13,71%** |

Với vocabulary toàn train, có 4.915 query validation và 8.613 query test mà toàn bộ target đều ngoài vocabulary. Mô hình chỉ chấm item trong vocabulary không thể hit các query đó. Nếu dựng vocabulary sau khi loại phiên một nhóm, độ phủ giảm thêm; cần chốt rõ chính sách này trước train.

Đây là kiểm tra khả năng phủ dữ liệu, không phải kết quả dự đoán. Không dùng thống kê test để chọn cấu hình/loss/ngưỡng lọc; giữ toàn bộ target theo giao thức đánh giá đã chọn và báo cáo tỷ lệ ngoài vocabulary. Metadata có thể tạo thêm hạn chế về ứng viên, chưa được tính trong lần rà soát này.

## 6. Phân biệt với tái lập GRU4Rec gốc

Bài gốc loại phiên một sự kiện trong thí nghiệm, dùng chuỗi item và dự đoán một item kế tiếp. Số 806.025 ở trên dựa trên **nhóm timestamp của project**, không được gọi là số mẫu tái lập nguyên xi bài báo. Hiện có 1.519 target train nhiều item (0,188%); cần chốt giao thức xử lý trước khi đưa vào mã gốc, tránh tự đặt thứ tự giữa các sự kiện đồng thời.

Để kiểm tra riêng lượng tín hiệu xem, giữ các nhóm có view trong chính các phiên đã tạo thì còn **294.469 phiên có ít nhất hai nhóm view**, tương ứng **739.654 vị trí dự đoán nhóm view kế tiếp**. Đây chỉ là thống kê bổ sung, chưa tạo tập view-only mới và chưa tái tạo phiên sau lọc. Nó cho thấy lượng dữ liệu vẫn đáng kể ngay cả khi xét riêng lượt xem.

Tham chiếu: [Hidasi và cộng sự, GRU4Rec 2016, mục 3.1 và 4](https://arxiv.org/html/1511.06939v4). Không so quy mô hay kết quả trên RetailRocket với số liệu bài gốc để suy ra cùng mức chất lượng.

## 7. Các bước cần làm để xác nhận bằng thực nghiệm

1. **Giữ chia tập hiện tại**, không tăng ngưỡng 30 phút chỉ để làm phiên dài hơn. Hoàn thiện bộ mẫu và quy định vocabulary/target trước khi train.
2. Chỉ dùng phiên có ít nhất hai bước cho loss dự đoán kế tiếp. Giữ sự kiện của phiên một bước cho các mục đích thống kê hoặc mô hình đối chứng phù hợp, không xóa khỏi raw.
3. Chạy thử GRU một lớp ở quy mô nhỏ để xác nhận reset trạng thái, thứ tự input–target, loss và cách lấy mẫu; sau đó mới train toàn bộ tập train. Chọn kích thước mạng theo đo bộ nhớ và validation, không có bảo đảm một cấu hình cụ thể tối ưu.
4. So chất lượng validation với Popularity, ItemKNN và Session-KNN trên cùng query/ứng viên. Phân tích theo độ dài prefix, độ phổ biến item, target lặp và target mới trong phiên.
5. Theo dõi cả train và validation để phát hiện học quá sát train. Chốt cấu hình trước đợt test cuối; không kết luận đủ chất lượng chỉ vì loss train giảm.

**Có thể báo cáo với thầy:** “Sau khi loại các phiên không tạo được mẫu dự đoán, tập train vẫn có khoảng 306 nghìn phiên và 806 nghìn vị trí học. Em đánh giá đủ quy mô để triển khai thử GRU4Rec. Tuy nhiên dữ liệu thưa theo sản phẩm và có item mới ở tập sau, nên chất lượng cần được xác nhận qua thực nghiệm so sánh, chưa thể cam kết GRU vượt các đối chứng.”

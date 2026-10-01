# Giao thức v1: gợi ý mua từ lịch sử đa hành vi

## 1. Thời gian và đơn vị dự đoán

- Timestamp raw millisecond chuyển UTC; cutoff quantile được làm tròn về millisecond.
- Session mới nếu cùng visitor nghỉ hơn 30 phút. Mỗi session chỉ nằm trong một split; session qua ranh giới mang `boundary_drop` và không tham gia history/label.
- Tạo một query cho mỗi session không qua ranh giới split, ngay sau khi quan sát **toàn bộ sự kiện có timestamp đầu tiên** của session đó. Chỉ giữ query có đủ cửa sổ nhãn trong split. `query_id = session_id`.
- History gồm các sự kiện không thuộc boundary của cùng visitor có `event_time <= query_time`, kể cả các session trước.
- `event_id` là chỉ số dòng raw, dùng tái lập thứ tự lưu. Không diễn giải thứ tự giữa sự kiện cùng timestamp là quan hệ nhân quả.
- Thời điểm query không được chọn theo việc visitor sẽ mua hay theo số sản phẩm cuối session. Việc giữ/loại session qua ranh giới được xác định offline từ thời gian bắt đầu và kết thúc session; quy tắc này có sử dụng thông tin về phần còn lại của session để chọn mẫu.

Đây là bài toán gợi ý sau khi đã quan sát hành vi đầu session, không phải gợi ý trước khi user thực hiện bất kỳ thao tác nào. Nếu một giao dịch là hành vi đầu tiên, nó đã thuộc history; chỉ giao dịch sau đó mới là target.

## 2. Nhãn và cửa sổ quan sát

Target là tập item có transaction trong `(query_time, query_time + 7 ngày]`, cùng visitor và cùng split, loại các sự kiện thuộc session `boundary_drop`. Một item mua nhiều lần trong cửa sổ là một target kèm `purchase_event_count`.

Toàn bộ cửa sổ phải nằm trong split của query. Những query gần cuối mỗi split được lưu trong `excluded_queries` với lý do thiếu cửa sổ, không gán nhãn âm giả.

Train/validation/test dùng cùng quy tắc. Không chuyển nhãn của ngày cuối train sang validation. Giữ query không mua trong bảng chính; nếu chỉ tính ranking metric trên query có target, báo cáo rõ số lượng/tỷ lệ đã chọn.

Các query của cùng user có thể chia sẻ giao dịch tương lai do cửa sổ chồng nhau. Không xem chúng là các quan sát thống kê hoàn toàn độc lập; có thể tính metric user-macro hoặc bootstrap theo user.

## 3. Tách label khỏi feature

`query_features` chỉ có ID/thời gian và thống kê history. Các cột `target_count`, `has_purchase_target`, `eligible_target_count`, `label_end` nằm ở bảng queries và **không phải feature**.

`purchase_targets` là bảng nhãn: cả `first_purchase_time`, `purchase_event_count` và việc một item có mặt trong bảng đều chứa thông tin tương lai. Metadata trong bảng này dùng phân tích độ phủ. Khi chấm điểm ứng viên, phải lấy metadata theo thời điểm từ catalog/property history, không dùng danh sách item trong bảng nhãn làm đầu vào mô hình.

Features số đếm toàn thời gian dùng history `<= t`. Số đếm cửa sổ W dùng `(t-W, t]`. Recency thiếu nghĩa là chưa từng quan sát hành vi đó, không tự điền 0 (0 nghĩa là vừa thực hiện).

`interactions_train` là snapshot tại cuối train. Không ghép snapshot này làm feature cho các query sớm hơn trong train. Mô hình học lịch sử từng query phải truy xuất lịch sử đúng `t`.

## 4. Model parameters và history online

Fit mapping và tham số mô hình trên train. Dùng validation để chọn siêu tham số. Giao thức mặc định không refit bằng test và không thay tập train khi báo cáo test.

Được dùng sự kiện validation/test **đã xảy ra trước hoặc đúng thời điểm query** để cập nhật history online mà không fit lại tham số. Vì thế user chưa có embedding train vẫn có thể được gợi ý từ lịch sử session.

Chỉ số mã hóa `user_idx`/`item_idx` của ID có trong train bắt đầu từ 2; PAD=0 và UNK=1. ID raw không bị đổi. User/item ngoài vocabulary train có idx=1 nhưng vẫn giữ ID raw để phân biệt thực thể. Một embedding UNK không thể tự phân biệt mọi sản phẩm mới. Mô hình chỉ dùng ID cần quy định cách xử lý item mới: dùng fallback trên tập ứng viên chung, hoặc đánh giá riêng với chính sách chỉ lấy item có trong train. Khi so sánh trực tiếp, mọi mô hình trong cùng bảng kết quả phải dùng cùng chính sách candidate; không âm thầm loại target của item mới khỏi mẫu số.

## 5. Catalog và metadata theo thời gian

`known_at` = thời điểm đầu tiên item xuất hiện trong log sự kiện hoặc thuộc tính category/available. Đây là định nghĩa biết đến sản phẩm theo nguồn dữ liệu được sử dụng; không khẳng định là ngày lên sàn thật.

Với mỗi query, dùng property mới nhất có `property_time <= query_time`. Khoảng hiệu lực của bản ghi là `[property_time, valid_to)`; valid_to rỗng nghĩa là chưa quan sát bản cập nhật sau.

Candidate phải có `known_at <= t` và availability mới nhất khác 0. Availability thiếu được cho phép theo chính sách v1, đồng thời giữ cờ unknown. Không tự lấy metadata tương lai để lấp thiếu.

Item thuộc category ngoài cây hoặc chưa có category nhận `category_id=-1`; hai nguyên nhân được phân biệt bằng cờ. Root hợp lệ có parent rỗng không bị xem là lỗi.

Không tự đưa positive vào candidate. Giữ target không nằm trong candidate để báo cáo giới hạn độ phủ; chỉ báo cáo metric trên subset hợp lệ như một metric bổ sung có tên rõ ràng. Không mặc định loại item từng xem, từng thêm giỏ hoặc từng mua: giao thức cho phép mua lặp.

## 6. Mẫu âm train

`negative_samples_train` dành cho thí nghiệm pairwise trên query train có ít nhất một target mua. Mỗi query lấy tối đa 5 item thuộc vocabulary train, đã biết tại t, đủ điều kiện availability tại t, không nằm trong tập purchase target của cửa sổ đó.

Lấy mẫu đồng đều trong tập hợp được phép, không coi lượt xem/giỏ là nhãn âm của các nhiệm vụ view/cart. Không đọc nhãn validation/test để lọc mẫu âm train. Trường hợp thiếu ứng viên được lưu thống kê thay vì bổ sung item tương lai.

Các query train không mua vẫn có ở bảng queries để dùng với mục tiêu khác. Mẫu âm này không phục vụ đánh giá ranking; validation/test xếp hạng toàn bộ tập candidate theo cùng chính sách giữa các mô hình.

## 7. Đánh giá dự kiến

- Recall@10/20, NDCG@10/20 trên query có purchase target; báo cáo riêng mẫu số và tỷ lệ query có mua.
- Báo cáo query-macro và user-macro; chia warm/cold-user/cold-item.
- Báo cáo target coverage và `macro_recall_candidate_ceiling_pct`, là trung bình tỷ lệ target nằm trong candidate trên các query có nhãn mua. Đây là cận trên chỉ xét candidate, chưa xét giới hạn K; cận trên Recall@K còn phụ thuộc `min(K, eligible_target_count) / target_count` của từng query.
- Coverage theo cặp query–item không nhất thiết bằng giới hạn Recall trung bình theo query, vì số target khác nhau.
- Khi so sánh chỉ mua / nhiều hành vi, giữ cùng query, target và candidate. Khi một model không xử lý cold items, báo cáo giới hạn ấy thay vì âm thầm thay tập đánh giá.
- Không diễn giải điểm mô hình hoặc xác suất học từ negative sampling như xác suất mua đã được hiệu chuẩn.

## 8. Tái lập

Seed, cutoff rule, cửa sổ, trọng số tính `interaction_score` và số mẫu âm nằm trong config. Các trọng số này chưa phải siêu tham số đã được tối ưu bằng kết quả mô hình. Manifest ghi version Python/thư viện, config hash và SHA-256 từng input/output. `requirements-lock.txt` ghi các phiên bản thư viện của lần xử lý đã lưu. Để so hash bitwise cần cùng môi trường; khác phiên bản Parquet có thể khác hash dù dữ liệu logic giống nhau.

Pipeline tạo marker `data/multibehavior/INCOMPLETE` trong lúc chạy. Chỉ dùng dữ liệu khi marker đã biến mất và tất cả checks đã qua. Chạy lại ghi đè các artifact do pipeline này quản lý, không thay raw hoặc output notebook cũ. Không chạy đồng thời hai pipeline vào cùng thư mục.

`require_complete()` chỉ kiểm tra thư mục tồn tại và không có marker `INCOMPLETE`; hàm này không xác minh đầy đủ file hoặc checksum. Dùng `python scripts/verify_artifacts.py` để đối chiếu số dòng/checksum đầu ra, checksum raw, config và mã pipeline với manifest. `summary.json` lưu kết quả kiểm tra tại lần tạo dữ liệu, không tự cập nhật khi chỉ đọc file.

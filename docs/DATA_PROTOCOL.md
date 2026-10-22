# Giao thức dữ liệu v1: chuẩn bị log đa hành vi và nhãn mua 7 ngày

Project hiện dừng ở xử lý và kiểm tra dữ liệu. Tài liệu này mô tả các bảng đã chuẩn bị, không quy định phương pháp huấn luyện hoặc triển khai một hệ thống gợi ý.

## 1. Thời gian và đơn vị query

- Timestamp raw millisecond chuyển UTC; cutoff phân vị được làm tròn về millisecond.
- Session mới nếu cùng visitor nghỉ hơn 30 phút. Session qua ranh giới mang `boundary_drop` và không tham gia history/label.
- Một query cho mỗi session không qua ranh giới, sau khi quan sát **toàn bộ sự kiện có timestamp đầu tiên**. `query_id = session_id`. Chỉ giữ query đủ cửa sổ nhãn.
- History gồm sự kiện không thuộc boundary của cùng visitor có `event_time <= query_time`, kể cả session trước.
- `event_id` là chỉ số dòng raw để tái lập thứ tự lưu, không chứng minh thứ tự nhân quả giữa sự kiện đồng thời.

Thời điểm query không chọn theo việc visitor sẽ mua hay số sản phẩm cuối session. Việc loại session qua ranh giới sử dụng thời điểm bắt đầu/kết thúc session và là quy tắc chọn mẫu ngoại tuyến. Một giao dịch xảy ra ngay tại query_time đã thuộc history; chỉ giao dịch sau đó mới có thể là target.

## 2. Nhãn và cửa sổ quan sát

Target là item có transaction trong `(query_time, query_time + 7 ngày]`, cùng visitor và split, không thuộc session `boundary_drop`. Một item mua nhiều lần trong cửa sổ tạo một target kèm `purchase_event_count`.

Toàn bộ cửa sổ phải nằm trong split. Query thiếu cửa sổ lưu ở `excluded_queries`, không gán âm giả. Giữ cả query không mua nhưng đủ cửa sổ. Không chuyển nhãn vượt cuối train sang validation.

Query cùng visitor có thể chia sẻ giao dịch vì cửa sổ chồng nhau. Số cặp query–item không phải số giao dịch độc lập hoặc số đơn hàng. Tỷ lệ query sau lọc không nhất thiết bằng 80/10/10.

## 3. Tách nhãn khỏi đặc trưng

`query_features` chỉ có ID/thời gian và thống kê history. `target_count`, `has_purchase_target`, `eligible_target_count`, `label_end` ở bảng queries chứa thông tin phục vụ kiểm tra nhãn, không phải đặc trưng quan sát tại query.

`purchase_targets` chứa thông tin tương lai. Metadata kèm bảng này dùng phân tích độ phủ; không dùng danh sách target làm catalog ứng viên.

Số đếm toàn lịch sử dùng sự kiện `<= t`; cửa sổ W dùng `(t-W, t]`. Recency thiếu nghĩa là chưa quan sát hành vi, không tự điền 0. `interactions_train` là tổng hợp tại cuối train, không dùng nguyên bảng như đặc trưng cho query sớm hơn mốc đó.

## 4. Mapping và đối tượng ngoài train

Mapping chỉ lấy visitor/item có tương tác train. Chỉ số đã biết bắt đầu từ 2; PAD=0, UNK=1. ID raw luôn được giữ. Nhiều item mới cùng có idx=1 vẫn là các item khác nhau theo ID raw; không gộp chúng thành một sản phẩm.

`cold_user` mô tả visitor của query chưa có trong train; `cold_item` mô tả item target chưa có trong train. Visitor ngoài train vẫn có thể có lịch sử đã quan sát tới query. Những cờ này là thông tin phân nhóm dữ liệu, không phải kết quả dự đoán.

## 5. Catalog và metadata theo thời gian

`known_at` là thời điểm đầu tiên item xuất hiện trong log hoặc thuộc tính category/available, không khẳng định ngày lên sàn thật. Mỗi query chỉ tra thuộc tính mới nhất có `property_time <= query_time`.

Khoảng hiệu lực là `[property_time, valid_to)`; valid_to rỗng nghĩa là chưa quan sát bản cập nhật sau. Category thiếu hoặc ngoài cây nhận `category_id=-1`, có cờ phân biệt hai nguyên nhân. Nút gốc có parent rỗng là hợp lệ.

Candidate phải có `known_at <= t` và available gần nhất khác 0. Available thiếu vẫn được giữ theo chính sách v1, kèm cờ unknown; không lấp thiếu bằng tương lai. Không tự thêm target vào candidate. Giữ target ngoài candidate để phản ánh độ phủ thực của dữ liệu. Không mặc định loại item đã tương tác vì dữ liệu có mua lặp.

Snapshot cuối train chỉ mô tả trạng thái tại cutoff, không thay metadata tại mọi thời điểm validation/test. Availability là trạng thái quan sát gần nhất, không bảo đảm tồn kho trực tiếp.

## 6. Mẫu âm đã chuẩn bị

`negative_samples_train` là bảng dữ liệu tùy chọn cho query train có ít nhất một target mua. Mỗi query lấy tối đa 5 item thuộc vocabulary train, đã biết và đủ điều kiện availability tại t, không nằm trong purchase target của cửa sổ ấy.

Âm chỉ có nghĩa là không quan sát mua trong cửa sổ, không chứng minh người dùng không thích. Item từng xem/giỏ có thể là âm của nhãn mua; không suy ra nhãn âm cho hành vi xem/giỏ. Không dùng nhãn validation/test để lọc âm train. Nếu thiếu ứng viên thì lưu số thiếu, không bổ sung item tương lai.

Query train không mua vẫn có trong bảng chính. Tệp âm là một đầu ra chuẩn bị dữ liệu, không phải kết quả huấn luyện hoặc toàn bộ catalog.

## 7. Kiểm tra chất lượng dữ liệu

Kiểm tra trùng/khóa, số dòng, bảo toàn số hành vi, lịch sử không từ tương lai, nhãn đúng cửa sổ, mapping chỉ từ train, metadata đúng thời điểm và mẫu âm hợp lệ.

`candidate_target_pair_coverage_pct` là tỷ lệ cặp query–item target nằm trong candidate. `macro_recall_candidate_ceiling_pct` là trung bình tỷ lệ target hợp lệ theo query có nhãn mua; đây là giới hạn do dữ liệu ứng viên, chưa xét K và chưa phải chỉ số mô hình. Hai số không nhất thiết bằng nhau vì số target/query khác nhau.

Các kết quả kiểm tra đã lưu phải được đọc kèm ngày và phạm vi thực hiện. Kiểm tra lấy mẫu không đồng nghĩa với kiểm tra độc lập mọi feature trên mọi query.

## 8. Tái lập

Config lưu seed, ngưỡng session, phân vị, cửa sổ nhãn/feature, trọng số tổng hợp và số mẫu âm. Trọng số chỉ là quy ước xử lý dữ liệu. Manifest ghi phiên bản Python/thư viện, config hash và SHA-256 từng input/output; `requirements-lock.txt` lưu phiên bản thư viện của lần xử lý.

Pipeline tạo `data/multibehavior/INCOMPLETE` trong lúc chạy. Chỉ dùng bộ hoàn tất, không còn marker. Chạy lại ghi lại các artifact được quản lý; không thay raw hoặc output notebook cũ. Không chạy đồng thời hai pipeline vào cùng thư mục.

`require_complete()` chỉ kiểm tra thư mục và marker, không kiểm hết checksum. Dùng `python scripts/verify_artifacts.py` để đối chiếu số dòng/checksum đầu ra, raw, config và mã pipeline. `summary.json` không tự cập nhật khi chỉ đọc file.

Khác phiên bản thư viện có thể làm thay đổi checksum Parquet dù dữ liệu logic tương đương. Muốn so checksum bitwise cần cùng môi trường của lần tạo dữ liệu đã ghi nhận.

# Báo cáo kiểm tra dữ liệu đa hành vi

Lần xử lý được ghi nhận: **29/09/2026**. Kiểm tra lại: **01/10/2026**. Pipeline: `scripts/prepare_multibehavior.py`; dữ liệu: `data/multibehavior/`. Cấu hình và phiên bản môi trường nằm trong [manifest](manifest.json).

## Kết quả xử lý

- Giữ **2.755.641 sự kiện khác biệt**: 2.664.218 view, 68.966 addtocart, 22.457 transaction. Loại đúng 460 dòng trùng hoàn toàn từ raw. Số sự kiện transaction là số dòng mua sản phẩm, không phải số đơn hàng.
- Giữ **1.713.144 cặp user–item train**, kèm số lần từng hành vi, thời điểm cuối và recency tại cuối train.
- Tạo **1.519.027 query**, **88.317 cặp query–item có nhãn mua**, **91.540 mẫu âm train**. Mỗi query train có nhãn mua nhận đủ 5 mẫu âm trong lần xử lý này.
- Đọc 20.275.902 dòng thuộc tính; giữ 2.291.853 bản cập nhật category/availability, timestamp nguồn và khoảng hiệu lực. Các thuộc tính khác chưa được dùng.
- Catalog toàn kỳ có **466.868 item**. Khi dựng ứng viên phải lọc `known_at <= query_time` và availability tại thời điểm đó; không dùng toàn bộ catalog như thể tất cả item đã được biết từ đầu train.

| Split | Query đủ cửa sổ | Query có nhãn mua | Cặp query–item có nhãn mua | Độ phủ target theo cặp | Query có visitor ngoài train |
|---|---:|---:|---:|---:|---:|
| Train | 1.312.772 | 18.308 | 73.969 | 89,73% | 0,00% |
| Validation | 99.873 | 1.553 | 8.250 | 88,40% | 85,33% |
| Test | 106.382 | 1.465 | 6.098 | 88,91% | 90,05% |

Một query là một thời điểm gợi ý; có thể có nhiều sản phẩm được mua sau đó hoặc không có nhãn mua. Số cặp query–item không phải số giao dịch độc lập, vì các cửa sổ nhãn của cùng visitor có thể chồng nhau.

## Quy tắc của bộ dữ liệu hiện tại

1. Giữ riêng view/addtocart/transaction, kể cả các hành vi trên cùng sản phẩm; không yêu cầu một chuỗi xem → giỏ → mua đầy đủ.
2. Đặt thời điểm query sau toàn bộ nhóm sự kiện có timestamp đầu tiên của session. Loại session qua ranh giới split và query thiếu cửa sổ nhãn; không lọc theo việc visitor có mua hay theo số sản phẩm tối thiểu trong session.
3. History và feature chỉ dùng sự kiện đã xảy ra, loại `boundary_drop`. Nhãn là các item có transaction trong 7 ngày sau query, cùng visitor và cùng split, cũng loại `boundary_drop`. Bảng feature không chứa các cột nhãn.
4. Metadata và candidate được tra tại thời điểm query. Snapshot cuối train không đại diện cho trạng thái sản phẩm ở mọi thời điểm validation/test.
5. Mapping chỉ lấy ID có trong train. Chỉ số mã hóa dành PAD=0, UNK=1; chỉ số đã biết bắt đầu từ 2, còn ID raw được giữ nguyên. Category chưa biết và availability chưa biết được thể hiện bằng giá trị/cờ riêng.
6. Mẫu âm chỉ dùng cho query train có nhãn mua, không trùng target và phải thuộc tập ứng viên tại thời điểm query. Validation/test không dùng các mẫu âm này làm tập ứng viên đánh giá.

## Kiểm tra đã thực hiện

Lần xử lý ngày 29/09/2026 ghi nhận **29 kiểm tra đạt** trong [summary.json](summary.json), gồm các ràng buộc khóa, thời gian, số đếm và tính hợp lệ của mẫu âm. Kết quả này thuộc lần tạo dữ liệu, không tự cập nhật khi đọc lại báo cáo.

Lần kiểm tra lại ngày 01/10/2026 ghi nhận trong [recheck_2026-10-01.json](recheck_2026-10-01.json):

- **11/11 unit tests đạt** cho các trường hợp trùng dữ liệu, nhiều hành vi cùng item, cùng timestamp, cửa sổ nhãn, feature lịch sử, cây danh mục, metadata theo thời gian và mẫu âm.
- **20 file Parquet** khớp số dòng, checksum và schema đã ghi. Checksum của bốn file raw, config và mã pipeline cũng khớp manifest.
- **51/51 kiểm tra dữ liệu đạt**, gồm đối chiếu events sạch với raw, mốc chia thời gian, mapping, nhãn mua, số target từng query, các file split và điều kiện của mẫu âm.
- Hàm kiểm tra sẵn có tính lại số đếm lịch sử trên mẫu 100 query. Kiểm tra độc lập bổ sung kiểm tra history, số đếm cửa sổ, recency và tập target trên **240 query**, lấy 40 query cho mỗi nhóm split × có/không có nhãn mua. Việc dựng lại toàn bộ nhãn dùng hàm của pipeline; không coi đó là một triển khai độc lập.

Các kiểm tra trên hỗ trợ kết luận rằng dữ liệu nhất quán với giao thức hiện tại. Chưa có kết quả huấn luyện hoặc đánh giá mô hình để kết luận chất lượng gợi ý.

## Giới hạn cần tính đến khi sử dụng

- **20,96% item train** thiếu/không biết category; **20,84%** chưa biết availability. Pipeline giữ trạng thái unknown, không suy đoán giá trị còn thiếu từ tương lai.
- Test có **1.465 query có nhãn mua trên 106.382 query đủ cửa sổ**, khoảng **1,38%**. Recall/NDCG trên các query có nhãn mua phải ghi rõ mẫu số; tỷ lệ này không phải conversion rate của một hệ thống gợi ý đang vận hành.
- Target coverage test theo cặp là **88,91%**; cận trên Recall trung bình theo query chỉ xét candidate là **91,01%**. Hai số khác nhau do số target mỗi query khác nhau. Cận trên chưa xét giới hạn K; đây không phải điểm mô hình đạt được.
- **90,05% query test** thuộc visitor chưa có trong train; **8,99% cặp target test** là item chưa có tương tác train. Mô hình phải quy định cách xử lý user/item mới và báo cáo rõ chính sách candidate khi so sánh.
- Có **83.501 query train, 82.661 validation và 76.439 test** bị loại vì không đủ cửa sổ 7 ngày trong split. Chúng nằm trong `excluded_queries`, không được gán thành nhãn âm. Tỷ lệ query sau lọc không phải 80/10/10.
- Các query cùng visitor có thể chia sẻ giao dịch tương lai. Khi đánh giá mô hình, cần xét kết quả trung bình theo visitor và tránh coi mọi query là quan sát thống kê độc lập.
- Availability là trạng thái quan sát gần nhất, không bảo đảm tồn kho trực tiếp. Thiếu exposure logs để phân biệt sản phẩm chưa được nhìn thấy với sản phẩm đã bị bỏ qua.

Độ phủ/cold-start ở đây thuộc **bài toán nhãn mua 7 ngày**. Các tỷ lệ next-item trong [báo cáo pipeline cũ](../data_audit_review_vi.md) dùng đơn vị đánh giá khác; không dùng chênh lệch giữa hai báo cáo để khẳng định chất lượng mô hình tăng.

## Tái lập

Dùng các lệnh trong [README](../../README.md) để tạo và xác minh dữ liệu. `requirements-lock.txt` lưu phiên bản thư viện của lần xử lý đã ghi nhận; manifest lưu cả phiên bản Python. Raw và dữ liệu đầu ra lớn không được theo dõi bằng Git.

Xem [giao thức dữ liệu](../../docs/DATA_PROTOCOL.md) và [từ điển bảng](../../docs/DATA_DICTIONARY.md) trước khi dùng dữ liệu huấn luyện hoặc đánh giá.

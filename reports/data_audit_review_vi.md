# Đánh giá dữ liệu Retailrocket của project

> Đây là audit pipeline notebook ban đầu. Bộ dữ liệu đa hành vi mới đã được triển khai riêng; xem [báo cáo sau hoàn thiện](multibehavior/REVIEW_VI.md) và [giao thức mới](../docs/DATA_PROTOCOL.md). Không so trực tiếp metric next-item dưới đây với nhãn mua 7 ngày của bộ mới.

Ngày kiểm tra: 29/09/2026. Phạm vi: hai notebook trong `notebooks/`, CSV raw và các Parquet đang lưu. Số liệu chất lượng được tính lại trực tiếp từ dữ liệu; không chạy lại pipeline để ghi đè dữ liệu. Các file JSON cùng thư mục lưu số liệu kiểm tra.

## 1. Kết luận

Dữ liệu đã có nền tảng tốt để xây dựng baseline gợi ý từ tương tác ngầm và dự đoán sản phẩm tiếp theo trong session. Việc làm sạch, chia thời gian, giới hạn metadata tại train cutoff và mã hóa ID theo train là hợp lý. Tuy nhiên, chưa nên xem bộ dữ liệu là hoàn thiện cho mọi bài toán: dữ liệu rất thưa, cold-start cao, metadata thiếu đáng kể trong nhóm item train và tập ứng viên tại train không bao phủ đủ target tương lai.

`01_data_audit.ipynb` mới khảo sát events raw, thêm datetime trong bộ nhớ và xem dữ liệu trùng; notebook này chưa thực hiện xử lý và lưu dữ liệu hoàn chỉnh. Pipeline thực tế nằm trong `01_prepare_retailrocket.ipynb`.

## 2. Dữ liệu raw

| File | Ý nghĩa |
|---|---|
| `events.csv` | Log hành vi, gồm timestamp millisecond, visitorid, event, itemid, transactionid |
| `item_properties_part1.csv`, `item_properties_part2.csv` | Lịch sử thuộc tính sản phẩm: timestamp, itemid, property, value; nhiều dòng cho một sản phẩm theo thuộc tính và thời gian |
| `category_tree.csv` | Quan hệ categoryid–parentid |

Events có **2.756.101 dòng**, **1.407.580 visitor**, **235.061 item**; thời gian từ 03/05/2015 đến 18/09/2015 (UTC), gần 138 ngày. Visitor ID được dùng như user ID trong pipeline; không có thông tin định danh/hồ sơ người dùng trong các bảng được xử lý.

| Hành vi raw | Số dòng | Tỷ lệ xấp xỉ |
|---|---:|---:|
| view | 2.664.312 | 96,67% |
| addtocart | 69.332 | 2,52% |
| transaction | 22.457 | 0,81% |

Không thiếu timestamp, visitorid, event, itemid. Có 460 dòng trùng hoàn toàn. `transactionid` thiếu ở 2.733.644 dòng không phải transaction: đây là thiếu theo ngữ nghĩa, không nên xóa những dòng này hoặc điền mã giao dịch giả. 22.457 dòng transaction tương ứng 17.672 transaction ID khác nhau; số dòng mua sản phẩm không đồng nghĩa số đơn hàng.

Theo log đọc đủ chunk của notebook, hai file properties lần lượt có 10.999.999 và 9.275.903 dòng, tổng 20.275.902; kết quả đọc lại raw nằm trong `data_raw_metrics.json`. Category tree có 1.669 danh mục.

## 3. Những công đoạn đã thực hiện

### 3.1. Làm sạch events

Giữ ba hành vi hợp lệ; loại dòng thiếu trường bắt buộc; loại trùng hoàn toàn; chuyển timestamp sang `event_time` UTC; sắp theo visitor, thời gian, item. Dữ liệu còn **2.755.641 dòng**: 2.664.218 view, 68.966 addtocart và 22.457 transaction. Chênh lệch so với raw đúng 460 dòng trùng.

### 3.2. Tạo session và chia thời gian

Session mới bắt đầu khi đổi visitor hoặc khoảng cách đến hành vi trước lớn hơn 30 phút. Thêm `session_id`, `session_position`, `session_length`; tổng **1.761.675 session**.

Lấy quantile 80% và 90% trên thời gian sự kiện, không phải chia ngẫu nhiên hoặc chia 80% số ngày:

- Train cutoff: `2015-08-18 04:23:20.445 UTC`.
- Validation cutoff: `2015-09-02 17:49:14.032 UTC`.
- Session đi qua mốc chia được gắn `boundary_drop`: **47 session, 328 events**. Vẫn lưu trong events_clean_split, nhưng không đưa vào các bảng mô hình.

| Split | Events | Users | Items | Sessions |
|---|---:|---:|---:|---:|
| Train | 2.204.470 | 1.123.755 | 212.915 | 1.396.273 |
| Validation | 275.359 | 156.925 | 77.664 | 182.534 |
| Test | 275.484 | 158.004 | 77.496 | 182.821 |

Thêm `user_seen_in_train`, `item_seen_in_train`, `cold_start_type`; lưu `events_clean_split.parquet`.

### 3.3. Xử lý metadata

Đọc properties theo chunk một triệu dòng, **chỉ giữ `categoryid` và `available`**, bỏ thiếu và trùng trong chunk rồi bỏ trùng toàn cục. Lưu 21 chunk và bảng `item_properties_relevant.parquet` gồm **2.291.853 dòng**: 1.503.639 available và 788.214 categoryid. Các thuộc tính khác chưa được khai thác trong dữ liệu processed.

Chỉ lấy bản ghi thuộc tính có thời gian không vượt train cutoff; chọn giá trị mới nhất cho mỗi cặp item–property rồi pivot. Có **412.882 item có ít nhất một metadata trước cutoff**.

Làm sạch cây danh mục, tính category gốc, độ sâu và đường dẫn. Hợp item metadata với item train thành `products_train_snapshot.parquet`: **456.974 item**, gồm 212.915 item đã có tương tác train và 244.059 item chỉ có metadata. Gắn `candidate_at_train = available thiếu hoặc available = 1`; đây là cờ ứng viên, chưa xóa item khỏi bảng.

### 3.4. Tạo bảng tương tác user–item

Chỉ dùng events train, đếm số lần view/addtocart/transaction của từng cặp; thêm cờ có giỏ hàng, có mua và thời điểm tương tác cuối. Có **1.713.144 cặp user–item**.

Điểm tương tác:

`log(1 + view) + 3 × log(1 + addtocart) + 5 × log(1 + transaction)`.

Đây là điểm trọng số do pipeline lựa chọn, không phải rating hay xác suất mua đã được kiểm định. Tạo user_idx và item_idx liên tiếp từ 0, chỉ fit trên train; lưu bảng mapping và interactions encoded.

### 3.5. Tạo dữ liệu session và nhãn

Bỏ boundary session; gộp các hành vi liên tiếp có cùng item bằng cách giữ dòng đầu. Ví dụ `view A → cart A → view B` trở thành `view A → view B`. Còn **2.475.048 bước** trong `session_events.parquet`.

- Train: lấy item kế tiếp trong session làm nhãn, có **583.094 cặp next-item**.
- Validation/test: chỉ lấy session có ít nhất hai bước sau rút gọn, giữ item cuối làm target, các bước trước là history.
- Validation có **64.243 dòng history nhưng chỉ 25.549 session đánh giá**.
- Test có **66.083 dòng history nhưng chỉ 25.996 session đánh giá**.

Một session có nhiều dòng history cùng một target. Khi đánh giá theo thiết kế này cần dựng một history hoàn chỉnh và tính một kết quả cho mỗi session; tính mỗi dòng như một mẫu độc lập sẽ đặt trọng số lớn hơn lên session dài.

## 4. Đánh giá chất lượng hiện tại

### Những điểm đã đạt

- Không còn events trùng theo năm cột raw; các trường bắt buộc và thông tin split/session đều đầy đủ.
- Mỗi session thuộc đúng một split; các khoảng thời gian train, validation, test không chồng lấn trong dữ liệu đã lưu.
- Bảng tương tác không trùng user–item, không thiếu giá trị; công thức score tính lại khớp.
- Bảng encoded giữ nguyên các tương tác và không thiếu chỉ số.
- Properties chọn lọc không thiếu, không trùng; không có nhiều value khác nhau trên cùng khóa item–property–timestamp.
- Category tree có 25 root; không có parent khác rỗng tham chiếu ra ngoài cây.
- Snapshot giới hạn metadata ở train cutoff, tránh lấy trạng thái sau cutoff làm feature của mô hình tại thời điểm đó.

### Vấn đề 1: Dữ liệu rất thưa và cold-start cao

**79,70% user train chỉ tương tác một item**; **36,31% item train chỉ có một user**. Mật độ ma trận user–item chỉ **0,000716%**. Vì vậy, nhiều user không đủ lịch sử để học sở thích cá nhân đáng tin cậy.

| Chỉ số trên session đánh giá, không phải dòng history | Validation | Test |
|---|---:|---:|
| User chưa xuất hiện trong train | 82,09% | 87,14% |
| Target item chưa xuất hiện trong train | 5,02% | 8,49% |
| Target đã xuất hiện trong history của chính session | 16,51% | 15,82% |

ID mapping chỉ bao phủ train nên cần chính sách unknown item và fallback cho user mới. Không nên mặc định loại mọi item từng xem khỏi kết quả next-item: điều đó sẽ loại một phần target hợp lệ ở các session quay lại sản phẩm cũ.

### Vấn đề 2: Metadata thiếu nhiều trong nhóm item thực sự được học

Trong **212.915 item train**, có **44.619 item thiếu category (20,96%)**, **44.365 item thiếu available (20,84%)**. Trong toàn bộ bảng products, thiếu category 45.167 và available 44.697. Có **132 item có category_id nhưng không tìm thấy category đó trong cây**; các item này nằm ngoài nhóm đã có tương tác train.

Không nên diễn giải mọi `parentid` rỗng là lỗi: root hợp lệ không có parent. Cần phân biệt metadata chưa biết, category không tồn tại trong cây và category root.

### Vấn đề 3: Candidate snapshot hạn chế khả năng đạt đúng target

Có **99.880 item** được đánh dấu candidate; chỉ **85.138** trong số đó có item_idx train. Nếu áp dụng đúng tập candidate này cho toàn bộ evaluation:

| Độ phủ target theo session | Validation | Test |
|---|---:|---:|
| Target nằm trong candidate_at_train | 75,52% | 68,52% |
| Target vừa là candidate vừa có item_idx train | 74,31% | 67,18% |

Như vậy, ngay cả xếp hạng lý tưởng cũng không thể đạt HitRate@K/Recall@K vượt mức độ phủ tương ứng trên bài toán một target/session, nếu tập ứng viên bị giới hạn như trên. Đây là giới hạn có điều kiện khi sử dụng cờ ứng viên; pipeline hiện mới lưu cờ, chưa cho thấy mã mô hình thực sự áp dụng nó.

Cần xác định rõ mục tiêu: đánh giá catalog cố định tại train cutoff hay trạng thái sản phẩm tại thời điểm gợi ý. Với mục tiêu thứ hai, cần tra thuộc tính gần nhất **không muộn hơn thời điểm dự đoán**, đồng thời công bố độ phủ target. Không nên âm thầm bỏ target không hợp lệ chỉ để tăng metric.

### Vấn đề 4: Rút gọn session làm mất loại hành vi

Trong train, số dòng addtocart giảm **54.688 → 17.287**, transaction giảm **17.864 → 6.278** sau rút gọn. Điều này không làm mất dữ liệu mua trong bảng interactions; nó chỉ ảnh hưởng nhánh session.

Quy tắc hiện tại phù hợp nếu muốn dự đoán **sản phẩm khác kế tiếp** và chủ động bỏ thao tác lặp trên cùng item. Nếu muốn dự đoán hành vi mua, chuyển đổi hoặc dùng mức độ ý định trong chuỗi, nên giữ chuỗi hành vi gốc hoặc gộp kèm số lần/cờ hành vi mạnh nhất đã xảy ra tại thời điểm dự đoán. Không dùng hành vi tương lai làm feature cho một dự đoán sớm hơn.

### Vấn đề 5: Tập đánh giá chỉ phản ánh session nhiều bước

Sau rút gọn, **84,79% session train**, **86,00% validation**, **85,78% test** chỉ còn một bước. Evaluation hiện phản ánh khoảng 14% session validation/test có đủ lịch sử; kết quả không đại diện cho toàn bộ lượt truy cập hoặc tình huống chưa có lịch sử.

### Vấn đề 6: Cần giữ đúng ngữ nghĩa thời gian của feature

Snapshot cuối train phù hợp làm catalog/features cố định để dự đoán sau cutoff. Nếu dùng snapshot đó cho từng ví dụ next-item nằm sớm hơn trong train, thuộc tính có thể đến từ tương lai của ví dụ; cần join theo thời gian nếu yêu cầu mô phỏng dự đoán tại từng sự kiện.

`session_length` trong bảng session là số events của session gốc, không phải số bước sau rút gọn; nó cũng chứa thông tin về kết thúc session. Không đưa độ dài toàn session vào feature dự đoán trực tuyến. File next_item_train hiện không chứa cột này, nhưng session_events có.

## 5. Ưu tiên hoàn thiện

1. Chốt bài toán là next-item, gợi ý mua hay xếp hạng user–item; quy định đơn vị đánh giá là session và tập ứng viên.
2. Báo cáo đồng thời kết quả toàn bộ session đủ điều kiện, warm/cold-user/cold-item và tỷ lệ target nằm trong candidate. Xử lý unknown ID rõ ràng.
3. Bổ sung kiểm tra tự động độ phủ candidate/metadata, tính khớp nhãn next-item, thời gian snapshot, chất lượng cây và mapping; không chỉ dựa vào các assert cơ bản cuối notebook.
4. Nếu dùng metadata động cho từng thời điểm, lưu timestamp nguồn của mỗi thuộc tính và thực hiện join theo thời gian. Duy trì trạng thái unknown riêng.
5. Chỉ thay cách rút gọn session khi mục tiêu yêu cầu giữ hành vi mua/giỏ. Giữ bảng events gốc đã sạch làm nguồn kiểm chứng.
6. Đóng gói tham số session gap, cutoff, trọng số tương tác và phiên bản dữ liệu vào cấu hình/manifest để tái lập. Khi chạy lại properties, tránh đọc nhầm chunk cũ còn sót từ lần chạy khác.

Chưa có bằng chứng trong các artifact đã kiểm tra về lọc bot/outlier, tạo negative samples, ma trận sparse, đặc trưng nội dung đầy đủ hoặc kết quả mô hình. Đây không tự động là lỗi: mức độ cần thiết phụ thuộc thuật toán và mục tiêu. User có tối đa 7.757 events và 3.283 item train khác nhau là đối tượng cần khảo sát thêm, chưa đủ cơ sở kết luận là bot.

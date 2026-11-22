# Hệ thống gợi ý sản phẩm từ tương tác đa hành vi

Project nghiên cứu **gợi ý sản phẩm cá nhân hóa theo phiên truy cập** trên dữ liệu RetailRocket. Từ các tương tác đã quan sát gồm **xem (`view`), thêm giỏ (`addtocart`) và mua (`transaction`)**, hệ thống dự kiến trả về **Top-10 hoặc Top-20 sản phẩm cho bước tương tác kế tiếp**.

README được tổng hợp từ [báo cáo tổng quan](docs/Bao_cao_tong_quan_RetailRocket.docx) và [slide đề xuất đề tài](docs/BAO_CAO_DE_XUAT_DE_TAI_HUST.pptx), đối chiếu với các tệp và báo cáo đang có. Bốn tệp dữ liệu gốc nằm trong `data/raw/`; [hướng dẫn tiền xử lý](docs/HUONG_DAN_TIEN_XU_LY_DU_LIEU.md) giải thích các quy tắc. Hai notebook đã xuất dữ liệu sự kiện sạch, phiên, nhóm timestamp và các tập theo thời gian; lịch sử thuộc tính, bộ mẫu prefix–target, huấn luyện và demo là các bước tiếp theo.

Có thể thực hiện bước kiểm tra và làm sạch sự kiện bằng [Jupyter Notebook 01](notebooks/01_kiem_tra_va_lam_sach_events.ipynb). Mở file trong VS Code hoặc Jupyter, chọn kernel **RetailRocket (Python 3.14)** có `ipykernel`, `pandas`, `pyarrow`, rồi chạy các cell từ trên xuống bằng **Shift+Enter** hoặc **Restart Kernel and Run All**. Notebook có giải thích, bảng kết quả và bước lưu/đọc kiểm tra `data/processed/events_clean.parquet`.

Tiếp tục với [Jupyter Notebook 02 — tạo phiên, nhóm timestamp và chia tập](notebooks/02_tao_phien_nhom_timestamp_va_chia_tap.ipynb). Notebook đọc `events_clean.parquet`, mở phiên khi khoảng nghỉ >30 phút, nhóm theo session/timestamp và chia tập theo toàn phiên. Ba tập sử dụng được lưu tại `data/processed/splits/`; các phiên qua biên được lưu riêng để đối chiếu.

Sau 01 và 02, chạy [notebook kiểm tra độc lập kết quả](notebooks/kiem_tra_ket_qua_xu_ly.ipynb) để đối chiếu toàn bộ raw → clean → phiên/nhóm → từng tệp split. Notebook này chỉ đọc các tệp dữ liệu và ghi báo cáo vào `reports/data_validation/metrics.json`.

Hai notebook chạy độc lập với các file `.py` của project. Notebook 01 đối chiếu báo cáo làm sạch lần trước nếu cùng đầu vào; notebook 02 kiểm tra trực tiếp mã băm Parquet với báo cáo notebook 01 nếu có báo cáo. Nếu mã băm khác thì dừng để tránh dùng lẫn phiên bản dữ liệu. Các notebook không còn tham chiếu script/báo cáo raw audit cũ.

## Hiện trạng project

**Đã hoàn thành phần xử lý sự kiện đến chia tập theo thời gian.** Các tập hiện có chứa sự kiện theo phiên/nhóm; chưa phải bộ mẫu huấn luyện có prefix, target và metadata tại thời điểm query.

| Thành phần đang có | Vai trò | Trạng thái xác nhận |
| --- | --- | --- |
| `docs/` | Báo cáo Word, slide đề xuất và hướng dẫn tiền xử lý | Có tài liệu xác định bài toán và lộ trình |
| `data/raw/` | Events, hai phần thuộc tính và cây danh mục | Đủ bốn CSV |
| Notebook 01 | Kiểm tra schema events, bỏ trùng và chuyển UTC | 8 cell code đã chạy lại từ kernel mới, không có lỗi được lưu |
| `events_clean.parquet` | Sự kiện sạch và `event_id`, `event_time` | 2.755.641 dòng; SHA-256 khớp báo cáo làm sạch |
| Notebook 02 | Tạo phiên, nhóm timestamp và chia tập | 9 cell code có kết quả chạy, không có lỗi được lưu |
| Notebook kiểm tra kết quả | Đối chiếu nội dung với raw và kiểm tra hai chiều các quy tắc phiên/nhóm/chia tập | Tách riêng với notebook tạo dữ liệu; không ghi đè raw/Parquet |
| `events_sessionized.parquet`, `sessions.parquet`, `timestamp_groups.parquet` | Các bảng liên kết event → group → session | Có dữ liệu, schema và số dòng khớp báo cáo |
| `data/processed/splits/` | Train, validation, test và boundary_drop | Đã xuất riêng; SHA-256 khớp báo cáo |
| `split_config.json` và hai báo cáo `metrics.json` | Quy tắc chia tập, thống kê và dấu vết dữ liệu | Đầu ra notebook 01 khớp đầu vào notebook 02 |
| Metadata theo thời điểm, ánh xạ ID, bộ query/target | Đầu vào cuối cùng cho mô hình | Chưa triển khai/xuất |
| Popularity, ItemKNN, Session-KNN, GRU4Rec; bộ đánh giá và demo | Thực nghiệm và ứng dụng | Đang ở mức kế hoạch, chưa có mã/mô hình/kết quả |

Kết quả kiểm tra hiện tại xác nhận raw `events.csv`, tám tệp Parquet đã xuất và cấu hình chia tập khớp mã băm lưu trong các báo cáo. Báo cáo làm sạch ghi nhận đã đọc lại Parquet để so sánh toàn bộ bảng; báo cáo phiên/chia tập ghi nhận kiểm tra tương tự cho từng đầu ra.

**Môi trường notebook:** đã đăng ký kernel **RetailRocket (Python 3.14)** với Python 3.14.5 tại `C:/Users/haimi/AppData/Local/Programs/Python/Python314/python.exe`, có đủ `ipykernel`, `pandas`, `pyarrow`. Cả hai notebook chỉ định kernel này và in `sys.executable` ở cell đầu. Trong VS Code, dùng **Select Kernel** để chọn tên trên hoặc chọn đúng Python qua **Python Environments**. Python 3.10.20 hiện thiếu các thư viện này; cần chọn lại kernel nếu IDE đang giữ lựa chọn cũ.

**Kết quả rà soát hai notebook:** đã chạy lại tuần tự 01 → 02 từ hai kernel mới, tổng 17 cell code thành công. Bổ sung kiểm tra giới hạn int64, ghi lý do lỗi khi dữ liệu thiếu, đối chiếu nguồn giữa hai notebook và ví dụ biên cho ngưỡng phiên/mốc chia. Mã băm của cả tám Parquet giữ nguyên sau lần chỉnh sửa; cấu hình/báo cáo được cập nhật với thông tin kiểm tra và môi trường mới.

## 1. Dữ liệu sử dụng

**RetailRocket** là bộ log thương mại điện tử công bố trên Kaggle, có dấu thời gian từ **03/05/2015 đến 18/09/2015 (UTC)**. Log gồm **1.407.580 visitor** và **235.061 item** với mã định danh ẩn danh.

| Nhóm dữ liệu | Tệp gốc | Số dòng | Trường thông tin chính |
| --- | --- | ---: | --- |
| Tương tác | `events.csv` | 2.756.101 | `timestamp`, `visitorid`, `event`, `itemid`, `transactionid` |
| Thuộc tính sản phẩm | `item_properties_part1.csv`, `item_properties_part2.csv` | 20.275.902 tổng hai tệp | `timestamp`, `itemid`, `property`, `value`; dùng chủ yếu `categoryid`, `available` |
| Cây danh mục | `category_tree.csv` | 1.669 | `categoryid`, `parentid` |

Sau khi loại **460 dòng trùng tuyệt đối**, báo cáo ghi nhận **2.755.641 sự kiện**:

| Hành vi | Số sự kiện | Tỷ trọng |
| --- | ---: | ---: |
| Xem | 2.664.218 | 96,68% |
| Thêm giỏ | 68.966 | 2,50% |
| Mua | 22.457 | 0,81% |

Các tỷ trọng được làm tròn và phản ánh phân bố sự kiện, không phải tỷ lệ chuyển đổi.

**Đặc điểm ảnh hưởng đến thiết kế:**

- Dữ liệu rất thưa; **71,16% visitor chỉ có một sự kiện**, nên cần tận dụng ngữ cảnh phiên và có phương án dự phòng.
- Lượt xem áp đảo; thêm giỏ và mua ít hơn nhưng thể hiện ý định mạnh hơn. Cần kiểm chứng lợi ích của việc phân biệt hành vi.
- Không có tên, ảnh hoặc mô tả sản phẩm dùng trực tiếp cho gợi ý; thông tin có ý nghĩa chủ yếu là danh mục và trạng thái khả dụng.
- Thuộc tính thay đổi theo thời gian và có thể thiếu. Chỉ dùng thông tin đã biết tại thời điểm gợi ý.
- Không có log hiển thị, do đó không thể coi sản phẩm chưa được tương tác là nhãn âm thật.

## 2. Định hướng và phạm vi

Mục tiêu là xây dựng pipeline dữ liệu có thể tái lập, so sánh các phương pháp trên cùng bộ đánh giá ngoại tuyến và tạo demo cập nhật gợi ý theo tương tác mới.

Ba câu hỏi nghiên cứu chính:

1. Phân biệt xem, thêm giỏ và mua có cải thiện chất lượng so với coi mọi tương tác như nhau không?
2. Thứ tự tương tác trong phiên có cung cấp thêm thông tin ngoài độ phổ biến và tương đồng sản phẩm không?
3. Phương pháp nào cân bằng tốt giữa chất lượng xếp hạng, thời gian chạy và khả năng phục vụ khách ít lịch sử?

| Phương pháp dự kiến | Cách tiếp cận | Vai trò |
| --- | --- | --- |
| Popularity | Xếp hạng theo số tương tác trong train; so sánh trọng số đều và trọng số theo hành vi | Mốc đối chứng và nguồn gợi ý dự phòng |
| ItemKNN đa hành vi | Tính cosine giữa các item từ ma trận phiên × item; gộp điểm từ các item trong prefix | Kiểm chứng tương đồng sản phẩm và trọng số hành vi |
| Session-KNN | Tìm các phiên train giống prefix hiện tại rồi tổng hợp điểm item | Đối chứng theo ngữ cảnh phiên |
| GRU4Rec | Đọc chuỗi nhóm timestamp, tổng hợp embedding item và hành vi trong từng nhóm | Mô hình chính theo định hướng slide đề xuất |

Giữ nguyên bốn phương pháp trên trong lộ trình hiện tại. Hàm mất mát của GRU4Rec cần được chốt khi triển khai: Word đề xuất cross-entropy cho target nhiều item, còn slide giới thiệu BPR/TOP1.

Xem [thuyết minh chi tiết các phương pháp](docs/PHUONG_PHAP_DE_XUAT_CHI_TIET.md) hoặc [bản Word](docs/PHUONG_PHAP_DE_XUAT_CHI_TIET.docx) để đọc phần đối chiếu PPTX–báo cáo, công thức, ví dụ tính điểm và thiết kế áp dụng cho từng mô hình.

Phạm vi chính là **gợi ý bước tương tác kế tiếp theo phiên**. Bài toán dự đoán item được mua trong 7 ngày sau query đầu phiên là hướng phụ, cần báo cáo riêng. Đề tài chưa bao gồm A/B test, đo tác động doanh thu hoặc triển khai hệ thống sản xuất.

## 3. Đầu vào và đầu ra

### Đầu vào tại thời điểm gợi ý

| Thành phần | Nội dung |
| --- | --- |
| Phiên hiện tại | Visitor/phiên và prefix gồm các sự kiện đã quan sát: item, loại hành vi, timestamp |
| Thời điểm `t` | Thời điểm tạo gợi ý; đọc đầy đủ nhóm sự kiện đồng thời tại `t` |
| Số lượng `K` | 10 hoặc 20 |
| Thông tin sản phẩm | Danh mục, trạng thái khả dụng và thời điểm item được biết đến, chỉ sử dụng dữ liệu có thời gian ≤ `t` |

### Đầu ra dự kiến

Danh sách Top-K **item không trùng**, xếp giảm dần theo điểm, gồm:

| Trường | Ý nghĩa |
| --- | --- |
| `rank` | Thứ hạng trong danh sách |
| `item_id` | Mã sản phẩm ẩn danh |
| `score` | Điểm xếp hạng tương đối, không mặc định là xác suất mua |
| `category_id` | Danh mục nếu đã biết tại `t` |
| `is_fallback` | Item có được bổ sung từ Popularity hay không |

Mục tiêu là trả đủ K khi còn đủ ứng viên hợp lệ có thể chấm điểm hoặc bổ sung bằng Popularity. Item đã xuất hiện trong prefix vẫn có thể được gợi ý lại vì khách có thể xem lại, thêm giỏ hoặc mua sản phẩm đó.

### Nhãn dùng khi huấn luyện và đánh giá

Mỗi phiên là chuỗi nhóm timestamp `g₁, g₂, …, gₙ`. Một mẫu gồm **prefix `P = (g₁, …, gₜ)`** và **target `G` là tập item phân biệt trong nhóm `gₜ₊₁`**, tính trên mọi loại hành vi. Target chỉ dùng khi huấn luyện/đánh giá, không phải đầu vào khi sinh gợi ý.

Ví dụ: phiên có `g₁ = xem A`, `g₂ = xem B`, `g₃ = thêm giỏ B và xem C` cùng timestamp. Tại cuối `g₂`, đầu vào là hai nhóm đầu, còn target là **`{B, C}`**. Không tự tạo thứ tự giữa các sự kiện đồng thời trong `g₃`.

## 4. Pipeline thực hiện

Xem [pipeline hoàn thành project](docs/PIPELINE_HOAN_THANH_PROJECT.md) để theo dõi phần đã có, công việc từ notebook 03–09, đầu ra, điều kiện chuyển bước và checklist bàn giao. Hướng dẫn này giữ nguyên bốn mô hình đã đề xuất.

```text
4 tệp CSV RetailRocket
        ↓
Kiểm tra schema → làm sạch → chuẩn hóa thời gian UTC
        ↓
Tạo phiên → nhóm sự kiện cùng timestamp → chia tập theo thời gian
        ↓
Tra thuộc tính tại t → mã hóa ID → tạo cặp prefix–target
        ↓
Lưu dữ liệu trung gian và bộ mẫu dưới dạng Parquet
        ↓
Huấn luyện Popularity / ItemKNN / Session-KNN / GRU4Rec
        ↓
Chọn cấu hình trên validation → đánh giá test với cấu hình đã đóng băng
        ↓
Lưu mô hình + cấu hình → demo gợi ý Top-K theo phiên
```

| Bước | Xử lý chính | Kết quả dự kiến |
| --- | --- | --- |
| 1. Khảo sát và làm sạch | Kiểm tra trường bắt buộc, kiểu dữ liệu và ba loại hành vi; cho phép thiếu `transactionid` với xem/thêm giỏ, yêu cầu có với mua; bỏ trùng tuyệt đối, giữ lần tương tác lại ở thời điểm khác; đổi timestamp mili-giây sang UTC | Log sự kiện đã chuẩn hóa |
| 2. Tạo phiên và nhóm thời gian | Với cùng visitor, mở phiên mới khi khoảng cách tương tác **hơn 30 phút**; gom sự kiện cùng timestamp trong phiên thành một nhóm | Session ID và chuỗi nhóm timestamp |
| 3. Chia tập | Dùng mốc phân vị **80% và 90% của thời gian sự kiện** để tạo train → validation → test; loại toàn bộ phiên cắt qua mốc chia khỏi lịch sử và nhãn | Ba tập có thứ tự thời gian, giữ nguyên phiên |
| 4. Chuẩn bị thuộc tính và ID | Tra bản cập nhật `categoryid`/`available` gần nhất có thời gian ≤ `t`; thiếu thì giữ `unknown`; xây từ vựng ID từ train | Metadata theo thời điểm và ánh xạ ID |
| 5. Tạo mẫu | Dùng các nhóm đã quan sát làm prefix, nhóm ngay sau làm target; phiên một nhóm không tạo mẫu nhưng vẫn giữ trong dữ liệu nền | Bộ mẫu prefix–target cho từng tập |
| 6. Huấn luyện và chọn cấu hình | Chỉ học độ phổ biến, tương đồng, kho phiên và tham số từ train; chọn trọng số hành vi, siêu tham số và dừng sớm trên validation | Mô hình và cấu hình đã chọn |
| 7. Sinh gợi ý | Lọc ứng viên tại `t`, chấm điểm, xếp hạng, bỏ item trùng và bổ sung bằng Popularity nếu thiếu | Top-10/20 và cờ dự phòng |
| 8. Đánh giá và demo | Đánh giá test sau khi đóng băng cấu hình; lưu kết quả, mô hình và phiên bản để demo cập nhật theo prefix | Báo cáo so sánh và demo dự kiến |

### Quy tắc chống rò rỉ và xử lý ứng viên

- Prefix không chứa sự kiện của target; không dùng thuộc tính hoặc tương tác xảy ra sau `t` để tạo đầu vào.
- Session-KNN chỉ tìm hàng xóm trong các phiên train. Không cập nhật mô hình hoặc độ phổ biến bằng test trong đánh giá cố định.
- Ứng viên phải đã được biết đến tại `t` (`known_at ≤ t`) và có trạng thái `available` gần nhất **khác 0**; trạng thái chưa biết vẫn được giữ.
- Item ngoài từ vựng train không được mô hình chấm điểm; nếu xuất hiện trong target vẫn tính vào mẫu số đánh giá và phải báo cáo tỷ lệ riêng.
- Popularity dự phòng cũng phải tuân thủ cùng bộ lọc ứng viên tại `t`.

## 5. Đánh giá và kết quả cần bàn giao

Chỉ số chính dự kiến là **NDCG@20**. Báo cáo thêm **Recall@K, Hit Rate@K và MRR@K**, với `K = 10, 20`; tính từng query rồi lấy trung bình trên cùng tập query có target không rỗng. Với target nhiều item, Recall đo tỷ lệ item đúng tìm được, Hit Rate đo có ít nhất một hit, còn MRR xét vị trí hit đầu tiên. NDCG dùng gain nhị phân và chiết khấu log cơ số 2.

Ngoài chất lượng xếp hạng, cần theo dõi thời gian huấn luyện/truy vấn, tỷ lệ dự phòng và ảnh hưởng của item mới hoặc phiên ít lịch sử. So sánh trọng số hành vi đều với trọng số phân biệt, cùng các đối chứng theo độ phổ biến, tương đồng và chuỗi để trả lời câu hỏi nghiên cứu.

Các sản phẩm bàn giao dự kiến:

- Dữ liệu sạch, phiên, metadata theo thời điểm, ánh xạ ID và bộ mẫu train/validation/test dạng Parquet.
- Pipeline tái lập được cùng cấu hình chia tập và quy tắc lọc ứng viên.
- Các mô hình đối chứng, GRU4Rec và cấu hình/phiên bản mô hình được chọn.
- Báo cáo đánh giá ngoại tuyến trên cùng bộ mẫu và demo Top-K theo phiên.

Kết quả đã có nằm trong [báo cáo làm sạch sự kiện](reports/events_clean/metrics.json), [báo cáo phiên/chia tập](reports/session_split/metrics.json) và [cấu hình chia tập](data/processed/split_config.json). Chạy lại bằng notebook 01 rồi notebook 02 trong đúng kernel. Chưa có bộ query/target, mã huấn luyện, checkpoint, điểm Recall/NDCG/MRR hoặc demo Top-K.

## 6. Kết quả khảo sát và dữ liệu đã xuất

Số liệu events và phiên/chia tập có báo cáo, đầu ra và mã băm đang được lưu. Các kết quả thuộc tính/cây danh mục bên dưới là kết quả khảo sát trước đó được ghi lại trong tài liệu; báo cáo raw audit gốc hiện không còn, và lần rà soát trạng thái này chưa quét lại toàn bộ hơn 20 triệu bản ghi thuộc tính. Bước metadata cần tái lập các kiểm tra đó và lưu báo cáo mới.

| Hạng mục | Kết quả kiểm tra | Việc cần làm |
| --- | --- | --- |
| Schema sự kiện | Đủ năm cột; không thiếu bốn trường bắt buộc; không có nhãn hành vi/ID/thời gian không hợp lệ theo các kiểm tra đã chạy | Giữ kiểm tra này trong pipeline |
| Mã giao dịch thiếu | 2.733.644 dòng raw thiếu `transactionid`, đều thuộc view/addtocart; không có lượt mua thiếu mã | Không xóa các dòng này bằng `dropna()` toàn bảng |
| Sự kiện trùng | 460 dòng: 94 view và 366 addtocart; còn 2.755.641 sự kiện | Chỉ bỏ trùng năm cột gốc, giữ tương tác lặp khác thời điểm |
| Thuộc tính | 20.275.902 dòng; không thiếu giá trị hoặc trùng tuyệt đối xuyên hai tệp; category/available không có xung đột cùng item–thuộc tính–timestamp | Đọc theo khối; giữ 2.291.853 cập nhật category/available cho pipeline chính |
| Khoảng thời gian | Events: 03/05–18/09/2015; thuộc tính: 10/05–13/09/2015, UTC | 137.193 sự kiện raw xảy ra trước bản thuộc tính đầu tiên; metadata tại các thời điểm đó phải là unknown |
| Item không có metadata | 49.815/235.061 item trong log không có bất kỳ bản thuộc tính nào | Giữ item, biểu diễn metadata thiếu; không điền từ tương lai |
| Cây danh mục | 1.669 danh mục, 25 gốc thiếu parent hợp lệ; không trùng ID, chu trình hoặc cha tham chiếu ngoài cây | Giữ các danh mục gốc, không điền parent bằng 0 |
| Danh mục ngoài cây | 30 mã trong thuộc tính không có trong cây, thuộc 137 item; không item nào trong số đó xuất hiện trong log sự kiện | Giữ mã danh mục gốc, đánh dấu chưa biết cha/đường dẫn; không tự dựng quan hệ |
| Thứ tự dữ liệu | Events và cả hai phần thuộc tính đều không được sắp xếp toàn cục theo thời gian | Sắp xếp trước khi tạo phiên hoặc as-of join; không coi part1/part2 là thứ tự thời gian |

Notebook 02 đã tạo và lưu **1.761.675 phiên**, **2.753.015 nhóm timestamp**, đồng thời tách **47 phiên / 328 sự kiện** qua biên. Các tệp `splits/train.parquet`, `splits/validation.parquet`, `splits/test.parquet` lần lượt có **2.204.470**, **275.359**, **275.484 sự kiện**, được kiểm tra không có phiên nằm ở hai tập và train trước validation trước test. Số cặp prefix–target dự kiến trước lọc ứng viên là **806.025 train, 92.566 validation, 92.468 test**; đây mới là số đếm từ nhóm, chưa tạo query/target thực tế. Các số đo khớp báo cáo audit của cùng đầu vào.

## 7. Trạng thái các bước xử lý

Schema là hợp đồng về tên cột, kiểu dữ liệu, trường được phép thiếu và giá trị hợp lệ. Kiểm tra schema trước khi làm sạch để tránh xóa nhầm lượt xem thiếu mã giao dịch hoặc tạo phiên từ timestamp sai đơn vị.

| Bước | Trạng thái | Kết quả đã có / việc còn lại |
| --- | --- | --- |
| A. Ghi nhận raw | Một phần | Đủ bốn CSV; dấu vết events còn trong báo cáo. Cần lưu lại schema, số dòng và SHA-256 hai phần thuộc tính/cây ở bước metadata |
| B. Kiểm tra schema | Events đã xong | Events có kiểm tra tái lập bằng notebook 01; thuộc tính/cây đã khảo sát trước nhưng cần đưa vào notebook xử lý riêng |
| C. Làm sạch events | Đã xong | Bỏ 460 dòng trùng tuyệt đối, giữ tương tác lặp khác thời điểm và ba loại hành vi |
| D. Thời gian UTC | Đã xong cho events | Có timestamp gốc và `event_time` UTC trong Parquet; thời gian metadata sẽ chuẩn hóa khi xử lý thuộc tính |
| E. Chuẩn bị metadata | Chưa triển khai đầu ra | Xử lý chung hai phần thuộc tính, giữ lịch sử category/available, kiểm tra cây và lưu `known_at` |
| F. Phiên và nhóm | Đã xong | 1.761.675 phiên, 2.753.015 nhóm; các sự kiện đồng thời có khóa nhóm chung |
| G. Chia thời gian | Đã xong | Có cấu hình và ba tập hợp lệ; 47 phiên / 328 sự kiện qua biên lưu riêng |
| H. Metadata tại t và ID | Chưa triển khai | As-of join ≤t, từ vựng item chỉ từ train, PAD/UNK và chính sách ứng viên |
| I. Query và target | Chưa triển khai | Tạo mẫu từ nhóm hiện tại → nhóm kế tiếp; số 806.025/92.566/92.468 mới là số cặp dự kiến |
| J. Đánh giá và mô hình | Chưa triển khai | Bộ đánh giá thống nhất, Popularity → ItemKNN → Session-KNN → GRU4Rec |

Chi tiết thao tác, lý do, ví dụ và đầu ra của từng bước nằm trong [hướng dẫn tiền xử lý](docs/HUONG_DAN_TIEN_XU_LY_DU_LIEU.md). Notebook 01 đã hoàn thành kiểm tra/làm sạch/thời gian cho events; notebook 02 đã hoàn thành bước F–G và đọc kiểm tra lại các Parquet. Tiếp theo là lịch sử thuộc tính ở bước E, metadata tại thời điểm query và mã hóa ID ở H, rồi tạo bộ mẫu prefix–target ở I. Các bảng tổng hợp vẫn có nhãn `boundary_drop` để truy vết; dùng ba tệp split hợp lệ hoặc lọc split khi xây lịch sử và nhãn.

## 8. Thứ tự triển khai tiếp theo

Các notebook sau đây là **kế hoạch, chưa có file triển khai**. Giữ nguyên bốn mô hình trong đề xuất. Chi tiết quy tắc ứng viên, loss khởi đầu, kiểm chứng và điều kiện hoàn thành nằm trong [pipeline tổng thể](docs/PIPELINE_HOAN_THANH_PROJECT.md).

| Ưu tiên | Công việc dự kiến | Đầu ra/điều kiện cần đạt |
| --- | --- | --- |
| 0 | Chọn đúng kernel để tiếp tục làm việc | Notebook chạy trong môi trường có ipykernel/pandas/pyarrow; không nhầm lỗi môi trường với lỗi dữ liệu |
| 1 — Notebook 03 | Chuẩn bị lịch sử thuộc tính và catalog | `item_property_history.parquet`, `category_tree.parquet`, `item_catalog.parquet` và báo cáo; kiểm tra schema, trùng/xung đột xuyên tệp, danh mục ngoài cây; giữ category/available theo thời gian và `known_at` |
| 2 — Notebook 04 | Metadata theo thời điểm, ánh xạ ID và bộ mẫu | Query có session, thời điểm, vị trí kết thúc prefix và tập target nhóm kế tiếp; mapping chỉ fit train, có PAD/UNK; thuộc tính ≤t và quy tắc ứng viên cố định |
| 3 — Notebook 05 | Bộ đánh giá và Popularity | Recall/Hit Rate/NDCG/MRR tại K=10,20; kiểm tra cách tính bằng ví dụ biết trước; đánh giá trên validation để thiết lập mốc chất lượng và dự phòng |
| 4 — Notebook 06 | ItemKNN đa hành vi và Session-KNN | So sánh trên cùng query/ứng viên; thử trọng số hành vi; theo dõi thời gian truy vấn và tỷ lệ dự phòng |
| 5 — Notebook 07 | GRU4Rec | Chuỗi nhóm timestamp, biểu diễn item/hành vi; chốt loss phù hợp target nhiều item; chọn cấu hình bằng NDCG@20 trên validation |
| 6 — Notebook 08 | Thực nghiệm cuối và báo cáo | Đóng băng dữ liệu/config/seed trước đợt test cuối; so sánh bốn mô hình, kiểm chứng hành vi/thứ tự, cập nhật Word/slide theo kết quả thật |
| 7 — Notebook 09 | Demo và bàn giao | Demo trong Jupyter tải artifact đã lưu, nhận prefix và trả Top-10/20; có hướng dẫn chạy từ kernel mới và ví dụ khớp đánh giá |

Trước huấn luyện, lưu giao thức thành cấu hình có phiên bản: known_at từ nguồn sự kiện hợp lệ/thuộc tính; metadata ≤t; ứng viên xếp hạng là catalog hợp lệ tại t giao với vocabulary train; thiếu K thì vị trí thiếu không hit. Target giữ ID gốc đầy đủ, kể cả ngoài train hoặc không đủ điều kiện ứng viên, và báo cáo tỷ lệ riêng. Với GRU, pipeline đề xuất mean embedding trong nhóm và cross-entropy nhãn đều trên các item đúng làm cấu hình khởi đầu; mọi thay đổi phải được thử trên validation. Không dùng boundary_drop làm lịch sử hoặc nhãn.

Project hoàn thành khi có metadata và bộ mẫu được kiểm tra, bốn mô hình với bộ đánh giá thống nhất, thực nghiệm/test theo cấu hình đã đóng băng, báo cáo/slide dựa trên kết quả thật và demo tái lập được. GRU4Rec không cần được ép vượt baseline; cần đo và giải thích kết quả một cách công bằng. Công việc triển khai kế tiếp là **notebook 03 — lịch sử thuộc tính và catalog**.

## 9. Lưu trữ và xác nhận dữ liệu xử lý đúng

Hiện tiếp tục dùng **CSV raw + Parquet đã xử lý + JSON cấu hình/báo cáo** cho pipeline ngoại tuyến trong Jupyter. Chưa cần nhập sang MySQL/PostgreSQL để huấn luyện hoặc kiểm tra dữ liệu. Nếu sau này có nhu cầu truy vấn SQL, DuckDB có thể đọc trực tiếp Parquet; không bắt buộc sao chép dữ liệu sang một cơ sở dữ liệu khác. Xem [tài liệu Parquet/PyArrow](https://arrow.apache.org/docs/25.0/python/parquet.html) và [truy vấn Parquet bằng DuckDB](https://duckdb.org/docs/guides/import/parquet_import).

Kiểm tra chất lượng dữ liệu cần cả nội dung và quy tắc, không chỉ số dòng hoặc mã băm:

| Lớp kiểm tra | Bằng chứng cần có |
| --- | --- |
| Phiên bản | Tệp hiện tại khớp SHA-256 trong báo cáo, đầu ra 01 khớp đầu vào 02 |
| Raw → clean | Mọi cột/dòng theo event_id khớp raw sau đúng quy tắc bỏ trùng và đổi UTC |
| Phiên | Cùng visitor và nghỉ ≤30 phút phải cùng phiên; đổi visitor hoặc nghỉ >30 phút phải khác phiên |
| Nhóm timestamp | Một session/timestamp chỉ có một group; group không lẫn phiên hoặc thời gian, item/hành vi giữ đủ |
| Chia tập | Đúng mốc thời gian, nguyên phiên, phiên qua biên chỉ ở boundary_drop |
| Tệp xuất | Toàn bộ nội dung từng partition khớp phần tương ứng trong bảng tổng; tổng ba tập và boundary bằng events sạch |

Notebook kiểm tra đã chạy thành công **36 phép kiểm tra** trên dữ liệu hiện tại, gồm đối chiếu toàn bộ 2.755.641 sự kiện sạch với raw và kiểm tra từng tệp split. Kết quả chi tiết nằm trong [báo cáo kiểm tra độc lập](reports/data_validation/metrics.json); mọi tệp đầu vào giữ nguyên trong lúc kiểm tra.

Chạy lại notebook kiểm tra sau khi thay đổi raw, thuật toán hoặc đầu ra. Các bước metadata, prefix–target và đánh giá mô hình chưa được triển khai nên cần thêm kiểm tra riêng ở giai đoạn sau. Các ví dụ biên trong notebook 01/02 và việc xem thủ công vài phiên bổ sung cho kiểm tra tự động; báo cáo PASS chỉ áp dụng cho phạm vi và phiên bản dữ liệu đã ghi.

# Hiện trạng project và ghi chú dữ liệu chi tiết

Tài liệu này bổ sung cho [README](../README.md): đặc điểm dữ liệu, phạm vi, giao thức và kết quả khảo sát dữ liệu. Danh sách notebook và cách chạy nằm ở README; kết quả từng bước nằm ở [`reports/`](../reports/README.md).

Nguồn: các report trong `reports/`. Slide báo cáo tiến độ: [BAO_CAO_TIEN_DO_PROJECT3_HUST_27_11.pptx](BAO_CAO_TIEN_DO_PROJECT3_HUST_27_11.pptx).

## 1. Dữ liệu sử dụng

**RetailRocket** là bộ log thương mại điện tử công bố trên Kaggle, có dấu thời gian từ **03/05/2015 đến 18/09/2015 (UTC)**. Log gồm **1.407.580 visitor** và **235.061 item** với mã định danh ẩn danh.

| Nhóm dữ liệu | Tệp gốc | Số dòng | Trường thông tin chính |
| --- | --- | ---: | --- |
| Tương tác | `events.csv` | 2.756.101 | `timestamp`, `visitorid`, `event`, `itemid`, `transactionid` |
| Thuộc tính sản phẩm | `item_properties_part1.csv`, `item_properties_part2.csv` | 10.999.999 + 9.275.903 = 20.275.902 | `timestamp`, `itemid`, `property`, `value`; dùng chủ yếu `categoryid`, `available` |
| Cây danh mục | `category_tree.csv` | 1.669 | `categoryid`, `parentid` |

Sau khi loại **460 dòng trùng tuyệt đối**, còn **2.755.641 sự kiện**:

| Hành vi | Số sự kiện | Tỷ trọng |
| --- | ---: | ---: |
| Xem | 2.664.218 | 96,68% |
| Thêm giỏ | 68.966 | 2,50% |
| Mua | 22.457 | 0,81% |

Đặc điểm ảnh hưởng đến thiết kế:

- Dữ liệu rất thưa; **71,16% visitor chỉ có một sự kiện**, nên cần tận dụng ngữ cảnh phiên và có phương án dự phòng.
- Lượt xem áp đảo; thêm giỏ và mua ít hơn nhưng thể hiện ý định mạnh hơn. Cần kiểm chứng lợi ích của việc phân biệt hành vi.
- Không có tên, ảnh hoặc mô tả sản phẩm; thông tin có ý nghĩa chủ yếu là danh mục và trạng thái khả dụng.
- Thuộc tính thay đổi theo thời gian và có thể thiếu. Chỉ dùng thông tin đã biết tại thời điểm gợi ý.
- Không có log hiển thị, do đó không thể coi sản phẩm chưa được tương tác là nhãn âm thật.

## 2. Định hướng và phạm vi

Ba câu hỏi nghiên cứu:

1. Phân biệt xem, thêm giỏ và mua có cải thiện chất lượng so với coi mọi tương tác như nhau không?
2. Thứ tự tương tác trong phiên có cung cấp thêm thông tin ngoài độ phổ biến và tương đồng sản phẩm không?
3. Phương pháp nào cân bằng tốt giữa chất lượng xếp hạng, thời gian chạy và khả năng phục vụ khách ít lịch sử?

| Phương pháp | Cách tiếp cận | Vai trò |
| --- | --- | --- |
| Popularity | Xếp hạng theo số tương tác trong train; trọng số đều và trọng số theo hành vi | Mốc đối chứng và nguồn gợi ý dự phòng |
| Repeat-last | Gợi ý lại item trong prefix theo độ gần đây tuyệt đối (`distance` nhỏ nhất, bỏ qua tần suất cũ) | Đo mức điểm của gợi ý lặp (quyết định D1) |
| ItemKNN đa hành vi | Cosine giữa item từ ma trận phiên × item; gộp điểm từ các item trong prefix | Tương đồng sản phẩm và trọng số hành vi |
| Session-KNN | Tìm các phiên train giống prefix rồi tổng hợp điểm item | Đối chứng theo ngữ cảnh phiên |
| GRU4Rec gốc | Chuỗi nhóm timestamp, chỉ mean embedding item mỗi nhóm; siêu tham số bản chính thức (bpr-max, hidden 224, batch 80) | Mô hình chính, đối chứng đơn hành vi |
| GRU4Rec đa hành vi | Như GRU4Rec gốc, cộng thêm mean embedding hành vi (xem / thêm giỏ / mua) của nhóm | Mô hình chính, đo lợi ích của thông tin hành vi |

Phạm vi hiện tại (protocol v1.3.2, 26/11/2026) là **gợi ý bước tương tác kế tiếp theo phiên**, với hai ngưỡng phiên 30 và 60 phút. Trọng tâm là so sánh GRU4Rec gốc với GRU4Rec đa hành vi; Popularity, Repeat-last và VSTAN là mốc tham chiếu. SR-GNN từng là neural đối chứng nhưng đã được gỡ khỏi phạm vi ở v1.3.1; S-POP, SR và V-SKNN được gỡ ở v1.3.2 (người dùng quyết định ngày 26/11/2026). Mọi chỉ số tính tại K = 20. Đề tài chưa bao gồm A/B test, đo tác động doanh thu hoặc triển khai hệ thống sản xuất.

## 3. Giao thức (tóm tắt — bản chính thức: `configs/protocol.json`)

- Phiên mới khi đổi visitor hoặc nghỉ **> 30 phút**; đúng 30 phút vẫn cùng phiên. Notebook 09 dựng thêm bộ dữ liệu với ngưỡng **60 phút** theo cùng luật và cùng mốc chia để so sánh.
- Nhóm = `(session_id, timestamp)`; không tạo thứ tự giả giữa sự kiện đồng thời.
- Mốc chia = phân vị 0,8 / 0,9 của timestamp events sạch; chia nguyên phiên; phiên cắt qua mốc → `boundary_drop`, không dùng làm lịch sử, ứng viên hay nhãn.
- Mẫu: prefix `P = (g₁, …, gₜ)`, target `G` = tập item phân biệt trong `gₜ₊₁`, mọi loại hành vi. Ví dụ `g₁ = xem A`, `g₂ = xem B`, `g₃ = thêm giỏ B + xem C` cùng timestamp → tại cuối `g₂` target là `{B, C}`.
- Ứng viên tại `t`: có trong vocabulary train và `known_at_ms ≤ t`. Áp dụng cả cho fallback Popularity. Luật `available` gần nhất `≠ 0` chỉ dùng cho báo cáo phụ (protocol v1.2.0+: 17,62% target validation có `available = 0` tại `t`).
- Target giữ ID gốc, kể cả ngoài vocabulary; vẫn tính vào mẫu số (0 điểm).
- Chỉ học từ train; chọn cấu hình trên validation; test chạy một lần với cấu hình đã đóng băng.

## 4. Kết quả khảo sát dữ liệu

| Hạng mục | Kết quả kiểm tra | Xử lý |
| --- | --- | --- |
| Schema sự kiện | Đủ năm cột; không thiếu bốn trường bắt buộc; không có nhãn/ID/thời gian không hợp lệ | Giữ kiểm tra trong notebook 01 |
| Mã giao dịch thiếu | 2.733.644 dòng raw thiếu `transactionid`, đều là view/addtocart | Không `dropna()` toàn bảng |
| Sự kiện trùng | 460 dòng: 94 view và 366 addtocart | Chỉ bỏ trùng năm cột gốc |
| Thuộc tính | 20.275.902 dòng; 2.291.853 cập nhật category/available | Đọc theo khối; notebook 03 tái lập và lưu report |
| Khoảng thời gian | Events 03/05–18/09/2015; thuộc tính 10/05–13/09/2015, UTC | 137.193 sự kiện xảy ra trước bản thuộc tính đầu tiên → metadata unknown |
| Item không có metadata | 49.815/235.061 item trong log | Giữ item, metadata thiếu; không điền từ tương lai |
| Cây danh mục | 1.669 danh mục, 25 gốc; không chu trình | Giữ gốc, không điền parent bằng 0 |
| Danh mục ngoài cây | 30 mã thuộc 137 item, không item nào có trong events | Đánh dấu chưa biết cha |
| Thứ tự dữ liệu | Events và hai phần thuộc tính không sắp toàn cục theo thời gian | Sắp trước khi tạo phiên hoặc as-of join |

Notebook 02 tạo **1.761.675 phiên**, **2.753.015 nhóm timestamp**, tách **47 phiên / 328 sự kiện** qua biên. Ba tập train / validation / test có **2.204.470 / 275.359 / 275.484 sự kiện** và **806.025 / 92.566 / 92.468 query**.

Đặc điểm tín hiệu học (`reports/02_gru_readiness/`): 41,01% lượt target train đã có trong prefix; 125 phiên train có hơn 100 nhóm; tỷ lệ target ngoài vocabulary train là 5,31% (validation) và 9,33% (test).

## 5. Lưu trữ và xác nhận dữ liệu

Dùng **CSV raw + Parquet đã xử lý + JSON cấu hình/báo cáo**. Chưa cần cơ sở dữ liệu; nếu cần truy vấn SQL, DuckDB đọc trực tiếp Parquet.

| Lớp kiểm tra | Bằng chứng |
| --- | --- |
| Phiên bản | Tệp khớp `sha256` (byte) và `content_sha256` (nội dung) trong report; đầu ra 01 khớp đầu vào 02 |
| Raw → clean | Mọi cột/dòng theo `event_id` khớp raw sau đúng quy tắc bỏ trùng và đổi UTC |
| Phiên | Cùng visitor và nghỉ ≤ 30 phút phải cùng phiên; đổi visitor hoặc nghỉ > 30 phút phải khác phiên |
| Nhóm timestamp | Một session/timestamp chỉ có một group; group không lẫn phiên hoặc thời gian |
| Chia tập | Đúng mốc, nguyên phiên, phiên qua biên chỉ ở `boundary_drop` |
| Tệp xuất | Nội dung từng partition khớp bảng tổng; tổng bốn tệp bằng events sạch |

Chạy lại notebook kiểm tra sau khi thay đổi raw, thuật toán hoặc đầu ra.

## 6. Kết quả thực nghiệm (27/11/2026)

Chi tiết: [`reports/10_final/BAO_CAO_KET_QUA.md`](../reports/10_final/BAO_CAO_KET_QUA.md) (test), `reports/09_validation_30_60/` (validation), `reports/12_data_ablation/`, `reports/13_gru4rec_official/`. Mọi report có `status: PASS`.

| Mô hình | NDCG@20 validation (30 phút, train) | NDCG@20 test (30 phút, refit) | NDCG@20 test (60 phút, refit) |
| --- | --- | --- | --- |
| Popularity | 0,0047 | 0,0041 | 0,0041 |
| Repeat-last | 0,3578 | 0,3337 | 0,3387 |
| VSTAN | 0,3747 | 0,3528 | 0,3559 |
| GRU4Rec gốc (chỉ item) | 0,3951 | 0,3713 | 0,3742 |
| GRU4Rec đa hành vi | 0,3950 | 0,3721 | 0,3752 |

- Thứ hạng GRU4Rec > VSTAN > Repeat-last > Popularity giữ nguyên ở validation và test, cả hai ngưỡng phiên.
- Hiệu ứng hành vi (đa hành vi − gốc, paired bootstrap): validation chứa 0; test bản refit khoảng +0,001 (CI 95% không chứa 0), bản `train_only` 60 phút lại âm (−0,0009). Kết luận: lợi ích rất nhỏ và chưa ổn định.
- Ngưỡng 60 phút nhỉnh hơn 30 phút khoảng +0,001 NDCG@20 trên cùng bộ query test.
- Notebook 13: GRU4Rec bản chính thức (10 epoch) đạt 0,3888 trên validation, thấp hơn bản tự cài 0,0063 (CI 95% [−0,0075; −0,0051]); bản chính thức tốt hơn ở target mới, kém ở target lặp.
- Lần chạy test v1.3.2 đầu tiên bị ngắt (26/11, không phải lỗi code); người dùng duyệt chạy lại cùng cấu hình đóng băng ngày 27/11. Lần chạy lại trùng khớp các số đã in trước khi ngắt. `configs/protocol.json` không ghi sự kiện này để giữ nguyên mã băm mà notebook 10–13 đã kiểm tra.

# GRU4REC GỐC: PIPELINE, HÀM LOSS VÀ HỎI ĐÁP

**Phạm vi:** giải thích GRU4Rec trong bài “Session-based Recommendations with Recurrent Neural Networks”, Hidasi và cộng sự, ICLR 2016. Tài liệu tập trung vào cấu hình đọc item hiện tại, cách huấn luyện theo phiên và hai loss BPR/TOP1.

## 1. Bài toán và đầu vào–đầu ra

**Mục tiêu:** từ các sản phẩm đã tương tác trong phiên hiện tại, xếp hạng sản phẩm cho tương tác tiếp theo.

A, B, C là ký hiệu của ba sản phẩm, ví dụ điện thoại, ốp lưng và kính cường lực. Với phiên A → B → C:

| Phần đã quan sát | Item đưa vào ở bước này | Đáp án dùng khi train/đánh giá |
| --- | --- | --- |
| A | A; trạng thái đầu h0 = 0 | B |
| A → B | B; mang trạng thái sau A sang | C |

Mạng đọc lần lượt từng item; trạng thái truyền giữa các bước mang thông tin lịch sử. Không phải đưa lại toàn bộ A → B ở mọi bước. Khi dự đoán C, chưa được đọc C. Một phiên n sự kiện tạo n−1 vị trí học; phiên chỉ một sự kiện không có đáp án kế tiếp trong phiên.

**Phân biệt ba thứ:** đầu vào là item đã quan sát; output của mạng là vector điểm; target là item thực tế xuất hiện tiếp theo. Sắp điểm giảm dần để có danh sách Top-K, không phải dự đoán xác suất mua hàng.

## 2. Pipeline qua các khối

```text
Item hiện tại → One-hot → [Embedding tùy chọn]
  → GRU, kết hợp với trạng thái bước trước
  → [Feedforward bổ sung, tùy chọn] → Điểm item
```

Từ điểm item, luồng tách theo mục đích:

```text
Huấn luyện: điểm item đúng + điểm mẫu đối chiếu
  → BPR hoặc TOP1 → gradient → cập nhật trọng số

Gợi ý: điểm các item cần xếp hạng
  → sắp giảm dần → danh sách Top-K
```

Gọi N là số item, H là chiều trạng thái và d là chiều embedding nếu dùng.

| Khối | Nhận gì? | Thực hiện gì? | Trả ra gì? |
| --- | --- | --- | --- |
| Input / one-hot | ID item hiện tại | Mã hóa vị trí item; ID không mang ý nghĩa lớn–nhỏ | Vector N chiều, chỉ vị trí item bằng 1 |
| Embedding, tùy chọn | ID hoặc one-hot | Tra vector trong bảng tham số được học | Vector d chiều; bỏ lớp này thì GRU nhận one-hot trực tiếp |
| GRU | Vector đầu vào và trạng thái cũ | Cập nhật ngữ cảnh bằng các cổng điều tiết thông tin | Trạng thái mới h có H số |
| Feedforward bổ sung, tùy chọn | Trạng thái của lớp GRU cuối | Biến đổi thêm biểu diễn | Vector đưa tới lớp chấm điểm |
| Output | Trạng thái/biểu diễn cuối | Ánh xạ thành điểm cho item | Một điểm cho mỗi item được chấm |

Trạng thái mới có hai vai trò: đưa tới lớp chấm điểm và chuyển sang bước thời gian tiếp theo. Nếu xếp nhiều lớp GRU, trạng thái lớp dưới là đầu vào lớp trên; đó là chiều sâu của mạng, không phải các sản phẩm A, B, C.

Bài gốc thử thêm embedding, feedforward và nhiều lớp, nhưng các thử nghiệm đó không cho thấy lợi ích so với cấu hình đơn giản. Một lớp GRU và đầu vào one-hot là lựa chọn tốt trong thí nghiệm của bài; không suy ra luôn tối ưu cho mọi dữ liệu. Bài cũng thử đầu vào tổng hợp lịch sử, nhưng phần giải thích này dùng cấu hình item hiện tại. [Bài gốc, mục 3.1 và 4.2](https://arxiv.org/html/1511.06939v4).

### Trạng thái và các cổng GRU

`h_t = GRU(x_t, h_(t−1))`

“Hồi tiếp” là sử dụng trạng thái bước trước khi xử lý bước mới. Cùng một bộ tham số được dùng qua các bước; h là vector tóm tắt ngữ cảnh, không phải bản sao nguyên vẹn của lịch sử.

- **Reset gate:** điều chỉnh thông tin cũ được dùng để tạo nội dung mới.
- **Update gate:** điều chỉnh cách trộn trạng thái cũ với nội dung mới.
- **Đổi phiên:** pipeline đặt h về 0; việc này khác reset gate bên trong GRU.

Ví dụ A → B và C → B cùng kết thúc ở B nhưng có trạng thái trước khác nhau, nên điểm gợi ý có thể khác. Đây là cá nhân hóa theo ngữ cảnh phiên.

### Từ trạng thái đến điểm

Lớp đầu ra dùng các trọng số được học để chuyển trạng thái thành điểm; có thể kèm hàm kích hoạt. Bài gốc ghi nhận tanh ở đầu ra có ích trong thí nghiệm. Ví dụ minh họa, sau A → B, điểm {A,B,C,D} là {0,1; 0,3; 0,8; 0,5}, thì Top-2 là C, D.

Khi train, bài gốc lấy mẫu đầu ra để giảm tính toán; không bắt buộc chấm toàn bộ N item cho mỗi bước. Khi đánh giá, xếp hạng trên tập item theo giao thức đã xác định. [Bài gốc, mục 3.1.2, 4.1 và 4.2](https://arxiv.org/html/1511.06939v4).

## 3. Huấn luyện theo phiên diễn ra thế nào?

### Một bước học

1. Đọc item hiện tại và trạng thái cũ; cập nhật trạng thái mới.
2. Chấm điểm item đúng tiếp theo và các item lấy mẫu để đối chiếu.
3. Tính một loss đã chọn: BPR hoặc TOP1.
4. Tính gradient, dùng optimizer cập nhật trọng số mạng và embedding nếu có.
5. Chuyển sang bước tiếp theo trong phiên; đổi phiên thì reset trạng thái.

**Giữ trạng thái và cập nhật trọng số là hai việc khác nhau.** Trạng thái biểu diễn phiên đang chạy; trọng số là những tham số dùng chung đã học từ nhiều phiên. Khi gợi ý, vẫn cập nhật trạng thái nhưng giữ nguyên trọng số, không cần target hay optimizer.

### Session-parallel mini-batch: nhiều phiên song song

Mỗi vị trí trong batch theo một phiên riêng. Giả sử S1 = A → B → C, S2 = D → E và S3 = F → G → H; batch có hai vị trí:

| Bước | Vị trí 1 | Vị trí 2 |
| --- | --- | --- |
| 1 | Đọc A, target B; h = 0 | Đọc D, target E; h = 0 |
| 2 | Đọc B, target C; giữ trạng thái S1 | S2 hết; thay bằng S3, reset rồi đọc F, target G |
| 3 | S1 hết; có thể nhận phiên mới | Đọc G, target H; giữ trạng thái S3 |

Không truyền trạng thái S2 sang S3 và không reset S1 chỉ vì S2 kết thúc. Thứ tự xử lý các phiên có thể thay đổi, nhưng thứ tự bên trong mỗi phiên phải giữ. [Bài gốc, mục 3.1.1](https://arxiv.org/html/1511.06939v4).

### Mẫu đối chiếu lấy từ đâu?

Bài gốc tận dụng item target ở các vị trí khác trong cùng batch. Ở bước 1 trên, B là target của vị trí 1, E là target của vị trí 2. Khi học từ A, dùng B làm item đúng và E làm mẫu đối chiếu; khi học từ D thì ngược lại.

Cách này giảm số điểm cần tính và tận dụng tính toán theo batch. Item phổ biến có xu hướng được lấy làm mẫu nhiều hơn. “Mẫu âm” không chứng minh người dùng đã thấy và không thích sản phẩm; đó là vai trò của item trong phép so sánh tại bước học này. [Bài gốc, mục 3.1.2](https://arxiv.org/html/1511.06939v4).

## 4. Hai hàm loss gốc: BPR và TOP1

**Vai trò chung:** cung cấp tín hiệu để cập nhật tham số theo mục tiêu xếp item đúng cao hơn các item đối chiếu. Loss dùng khi huấn luyện; Top-K dùng khi tạo danh sách gợi ý.

Ký hiệu: i là item đúng tiếp theo; j là mẫu đối chiếu; s_i và s_j là điểm mạng; M là số mẫu. σ là sigmoid, trả giá trị trong khoảng 0–1. Ký hiệu s ở đây tương ứng r̂ trên slide.

### BPR — Bayesian Personalized Ranking

`L_BPR = −(1/M) × Σ[j] log σ(s_i − s_j)`

**Đọc công thức theo ba bước:** lấy chênh lệch điểm đúng trừ điểm đối chiếu → đưa qua sigmoid và −log → lấy trung bình trên các mẫu.

- s_i cao hơn s_j: chênh lệch dương, loss thấp hơn.
- s_i thấp hơn s_j: chênh lệch âm, loss lớn hơn, tạo tín hiệu sửa thứ tự.
- Mục tiêu là quan hệ “item đúng nên đứng trên mẫu đối chiếu”, không yêu cầu dự đoán một mức điểm tuyệt đối.

**Ví dụ một cặp:** s_i = 0,8, s_j = 0,2 → loss ≈ 0,437. Đảo hai điểm → loss ≈ 1,037. Đây là số minh họa, không phải kết quả mô hình.

BPR quan tâm hiệu điểm: cộng cùng một hằng số vào hai điểm không đổi loss của cặp đó. BPR là mục tiêu xếp hạng có sẵn được đưa vào GRU4Rec, không phải loss do bài GRU4Rec phát minh.

### TOP1 — xấp xỉ thứ hạng và điều chuẩn điểm

`L_TOP1 = (1/M) × Σ[j] [σ(s_j − s_i) + σ(s_j²)]`

Hai thành phần có nhiệm vụ khác nhau:

| Thành phần | Khi nào tăng? | Vai trò |
| --- | --- | --- |
| σ(s_j − s_i) | Điểm đối chiếu cao hơn điểm đúng | Xấp xỉ mềm việc j đứng trên i, để giảm thứ hạng tương đối của item đúng |
| σ(s_j²) | Độ lớn của điểm đối chiếu tăng về phía dương hoặc âm | Khuyến khích điểm đối chiếu về gần 0, hạn chế cùng đẩy các điểm lên cao |

**Ví dụ:** s_i = 0,8, s_j = 0,2 → hai thành phần ≈ 0,354 và 0,510, tổng ≈ 0,864. Đảo hai điểm → tổng ≈ 1,300. Phần điều chuẩn thấp nhất bằng 0,5 khi s_j = 0; không cần kỳ vọng TOP1 về 0.

TOP1 được tác giả đề xuất cho bài toán này. Tên TOP1 không giới hạn hệ thống chỉ trả một sản phẩm và cũng không có nghĩa loss bằng trực tiếp chỉ số Hit@1. [Công thức và giải thích hai loss: bài gốc, mục 3.1.3](https://arxiv.org/html/1511.06939v4).

### Vì sao tác giả chọn hai loss này?

Đích của hệ gợi ý là xếp item đúng lên cao. BPR học chênh lệch điểm theo cặp; TOP1 bổ sung điều chuẩn điểm đối chiếu. Cả hai phù hợp cách huấn luyện dùng mẫu đầu ra và cho kết quả tốt trong các thí nghiệm của bài.

Bài năm 2016 cũng thử cross-entropy và ghi nhận vấn đề ổn định ở nhiều cấu hình được thử. Đây là kết quả trong điều kiện thí nghiệm đó, không phải khẳng định CE luôn không phù hợp. [Bài gốc, mục 4.2–4.3](https://arxiv.org/html/1511.06939v4).

**Chọn BPR hoặc TOP1 cho từng cấu hình; không mặc định cộng hai loss.** Chọn cấu hình bằng chất lượng gợi ý trên validation, không so số loss thô giữa hai hàm. Bài gốc báo cáo Recall@20 và MRR@20; loss và chỉ số đánh giá là hai khái niệm khác nhau.

## 5. Hỏi đáp nhanh

**A, B, C là gì?** Là các sản phẩm minh họa; mũi tên biểu diễn thứ tự tương tác, không phải các lớp mạng.

**Embedding chứa giá, màu hay thương hiệu không?** Không mặc định. Nó chứa các tham số số được học qua loss. Ví dụ vector 4 chiều chỉ có nghĩa mỗi item được biểu diễn bằng 4 số; không tự gán tên thuộc tính cho từng chiều.

**Không có embedding thì item có được học không?** Có. Ma trận trọng số nhận one-hot vẫn được học; nhân one-hot tương đương lấy cột trọng số ứng với item.

**Có học một mô hình riêng cho từng người không?** Không. Trọng số dùng chung, nhưng trạng thái thay đổi theo phiên. Đây là cá nhân hóa theo phiên, chưa phải ghi nhớ hồ sơ dài hạn của từng người.

**GRU có nhớ toàn bộ lịch sử chính xác không?** Không. Trạng thái chỉ là biểu diễn tóm tắt có kích thước hữu hạn.

**BPR và TOP1 có phải hai khối chạy nối tiếp không?** Không. Chúng là hai lựa chọn hàm mất mát đặt sau bước tính điểm khi train.

**Loss giảm có bảo đảm Top-K tốt hơn không?** Không bảo đảm. Cần đo trên dữ liệu validation chưa dùng để cập nhật trọng số; không suy ra chất lượng từ loss train đơn thuần.

**Khi đang gợi ý, có biết item kế tiếp để tính loss không?** Không. Mạng dùng phần đã quan sát để chấm điểm. Target chỉ có khi học hoặc đánh giá bằng dữ liệu lịch sử.

## 6. Đoạn trình bày ngắn

“GRU4Rec đọc lần lượt các sản phẩm trong phiên. Mỗi bước, GRU kết hợp đầu vào với trạng thái trước để cập nhật ngữ cảnh, rồi lớp đầu ra chấm điểm sản phẩm tiếp theo. Khi huấn luyện, BPR hoặc TOP1 hướng mô hình xếp item đúng cao hơn các mẫu đối chiếu; TOP1 thêm điều chuẩn điểm. Nhiều phiên được xử lý song song và trạng thái được đặt lại khi đổi phiên. Khi gợi ý, mô hình giữ nguyên trọng số, cập nhật trạng thái từ tương tác đã quan sát và trả các item có điểm cao nhất.”

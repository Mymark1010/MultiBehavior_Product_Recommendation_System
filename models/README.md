# models/ — mô hình đã huấn luyện

Chỉ `config.json` được đưa lên git. Trọng số (`model.pt`) lớn nên bị bỏ qua; chạy lại notebook tương ứng để tạo lại.

| Thư mục | Mô hình | Phiên | Do notebook |
| --- | --- | --- | --- |
| `gru4rec/` | GRU4Rec đa hành vi (embedding sản phẩm + embedding hành vi) | 30 phút | 07 |
| `gru4rec_item/` | GRU4Rec gốc (chỉ embedding sản phẩm) | 30 phút | 09 |
| `session60/gru4rec/` | GRU4Rec đa hành vi | 60 phút | 09 |
| `session60/gru4rec_item/` | GRU4Rec gốc | 60 phút | 09 |

Mọi mô hình trong bảng chỉ fit trên train. Biến thể fit lại trên train + validation được huấn luyện ngay trong notebook 10 và không lưu ra đây.

Popularity, Repeat-last và VSTAN không có trọng số cần lưu: chúng được tính lại từ dữ liệu train mỗi lần chạy. SR-GNN đã được gỡ khỏi phạm vi ở protocol v1.3.1 nên không còn thư mục `srgnn/`.

# CLAUDE.md — Quy ước cho Claude Code trong repo này

Project: gợi ý Top-K item cho bước tương tác kế tiếp theo phiên trên RetailRocket (view / addtocart / transaction).
Danh sách việc cần làm nằm trong `TASKS_HOAN_THIEN.md`. Đọc file đó trước khi sửa bất cứ thứ gì.

## Quy tắc bất biến (không được vi phạm)

1. **Không sửa hoặc ghi vào `data/raw/`.** Raw chỉ được đọc.
2. **Không đổi giao thức đã chốt** khi chưa có xác nhận của người dùng:
   - phiên mới khi đổi visitor hoặc nghỉ `> 1_800_000 ms` (30 phút);
   - nhóm = `(session_id, timestamp)`, không tạo thứ tự giả giữa các sự kiện cùng timestamp;
   - mốc chia = phân vị 0.8 / 0.9 của timestamp sự kiện sạch, chia nguyên phiên, phiên cắt qua mốc → `boundary_drop`;
   - `boundary_drop` không dùng làm lịch sử, ứng viên hay nhãn.
3. **Chống rò rỉ:** chỉ học popularity, độ tương đồng, kho phiên, vocabulary và trọng số từ `train`. Chọn siêu tham số trên `validation`. `test` chỉ chạy một lần với cấu hình đã đóng băng.
4. **Metadata tại thời điểm `t`:** chỉ dùng bản cập nhật thuộc tính có `timestamp <= t`. Thiếu thì để `unknown`, không điền từ tương lai.
5. **Target giữ đầy đủ ID gốc**, kể cả item ngoài vocabulary hoặc không đủ điều kiện ứng viên. Báo cáo riêng tỷ lệ này, không loại khỏi mẫu số.
6. Các mục đánh dấu **[CẦN XÁC NHẬN]** trong `TASKS_HOAN_THIEN.md` là quyết định phương pháp. Hỏi người dùng trước khi triển khai, không tự chọn.

## Quy ước kỹ thuật

- Python 3.14, pandas 3.0.x, pyarrow 25.0.x (xem `requirements.txt`). Mọi thời gian dùng UTC, timestamp là mili-giây kiểu `int64`.
- Tìm thư mục gốc project bằng cách dò thư mục cha từ `Path.cwd()` (giống notebook 01/02). Không ghi đường dẫn tuyệt đối vào code, output hay report.
- Ghi JSON bằng `write_text(..., encoding="utf-8", newline="\n")` để mã băm giống nhau trên Windows/macOS/Linux.
- Ghi Parquet theo mẫu `save_verified()` của notebook 02: ghi file tạm → đọc lại → `assert_frame_equal(check_exact=True)` → `replace`.
- Mỗi notebook mới có: (1) một cell kiểm tra đầu vào khớp SHA-256 trong report của bước trước, (2) các ví dụ biên bằng `assert` trên dữ liệu nhỏ tự tạo, (3) report `reports/NN_<buoc>/metrics.json` (`NN` = số notebook ghi report; xem `REPORT_DIRS` trong `src/rr/io.py`).
- Đặt tên notebook `notebooks/NN_ten_khong_dau.ipynb`. Markdown giải thích bằng tiếng Việt, code và tên cột bằng tiếng Anh.
- Chạy kiểm tra không tương tác: `jupyter nbconvert --to notebook --execute --inplace notebooks/<file>.ipynb`.
- Không commit `*.parquet`, `data/raw/*.csv` và các file lớn (đã có trong `.gitignore`).

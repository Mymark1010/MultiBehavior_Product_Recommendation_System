# Làm việc nhóm

1. Clone repo, tạo môi trường theo README và chạy unit tests.
2. Tạo branch: `git switch -c feature/ten-cong-viec`.
3. Với công việc mô hình, dùng `data/multibehavior/` và giao thức chung; không thay split riêng để cải thiện kết quả.
4. Chạy `python -m unittest discover -s tests -v` trước khi tạo PR. Khi sửa pipeline, chạy lại raw và cập nhật báo cáo/manifest.
5. PR nêu thay đổi, cách kiểm tra, ảnh hưởng đến schema/protocol và metric nếu có.

Không commit token, `.env`, virtualenv, raw, Parquet hoặc model checkpoints. Không ghi kết quả thử nghiệm ghi đè báo cáo chuẩn; mỗi experiment có config/seed và tên riêng.

Không thay các notebook khảo sát ban đầu nếu công việc không cần; viết notebook/module mới cho mô hình. Notebook mới nên xóa output trước khi commit để tránh lưu dữ liệu mẫu hoặc đường dẫn máy cá nhân.

Chủ repo có thể mời thành viên trong GitHub Settings → Collaborators. Mẫu CI nằm trong `docs/ci-workflow.yml`; chuyển thành `.github/workflows/ci.yml` khi có quyền workflow để kích hoạt. Khi nhóm thống nhất, bật yêu cầu pull request và CI thành công cho nhánh main.

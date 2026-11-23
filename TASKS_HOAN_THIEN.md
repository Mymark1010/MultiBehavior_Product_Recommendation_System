# Danh sách việc cần hoàn thiện — dành cho Claude Code

Trạng thái rà soát: commit `3ac0327` (22/11/2026); cập nhật 23/11/2026 theo protocol v1.2.0 (bộ phương pháp theo tài liệu tổng quan phương pháp gợi ý theo phiên trên RetailRocket). Đọc `CLAUDE.md` trước.
Làm theo thứ tự **P0 → P1 → P2**. Mỗi task có *Việc cần làm* và *Tiêu chí hoàn thành*. Chỉ đánh dấu xong khi mọi tiêu chí đạt.
Mục **[CẦN XÁC NHẬN]** phải hỏi người dùng trước khi làm.

Cách dùng: mở Claude Code tại thư mục gốc repo và nhắn
> "Đọc CLAUDE.md và TASKS_HOAN_THIEN.md, làm task T01."

Sau mỗi task, cập nhật ô `[ ]` thành `[x]` trong file này và commit riêng.

---

## P0 — Sửa repo để người khác clone và chạy được

### [x] T01. Bổ sung hoặc gỡ 5 file README đang trỏ tới nhưng không có trong repo

File thiếu:
- `docs/HUONG_DAN_TIEN_XU_LY_DU_LIEU.md`
- `docs/PHUONG_PHAP_DE_XUAT_CHI_TIET.md`
- `docs/PHUONG_PHAP_DE_XUAT_CHI_TIET.docx`
- `docs/PIPELINE_HOAN_THANH_PROJECT.md`
- `notebooks/kiem_tra_ket_qua_xu_ly.ipynb` — sinh ra `reports/data_validation/metrics.json` (36 check, PASS) đã commit

Việc cần làm:
1. Chạy `git status --ignored` và tìm 5 file trên ở máy local. Nếu có thì `git add` và commit.
2. Nếu không có file `kiem_tra_ket_qua_xu_ly.ipynb`: viết lại notebook này. Danh sách check phải khớp đúng các khóa `check` trong `reports/data_validation/metrics.json`: file_versions, notebook_lineage, raw_to_clean_exact_values, event_count_balance, clean_schema, purchase_ids, valid_events, clean_uniqueness, utc, clean_to_sessionized_exact_values, session_no_excess_splits, session_required_splits, one_visitor_per_session, session_summary, session_query_count, one_group_per_session_timestamp, group_summary, group_one_session, group_one_timestamp, group_order, event_group_links, time_cutoffs, boundary_drop, train/validation/test_time_range, event_split_links, group_split_links, session_not_shared, train/validation/test/boundary_drop_file_exact_values, partition_balance, strict_time_order, inputs_unchanged. Notebook chỉ đọc, không ghi đè dữ liệu.
3. Với các file docs không khôi phục được: xóa link khỏi README, hoặc ghi rõ "(chưa có)".

Tiêu chí hoàn thành:
- Lệnh kiểm tra link cục bộ trong README.md, docs/*.md và reports/**/*.md không còn link hỏng.
- Notebook kiểm tra chạy bằng `nbconvert --execute` ra `status: PASS`.

### [x] T02. Thêm `requirements.txt`

Việc cần làm: tạo `requirements.txt` theo phiên bản đã dùng thật trong report:
```
pandas==3.0.3
pyarrow==25.0.1
ipykernel
jupyter
nbstripout
```
Ghi chú Python 3.14 ở đầu file. Thêm `numpy`, `scipy` (cho KNN) và `torch` (cho GRU4Rec) khi làm tới P2. Không thêm trước.

Tiêu chí hoàn thành: `python -m venv .venv && pip install -r requirements.txt` rồi chạy notebook 01 → 02 thành công trên môi trường mới.

### [x] T03. Bỏ kernelspec gắn cứng

Hiện cả hai notebook dùng `"name": "retailrocket-project"`. Máy khác không có kernel tên này sẽ báo lỗi.

Việc cần làm: sửa `metadata.kernelspec` của mọi notebook thành
`{"name": "python3", "display_name": "Python 3", "language": "python"}`. Sửa đoạn README nói về kernel "RetailRocket (Python 3.14)" thành hướng dẫn chọn kernel của `.venv`.

Tiêu chí hoàn thành: `grep -r "retailrocket-project" notebooks/` không còn kết quả.

### [x] T04. Cố định ký tự xuống dòng để mã băm giống nhau trên mọi máy

Bằng chứng: `data/processed/split_config.json` trong repo (LF) có SHA-256 `6007b486…`, còn report ghi `44c889b4…`, tức mã băm của bản CRLF tạo trên Windows. Nguyên nhân: `Path.write_text()` trên Windows tự đổi `\n` thành `\r\n`.

Việc cần làm:
1. Tạo `.gitattributes`:
   ```
   * text=auto eol=lf
   *.docx binary
   *.pptx binary
   *.parquet binary
   ```
2. Trong `write_json()` của notebook 01 và 02, đổi thành `temporary.write_text(..., encoding="utf-8", newline="\n")`.
3. **Việc người dùng làm trên máy có dữ liệu:** chạy lại notebook 01 → 02 (→ notebook kiểm tra) để cập nhật report. Mã băm các file Parquet phải giữ nguyên. Chỉ mã băm `split_config.json` được phép thay đổi.

Tiêu chí hoàn thành: `sha256sum data/processed/split_config.json` khớp `reports/session_split/metrics.json → split_config.sha256` trên một bản clone mới.

### [x] T05. Kiểm tra Parquet theo nội dung, không chỉ theo byte

Mã băm file Parquet thay đổi khi pyarrow/pandas khác phiên bản, kể cả khi dữ liệu giống hệt. Người khác chạy lại sẽ thấy "lệch" dù không có lỗi.

Việc cần làm: trong `save_verified()` (notebook 02) và phần tương ứng của notebook 01, ghi thêm `content_sha256` vào report. Tính trên bảng đã sắp theo `event_id` (hoặc khóa chính của bảng), dùng `pd.util.hash_pandas_object(frame, index=False)` rồi băm SHA-256 mảng kết quả. Giữ `sha256` theo byte để truy vết phiên bản. Khi đối chiếu giữa các máy thì dùng `content_sha256`.

Tiêu chí hoàn thành: report có cả hai trường; README giải thích ngắn gọn sự khác nhau giữa hai trường.

### [x] T06. Gỡ đường dẫn cá nhân khỏi repo

Hiện có 6 file chứa đường dẫn tuyệt đối của máy cá nhân (thư mục người dùng Windows và thư mục project trên ổ D): README.md, 2 notebook, và 3 file `reports/*/metrics.json`.

Việc cần làm:
1. Trong notebook: in `ROOT.name` và đường dẫn tương đối (`SOURCE.relative_to(ROOT)`) thay cho đường dẫn tuyệt đối. Trong report: bỏ `python_executable`, giữ `python_version`.
2. Chạy `nbstripout --install` và thêm `*.ipynb filter=nbstripout` vào `.gitattributes`, để output notebook không vào git. Bằng chứng kết quả nằm trong `reports/`.
3. Xóa đoạn đường dẫn máy cá nhân trong README.

Tiêu chí hoàn thành: tìm tên người dùng Windows và tên thư mục ổ D của máy cá nhân bằng `grep -rI` (trừ `.git/`, `.venv/`, `data/`) không còn kết quả.

### [x] T07. Viết lại phần mở đầu README

Việc cần làm:
1. Thêm mục **"Chạy nhanh"** ở đầu README: clone → tạo venv → `pip install -r requirements.txt` → tải dữ liệu → chạy notebook 01, 02, kiểm tra.
2. Thêm hướng dẫn tải dữ liệu: Kaggle dataset `retailrocket/ecommerce-dataset`, lệnh `kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw --unzip`, và danh sách 4 file cần có kèm số dòng, SHA-256 của `events.csv` (`3745aa83…`, lấy từ `reports/data_validation/metrics.json`).
3. Chuyển bảng hiện trạng và ghi chú dài sang `docs/HIEN_TRANG.md`. README chính nên dưới khoảng 150 dòng.

Tiêu chí hoàn thành: một người chỉ đọc README có thể chạy lại 01 → 02 mà không cần hỏi thêm.

---

## P1 — Quyết định phương pháp (chốt trước khi viết mô hình)

Ghi mọi quyết định vào `configs/protocol.json` (tạo mới), có `version` và `decided_at`. Các notebook 04+ đọc giao thức từ file này, không gắn cứng trong code.

### [x] D1. [CẦN XÁC NHẬN] Xử lý target trùng item đã có trong prefix

Số liệu (`reports/gru_readiness`): 41,01% lượt item trong target train đã có trong prefix. 94.157 phiên train đủ mẫu chỉ có một item phân biệt. Gợi ý lại item cuối có thể đạt điểm cao một cách giả tạo.

Đề xuất mặc định:
- Giữ nguyên dữ liệu. Thêm cờ `target_is_repeat` cho từng (query, item).
- Mọi bảng kết quả báo cáo 3 nhóm: tất cả / chỉ target mới / chỉ target lặp.
- Thêm một baseline "Repeat-last": gợi ý lại các item trong prefix theo độ gần đây, để đo mức điểm của gợi ý lặp.
- Ablation tùy chọn: gộp các nhóm liên tiếp có cùng (item, hành vi) trong một phiên.

### [x] D2. [CẦN XÁC NHẬN] Có huấn luyện lại trên train + validation trước khi chạy test không

Số liệu: tỷ lệ target ngoài vocabulary train là 5,31% ở validation và 9,33% ở test. 8.613 query test có toàn bộ target ngoài vocabulary. Trần Recall chỉ do vocabulary là 0,907.

Đề xuất mặc định: chọn cấu hình trên validation, rồi fit lại popularity/KNN/GRU trên train + validation với đúng cấu hình đó (số epoch cố định theo validation), rồi chạy test một lần. Báo cáo cả hai biến thể nếu đủ thời gian.

### [x] D3. [CẦN XÁC NHẬN] Phiên rất dài

Số liệu: 125 phiên train có hơn 100 nhóm, đóng góp 18.348 query. Phiên dài nhất có 417 nhóm, phiên lâu nhất 12,1 giờ. 1% phiên dài nhất đóng góp 14,53% query.

Đề xuất mặc định: không loại phiên. Với GRU, cắt BPTT/độ dài prefix tối đa (ví dụ 50 nhóm gần nhất) chỉ khi huấn luyện. Báo cáo kết quả theo nhóm độ dài prefix (1, 2–4, 5–9, 10+).

### [x] D4. [CẦN XÁC NHẬN] Hàm mất mát GRU4Rec

Word đề xuất cross-entropy, slide ghi BPR/TOP1. Đề xuất mặc định: cross-entropy trên toàn bộ vocabulary (vocabulary cỡ khoảng 213k nên khả thi trên GPU), hoặc sampled softmax nếu thiếu bộ nhớ. Với target nhiều item (0,19% query train), dùng nhãn chia đều trên các item đúng. BPR-max là ablation. Cập nhật Word/slide cho thống nhất.

### [x] D5. [CẦN XÁC NHẬN] Phạm vi vocabulary

Đề xuất mặc định: vocabulary = mọi item xuất hiện trong `train`, kể cả phiên một nhóm (độ phủ test 90,7%, so với 86,3% nếu chỉ dùng phiên ≥2 nhóm). Mã hóa `PAD=0`, `UNK=1`, item từ 2. Thứ tự sắp theo itemid để tái lập được.

### [x] D6. [CẦN XÁC NHẬN] Code dùng chung giữa các notebook

Hiện `fingerprint()`, `write_json()`, `require()` bị chép lại ở mỗi notebook. README ghi notebook "chạy độc lập với .py". Đề xuất: từ notebook 03 trở đi, đặt hàm dùng chung và đặc biệt **hàm metric** vào `src/rr/` (`io.py`, `metrics.py`, `candidates.py`), có `tests/` chạy bằng `pytest`. Notebook import từ đó. Nếu người dùng muốn giữ nguyên tắc độc lập thì vẫn tạo `src/rr/metrics.py` và test, rồi chép hàm vào notebook kèm assert so sánh.

---

## P2 — Triển khai phần còn thiếu (notebook 03 → 11)

Số liệu kỳ vọng bên dưới lấy từ README và report hiện có. Kết quả lệch thì dừng và báo người dùng, không tự chỉnh cho khớp.

### [x] T08. Notebook 03 — lịch sử thuộc tính và catalog

- Đầu vào: `data/raw/item_properties_part1.csv`, `item_properties_part2.csv`, `category_tree.csv`. Đọc theo khối (`chunksize`), dữ liệu có 20.275.902 dòng.
- Đầu ra:
  - `data/processed/item_property_history.parquet`: `itemid int64, timestamp int64, property {categoryid, available}, value string, source_part int8`. Sắp theo `itemid, timestamp`.
  - `data/processed/category_tree.parquet`: `categoryid, parentid (nullable), is_root`.
  - `data/processed/item_catalog.parquet`: `itemid, first_event_ms, first_property_ms, known_at_ms = min(hai cột trên), has_metadata bool`.
  - `reports/metadata/metrics.json` với SHA-256 của cả 3 CSV.
- Tiêu chí (khớp khảo sát cũ): tổng 20.275.902 dòng thuộc tính; không có dòng trùng tuyệt đối xuyên hai file; 2.291.853 cập nhật categoryid/available; không có xung đột cùng (item, property, timestamp); 49.815/235.061 item trong log không có metadata; 1.669 danh mục, 25 gốc, không có chu trình; 30 mã danh mục ngoài cây thuộc 137 item, không item nào trong số đó có trong events; 137.193 sự kiện xảy ra trước bản thuộc tính đầu tiên.
- `known_at_ms` chỉ tính từ sự kiện thuộc `train` khi dùng cho lọc ứng viên ở giai đoạn train. Ghi rõ quy tắc vào `configs/protocol.json`.

### [x] T09. Notebook 04 — vocabulary, metadata tại t và bộ mẫu prefix–target

- Đầu vào: `timestamp_groups.parquet`, `events_sessionized.parquet`, các đầu ra của T08, `configs/protocol.json`.
- Đầu ra:
  - `data/processed/item_vocab.parquet`: `itemid, item_idx` theo D5.
  - `data/processed/samples/{train,validation,test}_queries.parquet`: `query_id, split, session_id, visitorid, query_time_ms (timestamp của nhóm cuối prefix), prefix_len_groups, target_group_id`.
  - `data/processed/samples/{split}_targets.parquet` (dạng dài): `query_id, itemid, item_idx (UNK nếu ngoài vocab), behaviors (list), target_is_repeat, in_vocab, candidate_eligible`.
  - Prefix không lưu trùng. Đọc lại từ `events_sessionized` theo `session_id` và `group_index <= prefix_len_groups`. Viết hàm `get_prefix(query)` và test nó.
  - Metadata as-of: `pd.merge_asof(..., by="itemid", direction="backward")` trên `timestamp <= query_time_ms`.
- Tiêu chí:
  - Số query: train 806.025 / validation 92.566 / test 92.468.
  - Số lượt target: validation 92.794, test 92.634.
  - Với vocabulary toàn train: OOV validation 4.923 (5,31%), test 8.639 (9,33%); query có toàn bộ target ngoài vocab: 4.915 / 8.613.
  - Query có target nhiều item: 1.519 / 177 / 140.
  - Test tự động: không target nào có timestamp ≤ `query_time_ms`; không prefix nào chứa sự kiện của nhóm target; không dùng metadata có timestamp > `query_time_ms`.

### [x] T10. Notebook 05 — bộ đánh giá và Popularity

- `metrics.py` (hoặc cell tương đương) cho mỗi query với danh sách Top-K không trùng và tập target `G`:
  - `hit@K = 1 nếu |TopK ∩ G| > 0`
  - `recall@K = |TopK ∩ G| / |G|`
  - `mrr@K = 1 / rank của hit đầu tiên, 0 nếu không có hit`
  - `ndcg@K = DCG / IDCG`, với `DCG = Σ_{hit tại rank r} 1/log2(r+1)` và `IDCG = Σ_{r=1}^{min(|G|,K)} 1/log2(r+1)`
  - Trung bình trên mọi query có `G` không rỗng, kể cả query có target ngoài vocab (tính 0 điểm). K ∈ {10, 20}. Chỉ số chính: NDCG@20.
- Test bằng ví dụ tính tay (ít nhất 6 ca: không hit, hit rank 1, hit rank K, nhiều target, trùng item trong TopK bị từ chối, K > số ứng viên).
- Lọc ứng viên tại `t`: `known_at_ms <= t`, available gần nhất `!= 0` (unknown vẫn giữ), có trong vocab. Áp dụng cho mọi mô hình và cho fallback.
- Popularity: (a) đếm đều; (b) có trọng số hành vi `w_view=1, w_cart, w_buy`, tìm lưới trên validation (ví dụ {1, 2, 5, 10}).
- Baseline Repeat-last theo D1.
- Đầu ra: `reports/eval/popularity_validation.json`, `reports/eval/repeat_last_validation.json`. Có bảng tách theo target mới/lặp và theo độ dài prefix.

### [x] T11. Notebook 06 — họ kNN (SKNN, V-SKNN, STAN, VSTAN), SR và ItemKNN

Cập nhật v1.2.0: thêm VSTAN (STAN + IDF) và luật tuần tự SR; S-POP nằm ở notebook 05.


- ItemKNN: ma trận phiên × item thưa (scipy.sparse) từ train, giá trị = trọng số hành vi. Cosine item–item, giữ top-M láng giềng mỗi item. Điểm = tổng có trọng số độ gần đây theo các item trong prefix.
- Session-KNN: chỉ tìm láng giềng trong phiên train. Tham số k láng giềng, số phiên mẫu, trọng số theo thời gian. Chỉ tune trên validation.
- Ghi thời gian fit, thời gian truy vấn trung bình mỗi query và tỷ lệ fallback.
- Tiêu chí: cùng bộ query, ứng viên và hàm metric như T10. Có báo cáo so sánh trọng số đều với trọng số theo hành vi (câu hỏi nghiên cứu 1).

### [x] T12. Notebook 07 — GRU4Rec (PyTorch)

Cập nhật v1.2.0: tự cài theo bộ tính năng bản chính thức (BPR-max và cross-entropy, 2048 negative dùng chung, constrained embedding), cấu hình xuất phát 224 hidden, batch 80; loss chọn trên validation.


- Input mỗi bước = mean embedding các item trong nhóm, cộng embedding hành vi (view/cart/buy), theo README. Reset hidden state giữa các phiên.
- Loss theo D4. Dropout và hidden size chọn trên validation. Early stopping theo NDCG@20 validation. Seed cố định.
- Bước đầu: chạy smoke test trên khoảng 1% train để kiểm tra input–target lệch đúng một bước và loss giảm.
- Lưu `models/gru4rec/{config.json, model.pt}` (không commit file `.pt` lớn) và `reports/eval/gru4rec_validation.json`.
- Ablation cho câu hỏi nghiên cứu 2: xáo trộn thứ tự nhóm trong prefix, hoặc chỉ dùng nhóm cuối.

### [x] T12b. Notebook 08 — SR-GNN (neural đối chứng)

- Đồ thị: nút = item trong prefix (≤ 50 nhóm gần nhất), cạnh từ mọi item nhóm g sang mọi item nhóm g+1, không thứ tự trong nhóm, embedding hành vi cộng vào nút.
- Sampled softmax như GRU4Rec, early stopping theo NDCG@20 validation. Đầu ra `models/srgnn/`, `reports/eval/srgnn_validation.json`.

### [x] T12c. Notebook 09 — reranker kiểu HySAR (LightGBM)

- Ứng viên = top-100 kNN tốt nhất ∪ top-100 GRU4Rec ∪ item trong prefix; đặc trưng hành vi, độ phổ biến, danh mục as-of t.
- Học trên validation (mô hình gốc fit trên train), kiểm định chéo 2 fold theo phiên, báo cáo out-of-fold.

### [ ] T13. Notebook 10 — thực nghiệm cuối trên test

Trạng thái 24/11/2026: đã chạy test một lần (`reports/final/metrics.json`, `reports/final/BAO_CAO_KET_QUA.md`). **Còn lại:** cập nhật Word/slide bằng số liệu thật.

Cập nhật: chạy diễn tập `RR_FINAL_SPLIT=validation` trước; báo cáo thêm bài toán 2 (cart/mua ở nhóm kế tiếp) và paired bootstrap so với Repeat-last.


- Đóng băng `configs/protocol.json` và cấu hình tốt nhất của 4 mô hình, ghi mã băm vào report trước khi chạy test.
- Áp dụng D2. Chạy test **một lần**.
- Báo cáo: bảng chính (Recall/Hit/MRR/NDCG @10, @20), bảng theo target mới/lặp, theo độ dài prefix, theo độ phổ biến item; thời gian train/query; tỷ lệ fallback; khoảng tin cậy bootstrap 95% theo query cho NDCG@20.
- Đầu ra: `reports/final/metrics.json` và `reports/final/BAO_CAO_KET_QUA.md`. Cập nhật Word/slide bằng số liệu thật.

### [x] T14. Notebook 11 — demo

- Tải artifact đã lưu. Nhận prefix là danh sách (itemid, event, timestamp). Trả về bảng `rank, item_id, score, category_id, is_fallback` như README mục 3.
- Có 3 ví dụ lấy từ test, kết quả khớp với lần chạy đánh giá cho cùng query.
- Hướng dẫn chạy từ kernel mới trong README.

---

## Ghi chú cho người dùng

- T01 (bước 1), T04 (bước 3) và mọi bước cần chạy trên dữ liệu thật phải làm trên máy có `data/raw/`. Claude Code chạy local trong Antigravity sẽ làm được, phiên Claude trên cloud thì không.
- Thứ tự nên làm nếu gấp: T02 → T03 → T04 → T07 → D1–D5 → T09 → T10. Không có T09 và T10 thì chưa so sánh được mô hình nào.

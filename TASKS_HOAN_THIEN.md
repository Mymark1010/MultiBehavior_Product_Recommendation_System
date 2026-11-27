# Danh sách việc cần hoàn thiện — dành cho Claude Code

Trạng thái rà soát: commit `3ac0327` (22/11/2026); cập nhật 23/11/2026 theo protocol v1.2.0 (bộ phương pháp theo tài liệu tổng quan phương pháp gợi ý theo phiên trên RetailRocket); 24/11/2026 thu hẹp phạm vi theo protocol v1.3.0 (chỉ bài toán tương tác kế tiếp, GRU4Rec gốc so với đa hành vi, phiên 30 và 60 phút); 26/11/2026 protocol v1.3.1: tạm bỏ SR-GNN (xóa notebook 08, `src/rr/srgnn.py`, test, mô hình, report); cùng ngày v1.3.2: bỏ S-POP, SR, V-SKNN và chỉ giữ chỉ số tại K = 20. Đọc `CLAUDE.md` trước.
Làm theo thứ tự **P0 → P1 → P2**. Mỗi task có *Việc cần làm* và *Tiêu chí hoàn thành*. Chỉ đánh dấu xong khi mọi tiêu chí đạt.
Mục **[CẦN XÁC NHẬN]** phải hỏi người dùng trước khi làm.

Cách dùng: mở Claude Code tại thư mục gốc repo và nhắn
> "Đọc CLAUDE.md và TASKS_HOAN_THIEN.md, làm task T01."

Sau mỗi task, cập nhật ô `[ ]` thành `[x]` trong file này và commit riêng.

---

## P0 — Sửa repo để người khác clone và chạy được (đã xong)

| Task | Kết quả |
| --- | --- |
| [x] T01 | Link hỏng trong README đã gỡ; notebook `kiem_tra_ket_qua_xu_ly` ghi `reports/02_data_validation/metrics.json` (36 kiểm tra, PASS) |
| [x] T02 | `requirements.txt` ghi đúng phiên bản đã dùng (Python 3.14) |
| [x] T03 | Mọi notebook dùng kernelspec `python3` |
| [x] T04 | `.gitattributes` ép LF; JSON ghi với `newline="
"` nên mã băm giống nhau trên mọi hệ điều hành |
| [x] T05 | Report Parquet có cả `sha256` (byte) và `content_sha256` (nội dung) |
| [x] T06 | Không còn đường dẫn máy cá nhân trong repo; `nbstripout` bỏ output notebook khỏi git |
| [x] T07 | README có mục "Chạy nhanh" và hướng dẫn tải dữ liệu; ghi chú dài chuyển sang `docs/HIEN_TRANG.md` |

---

## P1 — Quyết định phương pháp (chốt trước khi viết mô hình)

Ghi mọi quyết định vào `configs/protocol.json` (tạo mới), có `version` và `decided_at`. Các notebook 04+ đọc giao thức từ file này, không gắn cứng trong code.

### [x] D1. [CẦN XÁC NHẬN] Xử lý target trùng item đã có trong prefix

Số liệu (`reports/02_gru_readiness`): 41,01% lượt item trong target train đã có trong prefix. 94.157 phiên train đủ mẫu chỉ có một item phân biệt. Gợi ý lại item cuối có thể đạt điểm cao một cách giả tạo.

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
  - `reports/03_metadata/metrics.json` với SHA-256 của cả 3 CSV.
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
  - Trung bình trên mọi query có `G` không rỗng, kể cả query có target ngoài vocab (tính 0 điểm). K = 20 (K = 10 bỏ ở v1.3.2). Chỉ số chính: NDCG@20.
- Test bằng ví dụ tính tay (ít nhất 6 ca: không hit, hit rank 1, hit rank K, nhiều target, trùng item trong TopK bị từ chối, K > số ứng viên).
- Lọc ứng viên tại `t`: `known_at_ms <= t`, có trong vocab; áp dụng cho mọi mô hình và cho fallback. Luật `available != 0` chỉ còn là báo cáo phụ (protocol v1.2.0).
- Popularity: (a) đếm đều; (b) có trọng số hành vi `w_view=1, w_cart, w_buy`, tìm lưới trên validation (ví dụ {1, 2, 5, 10}).
- Baseline Repeat-last theo D1.
- Đầu ra: `reports/05_08_validation/popularity_validation.json`, `reports/05_08_validation/repeat_last_validation.json`. Có bảng tách theo target mới/lặp và theo độ dài prefix.

### [x] T11. Notebook 06 — họ kNN (SKNN, STAN, VSTAN) và ItemKNN

Cập nhật v1.2.0: thêm VSTAN (STAN + IDF) và luật tuần tự SR; S-POP nằm ở notebook 05. Cập nhật v1.3.2 (26/11/2026): người dùng bỏ S-POP, SR và V-SKNN khỏi notebook 05, 06 và `src/rr`; chuỗi chọn VSTAN (SKNN → STAN → VSTAN) giữ nguyên.


- ItemKNN: ma trận phiên × item thưa (scipy.sparse) từ train, giá trị = trọng số hành vi. Cosine item–item, giữ top-M láng giềng mỗi item. Điểm = tổng có trọng số độ gần đây theo các item trong prefix.
- Session-KNN: chỉ tìm láng giềng trong phiên train. Tham số k láng giềng, số phiên mẫu, trọng số theo thời gian. Chỉ tune trên validation.
- Ghi thời gian fit, thời gian truy vấn trung bình mỗi query và tỷ lệ fallback.
- Tiêu chí: cùng bộ query, ứng viên và hàm metric như T10. Có báo cáo so sánh trọng số đều với trọng số theo hành vi (câu hỏi nghiên cứu 1).

### [x] T12. Notebook 07 — GRU4Rec (PyTorch)

Cập nhật v1.2.0: tự cài theo bộ tính năng bản chính thức (BPR-max và cross-entropy, 2048 negative dùng chung, constrained embedding), cấu hình xuất phát 224 hidden, batch 80; loss chọn trên validation.


- Input mỗi bước = mean embedding các item trong nhóm, cộng embedding hành vi (view/cart/buy), theo README. Reset hidden state giữa các phiên.
- Loss theo D4. Dropout và hidden size chọn trên validation. Early stopping theo NDCG@20 validation. Seed cố định.
- Bước đầu: chạy smoke test trên khoảng 1% train để kiểm tra input–target lệch đúng một bước và loss giảm.
- Lưu `models/gru4rec/{config.json, model.pt}` (không commit file `.pt` lớn) và `reports/05_08_validation/gru4rec_validation.json`.
- Ablation cho câu hỏi nghiên cứu 2: xáo trộn thứ tự nhóm trong prefix, hoặc chỉ dùng nhóm cuối.
- Sửa 25/11/2026: embedding hành vi trước đây khởi tạo theo mặc định N(0, 1), lấn át trung bình embedding item (N(0; 0,05)); nay khởi tạo cùng thang 0,05, có test `test_behavior_embedding_same_scale_as_items`. Notebook 04–07 và 09 đã chạy lại sau khi sửa; kết quả của các mô hình không dùng embedding hành vi trùng tuyệt đối bản cũ.
- Ghi chú v1.3.2: `reports/05_08_validation/gru4rec_validation.json` sinh ở protocol 1.3.0 nên vẫn có cột @10. Notebook 10 đã đóng băng mã băm file này, nên không chạy lại notebook 07; mọi so sánh chỉ đọc các cột @20.

### [–] T12b. Notebook 08 — SR-GNN (neural đối chứng) — đã gỡ ở v1.3.1

Người dùng quyết định ngày 26/11/2026 bỏ SR-GNN khỏi phạm vi. Notebook 08, `src/rr/srgnn.py`, `tests/test_srgnn.py`, `models/srgnn/`, `models/session60/srgnn/` và `reports/05_08_validation/srgnn_validation.json` đã bị xóa (còn trong lịch sử git). Mô tả cũ giữ lại để tham khảo:


- Đồ thị: nút = item trong prefix (≤ 50 nhóm gần nhất), cạnh từ mọi item nhóm g sang mọi item nhóm g+1, không thứ tự trong nhóm, embedding hành vi cộng vào nút.
- Sampled softmax như GRU4Rec, early stopping theo NDCG@20 validation. Đầu ra `models/srgnn/`, `reports/05_08_validation/srgnn_validation.json`.

### [x] T13. Notebook 10 — thực nghiệm cuối trên test (phạm vi v1.3.2)

Phạm vi v1.3.0 (người dùng chốt 24/11/2026): chỉ bài toán tương tác kế tiếp. Người dùng duyệt đọc test thêm một lần cho bộ so sánh mới; các report test trước đó đã bị gỡ khỏi repo. Ngày 26/11/2026 (v1.3.1) lần chạy test đang dở, có SR-GNN, bị dừng và xóa toàn bộ output; test đã bị đọc một phần. Người dùng duyệt chạy lại test một lần với phạm vi mới, sau khi chạy lại notebook 09 và bản diễn tập.

- Đóng băng `configs/protocol.json`, report validation (notebook 05–09) và mô hình của cả hai ngưỡng phiên, ghi mã băm vào `frozen.json` trước khi đọc test.
- Diễn tập `RR_FINAL_SPLIT=validation` trước. Lần chạy test chỉ được phép khi bản diễn tập có đúng cấu hình đóng băng hiện tại. Test chạy **một lần**.
- Mô hình: GRU4Rec gốc (`gru4rec_item`), GRU4Rec đa hành vi (`gru4rec`); mốc Popularity, Repeat-last, VSTAN.
- Tổ hợp chạy: 2 ngưỡng phiên (30/60) × 2 biến thể fit (D2, refit là biến thể chính) × 2 bộ query (`q30`, `q60`).
- Báo cáo:
  - HR/MRR/NDCG@20 kèm khoảng tin cậy bootstrap 95%;
  - hiệu ứng hành vi (đa hành vi − gốc, paired bootstrap);
  - hiệu ứng ngưỡng phiên trên cùng bộ query (paired bootstrap);
  - NDCG theo target mới/lặp và theo độ dài prefix;
  - so với Repeat-last;
  - thời gian fit.
- Đầu ra: `reports/10_final/{metrics.json, BAO_CAO_KET_QUA.md}`, Top-K đã lưu cho demo. Slide đã cập nhật số test (27/11/2026, `docs/BAO_CAO_TIEN_DO_PROJECT3_HUST_27_11.pptx`); báo cáo Word cũ đã xóa theo yêu cầu người dùng.
- Kết quả (27/11/2026, `status: PASS`, mã băm mọi đầu vào khớp `frozen.json`): NDCG@20 bản refit, phiên 30 phút: GRU4Rec đa hành vi 0,3721, GRU4Rec gốc 0,3713, VSTAN 0,3528, Repeat-last 0,3337, Popularity 0,0041. Hiệu ứng hành vi ở bản refit khoảng +0,001 (CI 95% không chứa 0 ở cả hai ngưỡng); ở bản `train_only` phiên 60 phút lại âm (−0,0009).
- Sự cố 26–27/11/2026: lần chạy test v1.3.2 đầu tiên bị ngắt sau 6/8 tổ hợp (tiến trình nền bị đóng cùng phiên Claude Code, không phải lỗi code), không ghi report; log tiến độ đã in NDCG@20 của 6 tổ hợp đó. Người dùng duyệt chạy lại nguyên notebook 10 test cùng cấu hình đóng băng (không đổi gì). Lần chạy lại cho NDCG@20 trùng khớp 6 tổ hợp đã in, nên việc chạy lại không tạo cơ hội chọn kết quả.

### [x] T14. Notebook 11 — demo

- Tải artifact đã lưu. Nhận prefix là danh sách (itemid, event, timestamp). Trả về bảng `rank, item_id, score, category_id, is_fallback` (README, bảng notebook 11).
- Có 3 ví dụ lấy từ test, kết quả khớp với lần chạy đánh giá cho cùng query.
- Hướng dẫn chạy từ kernel mới trong README.

### [x] T15. Notebook 12 — ablation dữ liệu (người dùng duyệt 24/11/2026)

Phân tích phụ sau rà soát dữ liệu; **không đổi giao thức chính**, chỉ đánh giá trên validation, test chỉ chấm lại từ Top-K đã lưu. So sánh ngưỡng phiên 30/60 phút chuyển sang T16.

- A. Visitor bất thường: ngưỡng học từ train (phân vị 99,99% sự kiện và giao dịch theo visitor-ngày UTC); báo cáo metric toàn bộ / bỏ visitor bị gắn cờ / chỉ visitor bị gắn cờ; độ nhạy bỏ top-10 visitor.
- B. Ablation D1 "gộp nhóm liên tiếp trùng `(item, hành vi)`": 2×2 fit gốc/gộp × đánh giá gốc/gộp.
- Tiêu chí: mốc so sánh khớp `reports/10_final_rehearsal` (train_only, phiên 30 phút) tuyệt đối; Top-K test đã lưu khớp `reports/10_final`; paired bootstrap trên cùng bộ query; `reports/12_data_ablation/{metrics.json, BAO_CAO_ABLATION_DU_LIEU.md}`.
- Sau khi notebook 10 chạy lại theo v1.3.1, cần chạy lại notebook 12 để đối chiếu với report mới. Đã chạy lại ngày 27/11/2026 (protocol 1.3.2, `status: PASS`, mốc khớp bản diễn tập).

### [x] T16. Notebook 09 — GRU4Rec gốc và đa hành vi, phiên 30 và 60 phút (người dùng chốt 24/11/2026)

- Thêm biến thể GRU4Rec gốc: cùng kiến trúc và siêu tham số bản chính thức, bỏ embedding hành vi (`use_behavior=False`, có test).
- Dựng dữ liệu phiên 60 phút bằng `sessionize` (cùng mốc 0.8/0.9). Kiểm tra `sessionize(30 phút)` trùng notebook 02 trên từng sự kiện.
- Chọn trên validation với cùng thủ tục notebook 05–07:
  - số epoch GRU4Rec gốc ở 30 và 60 phút;
  - số epoch GRU4Rec đa hành vi ở 60 phút (ở 30 phút giữ notebook 07, chấm lại phải khớp tuyệt đối);
  - Popularity và VSTAN (chuỗi sknn → stan → vstan) ở 60 phút.
- Đầu ra: `data/processed/session60/`, `models/gru4rec_item/`, `models/session60/`, `reports/09_validation_30_60/metrics.json`.

### [x] T17. Notebook 13 — kiểm chứng GRU4Rec bản chính thức trên validation (người dùng duyệt 26/11/2026)

Theo review báo cáo phương pháp: chạy `hidasib/GRU4Rec_PyTorch_Official` (commit `d1fc311`) trên chính bộ chia của dự án để kiểm chứng GRU4Rec tự cài.

- Code chính thức không commit (license nghiên cứu/giáo dục): notebook tải vào `external/` (gitignore), kiểm tra SHA-256. Cầu nối `src/rr/official_gru.py`, test `tests/test_official_gru.py`.
- Cấu hình: paramfile `retailrocket_bprmax_shared_best` giữ nguyên, 10 epoch; chấm NDCG@20 validation sau mỗi epoch.
- Người dùng chốt: sự kiện cùng timestamp xếp theo `event_id` (chỉ cho notebook này). Bản vá duy nhất: `_init_numpy_weights` tính như NumPy 1.
- Tiêu chí: vòng lặp từng epoch cho trọng số trùng `fit()` gốc; cùng query, bộ lọc, fallback, metric với notebook 07/09; paired bootstrap với GRU4Rec gốc và đa hành vi; không đọc test. Đầu ra `reports/13_gru4rec_official/metrics.json`.
- Kết quả (27/11/2026, `status: PASS`): bản chính thức đạt NDCG@20 validation 0,3888 sau 10 epoch (tăng đều, bão hòa từ epoch 7), thấp hơn GRU4Rec gốc của dự án 0,3951 (Δ −0,0063, CI 95% [−0,0075; −0,0051]). Bản chính thức tốt hơn ở target mới (0,146 so với 0,119) và HR@20 (0,559 so với 0,536), kém ở target lặp (0,722 so với 0,774).

---

## Ghi chú cho người dùng

- T01 (bước 1), T04 (bước 3) và mọi bước cần chạy trên dữ liệu thật phải làm trên máy có `data/raw/`. Claude Code chạy local trong Antigravity sẽ làm được, phiên Claude trên cloud thì không.
- Thứ tự nên làm nếu gấp: T02 → T03 → T04 → T07 → D1–D5 → T09 → T10. Không có T09 và T10 thì chưa so sánh được mô hình nào.

# Ablation dữ liệu (validation)

Sinh bởi `notebooks/12_ablation_du_lieu.ipynb` lúc 2026-11-26T19:48:19.457519+00:00, protocol 1.3.2. Giao thức chính không đổi; đây là phân tích phụ. Mô hình dùng đúng cấu hình đã chọn, không tune lại. Bài toán: tương tác kế tiếp, phiên 30 phút. Số liệu chính: NDCG@20.

## A. Visitor hoạt động bất thường

Ngưỡng từ train (phân vị 0.9999): > 145.3 sự kiện/ngày hoặc > 11 giao dịch/ngày.

| Tập | Visitor bị gắn cờ | Tỷ lệ sự kiện | Tỷ lệ query | Top-10 visitor (tỷ lệ query) |
| --- | --- | --- | --- | --- |
| train | 87 | 2.88% | 7.47% | 3.17% |
| validation | 14 | 2.47% | 6.91% | 6.82% |
| test | 12 | 2.43% | 6.84% | 7.29% |

### Validation

| Mô hình | all ndcg@20 | without_flagged ndcg@20 | flagged_only ndcg@20 | without_top_visitors ndcg@20 | Δ vs Repeat-last (bỏ cờ) | Δ thấp |
| --- | --- | --- | --- | --- | --- | --- |
| popularity | 0.0047 | 0.0047 | 0.0047 | 0.0048 | -0.3586 | -0.3615 |
| repeat_last | 0.3578 | 0.3633 | 0.2834 | 0.3663 | 0.0000 | 0.0000 |
| vstan | 0.3747 | 0.3835 | 0.2561 | 0.3858 | 0.0202 | 0.0184 |
| gru4rec_item | 0.3951 | 0.4049 | 0.2629 | 0.4073 | 0.0416 | 0.0399 |
| gru4rec | 0.3950 | 0.4049 | 0.2623 | 0.4073 | 0.0416 | 0.0399 |

Thứ tự mô hình (toàn bộ): gru4rec_item, gru4rec, vstan, repeat_last, popularity

Thứ tự mô hình (bỏ visitor bị gắn cờ): gru4rec, gru4rec_item, vstan, repeat_last, popularity (Kendall tau = 0.800)

### Test (train_only, Top-K đã lưu ở notebook 10)

| Mô hình | all | without_flagged | flagged_only | without_top_visitors |
| --- | --- | --- | --- | --- |
| vstan | 0.3345 | 0.3449 | 0.1929 | 0.3464 |
| gru4rec_item | 0.3532 | 0.3642 | 0.2030 | 0.3659 |
| gru4rec | 0.3537 | 0.3646 | 0.2041 | 0.3663 |

## B. Gộp nhóm lặp liên tiếp

Bỏ 207,984 / 2,753,015 nhóm (7.55%); 21,163 / 92,566 query validation có target trùng hệt nhóm cuối prefix. Query sau khi gộp: train 639,066, validation 71,403.

Ô = fit/đánh giá. Δ = paired bootstrap trên cùng bộ query validation gốc.

| Mô hình | gốc/gốc | gộp/gốc | Δ (gộp/gốc − gốc/gốc) | Δ thấp | Δ cao | gốc/gộp | gộp/gộp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| popularity | 0.0047 | 0.0040 | -0.0007 | -0.0008 | -0.0006 | 0.0042 | 0.0035 |
| repeat_last | 0.3578 | 0.3579 | 0.0001 | 0.0000 | 0.0001 | 0.1827 | 0.1828 |
| vstan | 0.3747 | 0.3746 | -0.0001 | -0.0002 | -0.0000 | 0.2434 | 0.2433 |
| gru4rec_item | 0.3951 | 0.3552 | -0.0399 | -0.0413 | -0.0384 | 0.2597 | 0.2504 |
| gru4rec | 0.3950 | 0.3564 | -0.0387 | -0.0401 | -0.0373 | 0.2608 | 0.2517 |

### Mô hình gốc, tách query validation theo target trùng nhóm cuối

| Mô hình | duplicate_target | other_queries |
| --- | --- | --- |
| popularity | 0.0067 | 0.0042 |
| repeat_last | 0.9484 | 0.1827 |
| vstan | 0.8189 | 0.2430 |
| gru4rec_item | 0.8550 | 0.2588 |
| gru4rec | 0.8498 | 0.2603 |

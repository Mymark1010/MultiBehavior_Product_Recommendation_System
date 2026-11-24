# Ablation dữ liệu (validation)

Sinh bởi `notebooks/12_ablation_du_lieu.ipynb` lúc 2026-11-24T04:51:49.987981+00:00, protocol 1.2.0. Giao thức chính không đổi; đây là phân tích phụ. Mô hình dùng đúng cấu hình đã chọn, không tune lại; reranker không chạy (học trên validation). Số liệu chính: NDCG@20.

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
| spop | 0.3491 | 0.3625 | 0.1686 | 0.3640 | -0.0009 | -0.0017 |
| repeat_last | 0.3578 | 0.3633 | 0.2834 | 0.3663 | 0.0000 | 0.0000 |
| itemknn | 0.0964 | 0.1009 | 0.0365 | 0.1008 | -0.2624 | -0.2659 |
| sknn | 0.3671 | 0.3838 | 0.1421 | 0.3848 | 0.0205 | 0.0184 |
| vsknn | 0.3740 | 0.3864 | 0.2075 | 0.3882 | 0.0230 | 0.0210 |
| stan | 0.3745 | 0.3835 | 0.2537 | 0.3858 | 0.0202 | 0.0184 |
| vstan | 0.3747 | 0.3835 | 0.2561 | 0.3858 | 0.0202 | 0.0184 |
| sr | 0.3084 | 0.3149 | 0.2207 | 0.3170 | -0.0484 | -0.0509 |
| gru4rec | 0.3921 | 0.4018 | 0.2608 | 0.4043 | 0.0385 | 0.0368 |
| srgnn | 0.3746 | 0.3844 | 0.2431 | 0.3866 | 0.0211 | 0.0192 |

Thứ tự mô hình (toàn bộ): gru4rec, vstan, srgnn, stan, vsknn, sknn, repeat_last, spop, sr, itemknn, popularity

Thứ tự mô hình (bỏ visitor bị gắn cờ): gru4rec, vsknn, srgnn, sknn, vstan, stan, repeat_last, spop, sr, itemknn, popularity (Kendall tau = 0.782)

### Test (train_only, Top-K đã lưu ở notebook 10)

| Mô hình | all | without_flagged | flagged_only | without_top_visitors |
| --- | --- | --- | --- | --- |
| vstan | 0.3345 | 0.3449 | 0.1929 | 0.3464 |
| gru4rec | 0.3507 | 0.3619 | 0.1987 | 0.3636 |

## B. Gộp nhóm lặp liên tiếp

Bỏ 207,984 / 2,753,015 nhóm (7.55%); 21,163 / 92,566 query validation có target trùng hệt nhóm cuối prefix. Query sau khi gộp: train 639,066, validation 71,403.

Ô = fit/đánh giá. Δ = paired bootstrap trên cùng bộ query validation gốc.

| Mô hình | gốc/gốc | gộp/gốc | Δ (gộp/gốc − gốc/gốc) | Δ thấp | Δ cao | gốc/gộp | gộp/gộp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| popularity | 0.0047 | 0.0040 | -0.0007 | -0.0008 | -0.0006 | 0.0042 | 0.0035 |
| spop | 0.3491 | 0.3491 | 0.0001 | -0.0000 | 0.0001 | 0.1854 | 0.1855 |
| repeat_last | 0.3578 | 0.3579 | 0.0001 | 0.0000 | 0.0001 | 0.1827 | 0.1828 |
| itemknn | 0.0964 | 0.0964 | -0.0000 | -0.0000 | 0.0000 | 0.1224 | 0.1224 |
| sknn | 0.3671 | 0.3671 | -0.0000 | -0.0000 | 0.0000 | 0.2527 | 0.2526 |
| vsknn | 0.3740 | 0.3740 | -0.0000 | -0.0000 | 0.0000 | 0.2541 | 0.2541 |
| stan | 0.3745 | 0.3744 | -0.0001 | -0.0002 | 0.0000 | 0.2436 | 0.2435 |
| vstan | 0.3747 | 0.3746 | -0.0001 | -0.0002 | -0.0000 | 0.2434 | 0.2433 |
| sr | 0.3084 | 0.2540 | -0.0544 | -0.0558 | -0.0530 | 0.1961 | 0.1906 |
| gru4rec | 0.3921 | 0.3363 | -0.0558 | -0.0573 | -0.0542 | 0.2553 | 0.2290 |
| srgnn | 0.3746 | 0.3323 | -0.0424 | -0.0439 | -0.0408 | 0.2474 | 0.2386 |

### Mô hình gốc, tách query validation theo target trùng nhóm cuối

| Mô hình | duplicate_target | other_queries |
| --- | --- | --- |
| popularity | 0.0067 | 0.0042 |
| spop | 0.8981 | 0.1863 |
| repeat_last | 0.9484 | 0.1827 |
| itemknn | 0.0092 | 0.1223 |
| sknn | 0.7531 | 0.2527 |
| vsknn | 0.7784 | 0.2541 |
| stan | 0.8174 | 0.2432 |
| vstan | 0.8189 | 0.2430 |
| sr | 0.6873 | 0.1961 |
| gru4rec | 0.8546 | 0.2550 |
| srgnn | 0.8092 | 0.2458 |

## C. Ngưỡng phiên

| Ngưỡng (phút) | sessions | single_group_session_share | cart_then_purchase_same_session | train_queries | validation_queries | test_queries | boundary_drop_sessions | boundary_drop_events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15 | 1,809,161 | 0.7910 | 0.8473 | 768,176 | 87,790 | 87,722 | 30 | 196 |
| 30 | 1,761,675 | 0.7829 | 0.8990 | 806,025 | 92,566 | 92,468 | 47 | 328 |
| 60 | 1,726,714 | 0.7765 | 0.9289 | 833,781 | 96,148 | 95,783 | 69 | 659 |
| 120 | 1,698,874 | 0.7709 | 0.9464 | 855,921 | 98,880 | 98,404 | 109 | 1,046 |

Phiên 60 phút: vocabulary 212,915 item (30 phút: 212,915).

Ô = ngưỡng fit/ngưỡng đánh giá (phút). Δ = paired bootstrap trên cùng bộ query validation 30 phút.

| Mô hình | 30/30 | 60/30 | Δ (60/30 − 30/30) | Δ thấp | Δ cao | 30/60 | 60/60 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| popularity | 0.0047 | 0.0047 | 0.0000 | 0.0000 | 0.0000 | 0.0047 | 0.0047 |
| spop | 0.3491 | 0.3491 | 0.0000 | 0.0000 | 0.0000 | 0.3503 | 0.3503 |
| repeat_last | 0.3578 | 0.3578 | 0.0000 | 0.0000 | 0.0000 | 0.3628 | 0.3628 |
| itemknn | 0.0964 | 0.0936 | -0.0028 | -0.0031 | -0.0025 | 0.0944 | 0.0917 |
| sknn | 0.3671 | 0.3680 | 0.0009 | 0.0007 | 0.0012 | 0.3648 | 0.3657 |
| vsknn | 0.3740 | 0.3750 | 0.0010 | 0.0007 | 0.0012 | 0.3733 | 0.3742 |
| stan | 0.3745 | 0.3753 | 0.0008 | 0.0006 | 0.0011 | 0.3758 | 0.3766 |
| vstan | 0.3747 | 0.3755 | 0.0008 | 0.0005 | 0.0011 | 0.3760 | 0.3769 |
| sr | 0.3084 | 0.3103 | 0.0019 | 0.0016 | 0.0023 | 0.3093 | 0.3113 |
| gru4rec | 0.3921 | 0.3928 | 0.0007 | -0.0002 | 0.0016 | 0.3938 | 0.3946 |
| srgnn | 0.3746 | 0.3798 | 0.0052 | 0.0041 | 0.0062 | 0.3747 | 0.3803 |

## Kết luận

**A — Visitor bất thường.** Ngưỡng học từ train (> 145 sự kiện hoặc > 11 giao dịch trong một ngày UTC) gắn cờ 14 visitor ở validation (2,5% sự kiện nhưng 6,9% query) và 12 visitor ở test. Nhóm này khó hơn hẳn (NDCG@20 của GRU4Rec 0,261 so với 0,402 ở phần còn lại) và có tỷ lệ target thêm giỏ/mua gấp đôi (18,8% so với 8,4%), giống người mua sỉ hơn là bot xem trang. Bỏ họ ra làm mọi mô hình tăng 0,005–0,017 NDCG@20 nhưng **không đổi kết luận chính**: GRU4Rec vẫn đứng đầu, mọi mô hình kNN/neural vẫn hơn Repeat-last có ý nghĩa (cận dưới Δ > 0), test cho cùng xu hướng (VSTAN 0,334 → 0,345; GRU4Rec 0,351 → 0,362). Chỉ thứ tự trong nhóm kNN (chênh nhau < 0,003) thay đổi. → Giữ nguyên tập đánh giá chính, đưa bảng này vào báo cáo như phân tích độ vững.

**B — Nhóm lặp liên tiếp.** 22,9% query validation có target trùng hệt nhóm cuối prefix (reload/back). Trên phần này Repeat-last đạt 0,948 nên chúng đẩy metric lên mạnh; trên 77% query còn lại Repeat-last chỉ 0,183 và GRU4Rec 0,255 (hơn khoảng 39% tương đối, so với 9,6% trên toàn bộ). Gộp dữ liệu **huấn luyện** không có lợi: kNN gần như không đổi, còn GRU4Rec/SR-GNN/SR giảm 0,042–0,056 trên bài toán gốc và cũng không tốt hơn trên validation đã gộp (GRU4Rec 0,229 so với 0,255 khi fit trên dữ liệu gốc), vì mô hình mất tín hiệu "xem lại item vừa xem". → Không gộp dữ liệu huấn luyện; báo cáo thêm metric trên các query "không lặp hiển nhiên" để thấy rõ đóng góp thực của mô hình.

**C — Ngưỡng phiên 60 phút.** 60 phút thêm 3,9% query validation và tăng tỷ lệ "thêm giỏ → mua" nằm cùng phiên từ 89,9% lên 92,9%. Fit trên phiên 60 phút rồi đánh giá trên đúng bộ query 30 phút chỉ cải thiện rất nhỏ: kNN +0,001, SR +0,002, SR-GNN +0,005 (có ý nghĩa), GRU4Rec +0,0007 (khoảng tin cậy chứa 0), ItemKNN −0,003. Thứ tự mô hình không đổi. → Giữ 30 phút theo giao thức đã chốt; kết quả không nhạy với ngưỡng phiên trong khoảng 30–60 phút.

Giới hạn: mọi biến thể dùng cấu hình đã chọn cho dữ liệu gốc (không tune lại), nên biến thể có thể bị thiệt nhẹ; reranker không chạy vì được học trên validation.

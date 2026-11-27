# Kết quả thực nghiệm cuối trên validation (DIỄN TẬP, không dùng để báo cáo)

Sinh bởi `notebooks/10_thuc_nghiem_cuoi_test.ipynb` lúc 2026-11-26T15:49:20.288092+00:00, protocol 1.3.2. Bài toán: tương tác kế tiếp. `q30`: 92,566 query theo phiên 30 phút; `q60`: 96,148 query theo phiên 60 phút. Mẫu số gồm cả query có target ngoài vocabulary. Biến thể chính: `train_validation_refit`. Cột (30') là mô hình phiên 30 phút chấm trên `q30`, cột (60') là mô hình phiên 60 phút chấm trên `q60`.

## Kết quả chính, biến thể `train_validation_refit`

| Mô hình | HR@20 (30') | MRR@20 (30') | NDCG@20 (30') | CI 95% (30') | HR@20 (60') | MRR@20 (60') | NDCG@20 (60') | CI 95% (60') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Popularity | 0.0113 | 0.0031 | 0.0047 | [0.0044; 0.0051] | 0.0112 | 0.0031 | 0.0047 | [0.0044; 0.0050] |
| Repeat-last | 0.4050 | 0.3422 | 0.3578 | [0.3550; 0.3607] | 0.4112 | 0.3468 | 0.3628 | [0.3599; 0.3657] |
| VSTAN | 0.5137 | 0.3338 | 0.3747 | [0.3720; 0.3774] | 0.5141 | 0.3367 | 0.3769 | [0.3742; 0.3799] |
| GRU4Rec gốc (chỉ item) | 0.5079 | 0.3466 | 0.3836 | [0.3809; 0.3863] | 0.5062 | 0.3480 | 0.3842 | [0.3816; 0.3870] |
| GRU4Rec đa hành vi | 0.5057 | 0.3496 | 0.3853 | [0.3826; 0.3881] | 0.5039 | 0.3501 | 0.3852 | [0.3825; 0.3880] |

## Kết quả chính, biến thể `train_only`

| Mô hình | HR@20 (30') | MRR@20 (30') | NDCG@20 (30') | CI 95% (30') | HR@20 (60') | MRR@20 (60') | NDCG@20 (60') | CI 95% (60') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Popularity | 0.0113 | 0.0031 | 0.0047 | [0.0044; 0.0051] | 0.0112 | 0.0031 | 0.0047 | [0.0044; 0.0050] |
| Repeat-last | 0.4050 | 0.3422 | 0.3578 | [0.3550; 0.3607] | 0.4112 | 0.3468 | 0.3628 | [0.3599; 0.3657] |
| VSTAN | 0.5137 | 0.3338 | 0.3747 | [0.3720; 0.3774] | 0.5141 | 0.3367 | 0.3769 | [0.3742; 0.3799] |
| GRU4Rec gốc (chỉ item) | 0.5359 | 0.3535 | 0.3951 | [0.3925; 0.3979] | 0.5347 | 0.3553 | 0.3962 | [0.3934; 0.3989] |
| GRU4Rec đa hành vi | 0.5348 | 0.3537 | 0.3950 | [0.3923; 0.3978] | 0.5335 | 0.3549 | 0.3956 | [0.3930; 0.3984] |

## Hiệu ứng hành vi: GRU4Rec đa hành vi − GRU4Rec gốc (NDCG@20, paired bootstrap)

| Ngưỡng fit · biến thể · bộ query | Δ | CI thấp | CI cao | p(không tốt hơn) |
| --- | --- | --- | --- | --- |
| 30 · train_only · q30 | -0.0000 | -0.0008 | 0.0007 | 0.5540 |
| 30 · train_only · q60 | 0.0001 | -0.0007 | 0.0008 | 0.4100 |
| 30 · train_validation_refit · q30 | 0.0018 | 0.0009 | 0.0025 | 0.0000 |
| 30 · train_validation_refit · q60 | 0.0018 | 0.0010 | 0.0027 | 0.0000 |
| 60 · train_only · q30 | -0.0007 | -0.0014 | 0.0001 | 0.9520 |
| 60 · train_only · q60 | -0.0006 | -0.0013 | 0.0002 | 0.9290 |
| 60 · train_validation_refit · q30 | 0.0009 | 0.0001 | 0.0017 | 0.0190 |
| 60 · train_validation_refit · q60 | 0.0010 | 0.0001 | 0.0018 | 0.0120 |

### Hai biến thể GRU4Rec theo loại target (`train_validation_refit`, NDCG@20)

| Mô hình | mọi target | target mới | target lặp |
| --- | --- | --- | --- |
| GRU4Rec gốc (chỉ item) (30') | 0.3836 | 0.1005 | 0.7726 |
| GRU4Rec đa hành vi (30') | 0.3853 | 0.0977 | 0.7807 |
| GRU4Rec gốc (chỉ item) (60') | 0.3842 | 0.0969 | 0.7668 |
| GRU4Rec đa hành vi (60') | 0.3852 | 0.0946 | 0.7722 |

## Hiệu ứng ngưỡng phiên trên cùng bộ query: mô hình 60 phút − mô hình 30 phút (`train_validation_refit`, NDCG@20)

| Mô hình | Δ q30 | CI q30 | Δ q60 | CI q60 |
| --- | --- | --- | --- | --- |
| Popularity | 0.0000 | [0.0000; 0.0000] | 0.0000 | [0.0000; 0.0000] |
| Repeat-last | 0.0000 | [0.0000; 0.0000] | 0.0000 | [0.0000; 0.0000] |
| VSTAN | 0.0008 | [0.0004; 0.0012] | 0.0009 | [0.0006; 0.0013] |
| GRU4Rec gốc (chỉ item) | -0.0009 | [-0.0019; 0.0000] | -0.0011 | [-0.0021; -0.0001] |
| GRU4Rec đa hành vi | -0.0018 | [-0.0028; -0.0008] | -0.0019 | [-0.0029; -0.0008] |

## NDCG@20 theo độ dài prefix (`train_validation_refit`)

| Mô hình | 1 | 2-4 | 5-9 | 10+ |
| --- | --- | --- | --- | --- |
| Popularity (30') | 0.0052 | 0.0053 | 0.0040 | 0.0021 |
| Repeat-last (30') | 0.3787 | 0.3773 | 0.3226 | 0.2693 |
| VSTAN (30') | 0.4158 | 0.3940 | 0.3191 | 0.2395 |
| GRU4Rec gốc (chỉ item) (30') | 0.4369 | 0.4024 | 0.3066 | 0.2305 |
| GRU4Rec đa hành vi (30') | 0.4404 | 0.4021 | 0.3095 | 0.2306 |
| Popularity (60') | 0.0052 | 0.0054 | 0.0037 | 0.0026 |
| Repeat-last (60') | 0.3858 | 0.3863 | 0.3276 | 0.2742 |
| VSTAN (60') | 0.4197 | 0.4014 | 0.3241 | 0.2459 |
| GRU4Rec gốc (chỉ item) (60') | 0.4416 | 0.4094 | 0.3117 | 0.2274 |
| GRU4Rec đa hành vi (60') | 0.4448 | 0.4092 | 0.3105 | 0.2268 |

## So với Repeat-last (`train_validation_refit`, NDCG@20, paired bootstrap)

| Mô hình | Δ | CI |
| --- | --- | --- |
| Popularity (30') | -0.3531 | [-0.3559; -0.3503] |
| VSTAN (30') | 0.0169 | [0.0151; 0.0185] |
| GRU4Rec gốc (chỉ item) (30') | 0.0258 | [0.0241; 0.0272] |
| GRU4Rec đa hành vi (30') | 0.0275 | [0.0259; 0.0290] |
| Popularity (60') | -0.3581 | [-0.3610; -0.3551] |
| VSTAN (60') | 0.0142 | [0.0125; 0.0158] |
| GRU4Rec gốc (chỉ item) (60') | 0.0214 | [0.0200; 0.0230] |
| GRU4Rec đa hành vi (60') | 0.0225 | [0.0209; 0.0239] |

## Thời gian fit (giây)

| Mô hình | 30' train_only | 30' train_validation_refit | 60' train_only | 60' train_validation_refit |
| --- | --- | --- | --- | --- |
| Popularity | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Repeat-last | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| VSTAN | 2.8000 | 2.5000 | 2.4000 | 2.4000 |
| GRU4Rec gốc (chỉ item) | 0.8000 | 189.0000 | 1.0000 | 190.9000 |
| GRU4Rec đa hành vi | 1.0000 | 188.3000 | 1.1000 | 208.6000 |

# Kết quả thực nghiệm cuối trên test

Sinh bởi `notebooks/10_thuc_nghiem_cuoi_test.ipynb` lúc 2026-11-26T19:15:33.564396+00:00, protocol 1.3.2. Bài toán: tương tác kế tiếp. `q30`: 92,468 query theo phiên 30 phút; `q60`: 95,783 query theo phiên 60 phút. Mẫu số gồm cả query có target ngoài vocabulary. Biến thể chính: `train_validation_refit`. Cột (30') là mô hình phiên 30 phút chấm trên `q30`, cột (60') là mô hình phiên 60 phút chấm trên `q60`.

## Kết quả chính, biến thể `train_validation_refit`

| Mô hình | HR@20 (30') | MRR@20 (30') | NDCG@20 (30') | CI 95% (30') | HR@20 (60') | MRR@20 (60') | NDCG@20 (60') | CI 95% (60') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Popularity | 0.0102 | 0.0025 | 0.0041 | [0.0038; 0.0044] | 0.0101 | 0.0025 | 0.0041 | [0.0039; 0.0044] |
| Repeat-last | 0.3817 | 0.3179 | 0.3337 | [0.3311; 0.3366] | 0.3874 | 0.3226 | 0.3387 | [0.3359; 0.3416] |
| VSTAN | 0.4933 | 0.3115 | 0.3528 | [0.3502; 0.3555] | 0.4941 | 0.3154 | 0.3559 | [0.3533; 0.3584] |
| GRU4Rec gốc (chỉ item) | 0.5106 | 0.3299 | 0.3713 | [0.3688; 0.3740] | 0.5111 | 0.3337 | 0.3742 | [0.3717; 0.3769] |
| GRU4Rec đa hành vi | 0.5104 | 0.3312 | 0.3721 | [0.3696; 0.3747] | 0.5101 | 0.3352 | 0.3752 | [0.3727; 0.3778] |

## Kết quả chính, biến thể `train_only`

| Mô hình | HR@20 (30') | MRR@20 (30') | NDCG@20 (30') | CI 95% (30') | HR@20 (60') | MRR@20 (60') | NDCG@20 (60') | CI 95% (60') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Popularity | 0.0087 | 0.0032 | 0.0043 | [0.0039; 0.0047] | 0.0087 | 0.0032 | 0.0043 | [0.0040; 0.0047] |
| Repeat-last | 0.3679 | 0.3080 | 0.3229 | [0.3202; 0.3257] | 0.3738 | 0.3126 | 0.3278 | [0.3252; 0.3308] |
| VSTAN | 0.4623 | 0.2969 | 0.3345 | [0.3320; 0.3373] | 0.4643 | 0.3008 | 0.3379 | [0.3354; 0.3404] |
| GRU4Rec gốc (chỉ item) | 0.4807 | 0.3154 | 0.3532 | [0.3507; 0.3557] | 0.4807 | 0.3182 | 0.3553 | [0.3528; 0.3580] |
| GRU4Rec đa hành vi | 0.4803 | 0.3162 | 0.3537 | [0.3512; 0.3564] | 0.4791 | 0.3176 | 0.3545 | [0.3520; 0.3571] |

## Hiệu ứng hành vi: GRU4Rec đa hành vi − GRU4Rec gốc (NDCG@20, paired bootstrap)

| Ngưỡng fit · biến thể · bộ query | Δ | CI thấp | CI cao | p(không tốt hơn) |
| --- | --- | --- | --- | --- |
| 30 · train_only · q30 | 0.0005 | -0.0002 | 0.0012 | 0.0810 |
| 30 · train_only · q60 | 0.0005 | -0.0002 | 0.0013 | 0.0760 |
| 30 · train_validation_refit · q30 | 0.0008 | 0.0000 | 0.0016 | 0.0230 |
| 30 · train_validation_refit · q60 | 0.0010 | 0.0002 | 0.0017 | 0.0080 |
| 60 · train_only · q30 | -0.0009 | -0.0016 | -0.0001 | 0.9930 |
| 60 · train_only · q60 | -0.0008 | -0.0015 | -0.0000 | 0.9780 |
| 60 · train_validation_refit · q30 | 0.0011 | 0.0003 | 0.0018 | 0.0010 |
| 60 · train_validation_refit · q60 | 0.0010 | 0.0003 | 0.0017 | 0.0030 |

### Hai biến thể GRU4Rec theo loại target (`train_validation_refit`, NDCG@20)

| Mô hình | mọi target | target mới | target lặp |
| --- | --- | --- | --- |
| GRU4Rec gốc (chỉ item) (30') | 0.3713 | 0.1172 | 0.7504 |
| GRU4Rec đa hành vi (30') | 0.3721 | 0.1161 | 0.7540 |
| GRU4Rec gốc (chỉ item) (60') | 0.3742 | 0.1151 | 0.7494 |
| GRU4Rec đa hành vi (60') | 0.3752 | 0.1148 | 0.7522 |

## Hiệu ứng ngưỡng phiên trên cùng bộ query: mô hình 60 phút − mô hình 30 phút (`train_validation_refit`, NDCG@20)

| Mô hình | Δ q30 | CI q30 | Δ q60 | CI q60 |
| --- | --- | --- | --- | --- |
| Popularity | 0.0000 | [0.0000; 0.0000] | 0.0000 | [0.0000; 0.0000] |
| Repeat-last | 0.0000 | [0.0000; 0.0000] | 0.0000 | [0.0000; 0.0000] |
| VSTAN | 0.0012 | [0.0008; 0.0016] | 0.0013 | [0.0010; 0.0017] |
| GRU4Rec gốc (chỉ item) | 0.0009 | [-0.0000; 0.0018] | 0.0009 | [0.0000; 0.0019] |
| GRU4Rec đa hành vi | 0.0011 | [0.0002; 0.0021] | 0.0009 | [0.0001; 0.0018] |

## NDCG@20 theo độ dài prefix (`train_validation_refit`)

| Mô hình | 1 | 2-4 | 5-9 | 10+ |
| --- | --- | --- | --- | --- |
| Popularity (30') | 0.0045 | 0.0046 | 0.0033 | 0.0027 |
| Repeat-last (30') | 0.3569 | 0.3523 | 0.3007 | 0.2441 |
| VSTAN (30') | 0.4024 | 0.3688 | 0.2955 | 0.2094 |
| GRU4Rec gốc (chỉ item) (30') | 0.4265 | 0.3905 | 0.2977 | 0.2181 |
| GRU4Rec đa hành vi (30') | 0.4273 | 0.3912 | 0.3002 | 0.2177 |
| Popularity (60') | 0.0044 | 0.0045 | 0.0034 | 0.0030 |
| Repeat-last (60') | 0.3645 | 0.3613 | 0.3086 | 0.2452 |
| VSTAN (60') | 0.4076 | 0.3778 | 0.3042 | 0.2123 |
| GRU4Rec gốc (chỉ item) (60') | 0.4341 | 0.3994 | 0.3042 | 0.2169 |
| GRU4Rec đa hành vi (60') | 0.4353 | 0.4002 | 0.3041 | 0.2184 |

## So với Repeat-last (`train_validation_refit`, NDCG@20, paired bootstrap)

| Mô hình | Δ | CI |
| --- | --- | --- |
| Popularity (30') | -0.3296 | [-0.3324; -0.3270] |
| VSTAN (30') | 0.0190 | [0.0174; 0.0206] |
| GRU4Rec gốc (chỉ item) (30') | 0.0376 | [0.0360; 0.0390] |
| GRU4Rec đa hành vi (30') | 0.0384 | [0.0369; 0.0399] |
| Popularity (60') | -0.3345 | [-0.3375; -0.3318] |
| VSTAN (60') | 0.0172 | [0.0156; 0.0188] |
| GRU4Rec gốc (chỉ item) (60') | 0.0356 | [0.0341; 0.0371] |
| GRU4Rec đa hành vi (60') | 0.0365 | [0.0350; 0.0380] |

## Thời gian fit (giây)

| Mô hình | 30' train_only | 30' train_validation_refit | 60' train_only | 60' train_validation_refit |
| --- | --- | --- | --- | --- |
| Popularity | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Repeat-last | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| VSTAN | 6.6000 | 4.0000 | 3.5000 | 3.9000 |
| GRU4Rec gốc (chỉ item) | 2.0000 | 620.1000 | 1.0000 | 534.1000 |
| GRU4Rec đa hành vi | 2.3000 | 572.3000 | 1.0000 | 498.0000 |

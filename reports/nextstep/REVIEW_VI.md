# Rà soát bộ next-step đã chạy trên dữ liệu thật

| Tập | Query next-step | Phiên tạo query | Cặp query–item target | Query visitor ngoài train | Độ phủ cặp target trong ứng viên |
|---|---:|---:|---:|---:|---:|
| Train | 806.025 | 306.332 | 807.908 | 0,00% | 82,27% |
| Validation | 92.566 | 38.290 | 92.794 | 77,07% | 79,90% |
| Test | 92.468 | 37.864 | 92.634 | 83,60% | 79,17% |

30 kiểm thử đạt, 38 kiểm tra pipeline đạt, 20 Parquet khớp manifest. Audit độc lập 240 query / 243 cặp target đạt.

Độ phủ ứng viên không phải chỉ số mô hình. Xem [báo cáo đầy đủ](../../docs/BAO_CAO_XU_LY_DU_LIEU_VI.md).

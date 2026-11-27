# data/ — dữ liệu

Dữ liệu không được đưa lên git (quá lớn); chỉ có `processed/split_config.json`. Xem README chính để tải dữ liệu gốc.

```
data/
├── raw/                          4 tệp CSV gốc từ Kaggle — chỉ đọc, không bao giờ sửa
└── processed/                    do notebook sinh ra (Parquet)
    ├── events_clean.parquet      sự kiện đã bỏ trùng, thời gian UTC            (notebook 01)
    ├── events_sessionized.parquet  sự kiện kèm phiên, nhóm, tập (phiên 30 phút) (notebook 02)
    ├── sessions.parquet, timestamp_groups.parquet, splits/                    (notebook 02)
    ├── split_config.json         mốc chia train / validation / test           (notebook 02)
    ├── item_property_history.parquet, category_tree.parquet, item_catalog.parquet (notebook 03)
    ├── item_vocab.parquet        sản phẩm đã biết trong train                 (notebook 04)
    ├── samples/                  cặp prefix → target của từng tập, phiên 30 phút (notebook 04)
    ├── session60/                bản tương ứng cho phiên 60 phút              (notebook 09)
    └── final/                    Top-K và điểm từng query của lần chạy cuối   (notebook 10)
```

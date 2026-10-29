"""Refresh next-step documentation from a verified full-data manifest and summary."""
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    summary=json.loads((ROOT/'reports/nextstep/summary.json').read_text())
    manifest=json.loads((ROOT/'reports/nextstep/manifest.json').read_text())
    analysis=json.loads((ROOT/'reports/nextstep/analysis.json').read_text())
    evidence=json.loads((ROOT/'reports/nextstep/validation.json').read_text())
    if not (manifest.get('task')=='next_step' and manifest['checks_passed'] and evidence['full_dataset_verified']):
        raise ValueError('A verified full next-step run is required before refreshing documentation')
    for name,key in [('scripts/prepare_nextstep.py','pipeline_sha256'),('scripts/dataset.py','reader_sha256')]:
        if hashlib.file_digest((ROOT/name).open('rb'),'sha256').hexdigest()!=manifest[key]:
            raise ValueError('Pipeline/reader changed: rebuild and validate the dataset before generating reports')
    if hashlib.file_digest((ROOT/'tests/test_pipeline.py').open('rb'),'sha256').hexdigest()!=evidence['test_suite_sha256']:
        raise ValueError('Tests changed: run validate_project.py --full-data before refreshing reports')
    n=lambda x:f'{int(x):,}'.replace(',','.')
    pct=lambda x:f'{x:.2f}'.replace('.',',')+'%'
    split_rows=[]
    for split,label in [('train','Train'),('validation','Validation'),('test','Test')]:
        s=summary['split_summary'][split]
        split_rows.append(f"| {label} | {n(s['queries'])} | {n(s['sessions'])} | {n(s['nextstep_target_pairs'])} | {pct(s['cold_user_query_pct'])} | {pct(s['candidate_target_pair_coverage_pct'])} |")
    split_table='\n'.join(['| Tập | Query next-step | Phiên tạo query | Cặp query–item target | Query visitor ngoài train | Độ phủ cặp target trong ứng viên |',
                           '|---|---:|---:|---:|---:|---:|',*split_rows])
    model_status='Các mô hình Popularity, ItemKNN, Session-KNN/V-SKNN, GRU4Rec/SR-GNN, bộ đánh giá xếp hạng và demo là thiết kế giai đoạn tiếp theo; chưa huấn luyện hoặc có kết quả chất lượng mô hình.'
    intro=f'''# Hệ gợi ý RetailRocket đa hành vi — next-step theo phiên

Mục tiêu duy nhất: từ prefix của một phiên truy cập, xếp hạng Top-10/20 item sẽ xuất hiện trong **nhóm timestamp kế tiếp của cùng phiên**, với cả `view`, `addtocart` và `transaction`.

## Trạng thái đã kiểm chứng

Đã chạy pipeline schema v2 trên bốn CSV RetailRocket thật: **{n(summary['raw_events'])} sự kiện nguồn → {n(summary['clean_events'])} sự kiện sạch → {n(summary['queries'])} query next-step và {n(summary['nextstep_target_pairs'])} cặp query–item target**.

{evidence['unit_tests_passed']}/{evidence['unit_tests_passed']} kiểm thử đạt; {len(summary['checks'])}/{len(summary['checks'])} kiểm tra dữ liệu đạt; {len(manifest['artifacts'])} Parquet đã xác minh SHA-256. Kiểm tra độc lập lịch sử/nhãn/metadata trên {analysis['independent_sample_queries']} query và {analysis['independent_sample_target_pairs']} cặp target cũng đạt. Đây là kiểm chứng dữ liệu, không phải kết quả mô hình.

{split_table}

**Độ phủ ứng viên không phải Recall/NDCG của mô hình.** Ở test, {pct(summary['split_summary']['test']['candidate_target_pair_coverage_pct'])} cặp target thỏa chính sách ứng viên tại t. Vẫn giữ các target còn lại trong bộ dữ liệu và mẫu số đánh giá. {n(summary['target_pairs_repeating_prefix_item'])}/{n(summary['nextstep_target_pairs'])} cặp target là item đã xuất hiện trong prefix; không mặc định lọc item đã xem.

{model_status}

## Cài đặt và kiểm tra

Chạy từ thư mục project; Python 3.12 và dependency ghim được dùng cho lần chạy đã ghi nhận:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_repository.py
```

Trên Windows, thay `.venv/bin/python` bằng `.venv\\Scripts\\python.exe`. CSV/Parquet không lưu trong Git; clone mới cần tải dữ liệu theo [data/README.md](data/README.md), không mặc định có sẵn dữ liệu chỉ vì các báo cáo đã được commit.

## Tái tạo và xác minh dữ liệu

```bash
.venv/bin/python scripts/prepare_nextstep.py
.venv/bin/python scripts/verify_artifacts.py
.venv/bin/python scripts/audit_nextstep.py
.venv/bin/python scripts/validate_project.py --full-data --notebooks
```

Đầu ra duy nhất: `data/nextstep/`; báo cáo của lần chạy: `reports/nextstep/summary.json` và `manifest.json`. Chạy lại ghi đè các bảng do pipeline quản lý. Không nạp khi còn marker `INCOMPLETE`. Không chạy hai pipeline đồng thời vào một thư mục. Nếu dùng `--config`, truyền cùng cấu hình khi chạy verifier.

## Giao thức chính

1. UTC; loại đúng dòng trùng raw; giữ ba hành vi và các lần lặp item khác thời điểm.
2. Phiên mới khi gap >30 phút. Phân vị thời gian 80%/90%; loại toàn phiên cắt qua mốc split.
3. Gộp mọi sự kiện cùng `(session_id, timestamp)`; sau mỗi nhóm trừ nhóm cuối tạo một query.
4. Target là tập item phân biệt tại đúng nhóm kế tiếp cùng session; cho phép item lặp từ prefix.
5. Prefix chỉ thuộc cùng session, có `event_time <= query_time`. Feature không có nhãn, target_time hoặc độ dài phiên hoàn chỉnh.
6. Ứng viên có `known_at <= t`, available gần nhất là 1 hoặc unknown. Mapping/thống kê chỉ từ train. Không ép target vào candidate hoặc bỏ target khó.

Mẫu âm train mặc định tắt (`negative_samples_per_query=0`); khi bật phải loại toàn bộ tập item target. Không dùng âm train làm tập ứng viên đánh giá.

## Mã và tài liệu hiện hành

- [Pipeline](scripts/prepare_nextstep.py), [cấu hình](configs/nextstep.json), [reader](scripts/dataset.py), [verifier](scripts/verify_artifacts.py), [audit độc lập](scripts/audit_nextstep.py).
- [Giao thức](docs/DATA_PROTOCOL.md), [từ điển dữ liệu](docs/DATA_DICTIONARY.md), [báo cáo dữ liệu](docs/BAO_CAO_XU_LY_DU_LIEU_VI.md), [đề cương](docs/DE_CUONG_NEXTSTEP.md).
- [Slide PowerPoint theo template HUST](docs/Bao_cao_RetailRocket_nextstep.pptx), [báo cáo Word](docs/BAO_CAO_XU_LY_DU_LIEU_VI.docx), [đề cương Word](docs/DE_CUONG_NEXTSTEP.docx).
- [Kết quả kiểm chứng](reports/nextstep/validation.json), [review dữ liệu](reports/nextstep/REVIEW_VI.md), [phân tích prefix](reports/nextstep/analysis.json).
- Ba notebook chỉ đọc: [khảo sát raw](notebooks/01_data_audit.ipynb), [prefix/target](notebooks/02_nextstep_overview.ipynb), [xem Parquet](notebooks/03_xem_du_lieu_parquet.ipynb). Không còn pipeline notebook tạo mẫu song song.

Notebook: cài thêm `requirements-notebooks.txt` và chạy `.venv/bin/jupyter lab`. Báo cáo/slide: cài thêm `requirements-docs.txt`, xác minh bằng `validate_project.py --full-data --notebooks`, rồi chạy `build_nextstep_report.py`, `build_data_report.py`, `build_slides.py` theo thứ tự. Các lệnh tạo tài liệu dùng số liệu trong báo cáo của lần chạy, không tự huấn luyện mô hình.

Báo cáo raw trong `reports/raw_analysis/` là khảo sát nguồn đã ghi nhận; không phải số liệu nhãn hoặc chỉ số mô hình của pipeline khác. Dữ liệu nguồn tải từ Kaggle đã khớp SHA-256 raw được ghi nhận trước đây; xem [nguồn và checksum](reports/nextstep/raw_source.json).
'''
    (ROOT/'README.md').write_text(intro,encoding='utf-8')
    artifacts='\n'.join(f"| `{name}` | {n(info['rows'])} |" for name,info in manifest['artifacts'].items())
    report=f'''# Báo cáo xử lý dữ liệu RetailRocket — next-step theo phiên

Cập nhật theo lần chạy thật ngày 29/10/2026. Nguồn số liệu: `reports/nextstep/summary.json`, `manifest.json`, `analysis.json`, `validation.json`. Không có kết quả xếp hạng mô hình trong báo cáo này.

## 1. Bài toán và dữ liệu nguồn

Một phiên gồm các nhóm timestamp g1,…,gm. Mỗi query dùng prefix g1,…,gt; target là tập item phân biệt ở g(t+1), không giới hạn ở hành vi mua. Một phiên m nhóm tạo m−1 query. Phiên chỉ một nhóm không tạo mẫu có nhãn nhưng vẫn giữ trong events. Sự kiện đồng thời không được tự gán thứ tự nhân quả.

Bốn CSV nguồn có {n(summary['raw_events'])} sự kiện, {n(sum(p['raw_rows'] for p in summary['properties'].values()))} dòng thuộc tính và 1.669 dòng cây danh mục. Nguồn: bộ RetailRocket công khai trên Kaggle. Bốn SHA-256 raw đều khớp chuẩn đối chiếu đã ghi nhận; checksum cụ thể được lưu trong manifest và raw_source.json để tái lập.

## 2. Làm sạch và chia tập

Giữ view, addtocart, transaction; loại thiếu trường bắt buộc và {n(summary['removed_events'])} dòng trùng tuyệt đối, còn {n(summary['clean_events'])} sự kiện. Transaction phải có transactionid, view/giỏ được phép thiếu. UTC; session mới nếu gap >30 phút.

Train_end: {manifest['cutoffs']['train_end']}. Validation_end: {manifest['cutoffs']['validation_end']}. Mọi sự kiện của phiên cắt qua split bị gắn boundary_drop ({n(summary['events_by_split']['boundary_drop'])} sự kiện). Tỷ lệ query sau lọc không bắt buộc 80/10/10.

## 3. Query, prefix và nhãn

Tổng {n(summary['sessions'])} phiên, {n(summary['timestamp_groups'])} nhóm timestamp; {n(summary['queries'])} query và {n(summary['nextstep_target_pairs'])} cặp target. {n(summary['queries_with_multiple_targets'])} query có nhiều target. Query ID là ID nhóm xác định, không phải session ID. Target_event_count ghi số sự kiện của item ở nhóm kế, còn relevance là nhị phân theo item.

{split_table}

Có {n(analysis['single_timestamp_group_sessions'])} phiên chỉ một nhóm ({pct(analysis['single_timestamp_group_session_pct'])}). Nhóm cuối/phiên boundary nằm trong excluded_steps; không gán thành query nhãn rỗng.

Prefix lấy cùng session và không vượt query_time; không dùng phiên trước của visitor. Feature gồm số nhóm/sự kiện đã quan sát, số đếm hành vi, lần gần nhất và recency_seconds. Không đưa target_time/target_count/eligible_target_count vào feature. {n(summary['target_pairs_repeating_prefix_item'])} cặp target ({pct(summary['target_pairs_repeating_prefix_item']/summary['nextstep_target_pairs']*100)}) lặp item trong prefix, nên không loại mặc định sản phẩm đã xem.

## 4. Metadata và ứng viên

Đọc thuộc tính theo chunk, giữ {n(summary['retained_property_rows'])} bản cập nhật categoryid/available; catalog hợp log và thuộc tính có {n(summary['catalog_items'])} item. Mapping chỉ từ train: {n(summary['train_users'])} visitor và {n(summary['train_items'])} item. PAD=0, UNK=1; ID raw luôn được giữ.

Metadata dùng bản gần nhất có property_time<=query_time. Available unknown vẫn được giữ, available=0 bị loại. Item phải được biết tại t. Category thiếu/ngoài cây nhận -1 kèm cờ. Snapshot cuối train không thay được metadata từng query. Cạnh visitor–item–behavior không phải đồ thị chuyển tiếp prefix cho SR-GNN.

Độ phủ target trong ứng viên ở test là {pct(summary['split_summary']['test']['candidate_target_pair_coverage_pct'])}; trung bình tỷ lệ target có thể phục vụ theo query là {pct(summary['split_summary']['test']['macro_recall_candidate_ceiling_pct'])}. Đây là hai thống kê giới hạn dữ liệu khác nhau, chưa xét K và không phải Recall của mô hình. Target ngoài ứng viên hoặc train vẫn được giữ trong mẫu số chính.

Mẫu âm tùy chọn mặc định tắt; tệp hiện có {n(summary['negative_training_pairs'])} dòng. Khi bật, âm là item hợp lệ tại t không nằm trong toàn bộ nhóm target, không phải nhãn không thích. Không dùng mẫu âm train làm catalog đánh giá.

## 5. Kiểm chứng

{evidence['unit_tests_passed']} kiểm thử tự động đạt, {len(summary['checks'])} kiểm tra dữ liệu đạt trên toàn bộ lần chạy, {len(manifest['artifacts'])} Parquet khớp checksum raw/config/pipeline/reader. Kiểm tra độc lập trên {analysis['independent_sample_queries']} query và {analysis['independent_sample_target_pairs']} cặp target: tái tính nhóm kế, tập item đúng, đếm/recency prefix, metadata và eligibility tại t. Kiểm tra độc lập feature/metadata là lấy mẫu xác định, không phải phép tái tính độc lập mọi ô trên toàn dữ liệu.

Marker INCOMPLETE ngăn đọc sau lần chạy lỗi. dataset_info.json xác định task/schema. require_complete() không thay được verifier checksum. Thử nghiệm cố ý sửa target/Parquet đã xác nhận audit và verifier phát hiện lỗi.

## 6. Danh mục artifact

| File trong data/nextstep/ | Số dòng |
|---|---:|
{artifacts}

## 7. Giới hạn và bước tiếp theo

Visitor là mã ẩn danh; không có log hiển thị, nhãn không thích hoặc tên/ảnh sản phẩm đầy đủ. Available là trạng thái ghi nhận, không bảo đảm tồn kho trực tiếp. Prefix trong cùng phiên phụ thuộc nhau; phân tích hoặc bootstrap cần theo session/visitor khi phù hợp.

{model_status}

Bước kế tiếp: xây bộ đánh giá chung trên query/target/ứng viên này, triển khai Popularity, ItemKNN và Session-KNN; chọn một mô hình nâng cao; ablation có/không hành vi; chọn bằng validation trước khi đánh giá test. Không dùng số liệu kiểm tra dữ liệu để tuyên bố chất lượng mô hình hoặc tăng doanh thu.
'''
    (ROOT/'docs/BAO_CAO_XU_LY_DU_LIEU_VI.md').write_text(report,encoding='utf-8')
    review='# Rà soát bộ next-step đã chạy trên dữ liệu thật\n\n'+split_table+f"\n\n{evidence['unit_tests_passed']} kiểm thử đạt, {len(summary['checks'])} kiểm tra pipeline đạt, {len(manifest['artifacts'])} Parquet khớp manifest. Audit độc lập {analysis['independent_sample_queries']} query / {analysis['independent_sample_target_pairs']} cặp target đạt.\n\nĐộ phủ ứng viên không phải chỉ số mô hình. Xem [báo cáo đầy đủ](../../docs/BAO_CAO_XU_LY_DU_LIEU_VI.md).\n"
    (ROOT/'reports/nextstep/REVIEW_VI.md').write_text(review,encoding='utf-8')
    print('Updated README, data report and next-step review from the completed run.')


if __name__=='__main__':
    main()

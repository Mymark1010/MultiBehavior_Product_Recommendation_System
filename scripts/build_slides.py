"""Refresh the committed HUST-template deck from the verified next-step reports.

The committed PowerPoint is also the layout template. Media, themes, layouts and
masters are retained byte-for-byte; editable text is refreshed from reports.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
from pptx import Presentation

ROOT=Path(__file__).resolve().parents[1]
TEMPLATE=ROOT/'docs/Bao_cao_RetailRocket_nextstep.pptx'


def set_text(shape,text):
    frame=shape.text_frame
    old=frame.paragraphs[0]
    paragraph_style=deepcopy(old._p.pPr) if old._p.pPr is not None else None
    runs=list(old.runs)
    run_style=deepcopy(runs[0]._r.rPr) if runs and runs[0]._r.rPr is not None else None
    frame.clear()
    for i,line in enumerate(text.split('\n')):
        p=frame.paragraphs[0] if i==0 else frame.add_paragraph()
        if paragraph_style is not None:
            if p._p.pPr is not None:p._p.remove(p._p.pPr)
            p._p.insert(0,deepcopy(paragraph_style))
        run=p.add_run();run.text=line
        if run_style is not None:run._r.insert(0,deepcopy(run_style))


def build(output=TEMPLATE):
    s=json.loads((ROOT/'reports/nextstep/summary.json').read_text())
    m=json.loads((ROOT/'reports/nextstep/manifest.json').read_text())
    e=json.loads((ROOT/'reports/nextstep/validation.json').read_text())
    a=json.loads((ROOT/'reports/nextstep/analysis.json').read_text())
    if not (m['checks_passed'] and e['full_dataset_verified']):
        raise ValueError('Verified full next-step reports are required')
    n=lambda x:f'{int(x):,}'.replace(',','.')
    pct=lambda x:f'{x:.2f}'.replace('.',',')+'%'
    tests=e['unit_tests_passed'];checks=len(s['checks']);artifacts=len(m['artifacts'])
    snapshot=pd.read_parquet(ROOT/'data/nextstep/products_train_snapshot.parquet')
    train_snapshot=snapshot[snapshot.seen_in_train]
    category_unknown=pct(train_snapshot.category_id.eq(-1).mean()*100)
    available_unknown=pct(train_snapshot.availability_unknown.mean()*100)
    with ZipFile(TEMPLATE) as z:
        original={k:z.read(k) for k in z.namelist() if k.startswith(('ppt/media/','ppt/theme/','ppt/slideMasters/','ppt/slideLayouts/'))}
    deck=Presentation(TEMPLATE)
    def patch(page,values):
        slide=deck.slides[page-1]
        for key,value in values.items():
            shape=slide.shapes[key] if isinstance(key,int) else next(x for x in slide.shapes if x.name==key)
            set_text(shape,str(value))
    patch(1,{7:'Vũ Hải Minh  •  MSSV 20235166\nDữ liệu next-step đã hoàn tất • Phương pháp gợi ý đề xuất'})
    patch(4,{2:'Hiện trạng theo lần chạy đầy đủ trên bốn CSV thật; tách kiểm chứng dữ liệu khỏi chất lượng mô hình.',
        'body-0-0':f"Pipeline schema v2 đã tạo {n(s['queries'])} query và {n(s['nextstep_target_pairs'])} cặp target; prefix cùng phiên, target đúng nhóm timestamp kế tiếp.",
        'body-0-1':f'{tests} kiểm thử đạt; {checks} kiểm tra dữ liệu đạt; {artifacts} Parquet khớp manifest. Ba notebook đã thực thi đầy đủ trên dữ liệu thật.',
        'heading-1-0':'Bước tiếp theo',
        'body-1-0':'Bộ đánh giá xếp hạng, Popularity/ItemKNN/Session-KNN, một mô hình nâng cao và demo Top-K. Chưa có checkpoint hoặc chất lượng mô hình.'})
    patch(7,{2:'Khối dữ liệu đã chạy và xác minh trên toàn RetailRocket; khối mô hình và demo vẫn là đề xuất.',12:'20 Parquet',13:'Prefix, target, catalog'})
    patch(8,{2:'Log 03/05–18/09/2015 UTC • Bốn CSV thật đã đối chiếu checksum và xử lý đầy đủ.'})
    patch(12,{2:f"Đã chạy lại trên CSV thật: bỏ {n(s['removed_events'])} bản sao tuyệt đối, giữ ba loại hành vi và các lần lặp item.",4:n(s['raw_events']),6:n(s['removed_events']),8:n(s['clean_events'])})
    patch(14,{2:'train_tables() đã chạy trên dữ liệu thật; không dùng tổng hợp cuối train làm feature cho mọi query sớm hơn.',
        'body-0-0':f"{n(s['train_user_item_pairs'])} cặp visitor–item train. Đếm hành vi, thời điểm cuối và recency_hours tại cutoff train.",
        'body-1-0':f"{n(m['artifacts']['behavior_edges_train.parquet']['rows'])} cạnh visitor–item–behavior. Không phải đồ thị chuyển tiếp item của SR-GNN; đồ thị đó phải dựng riêng từ prefix.",
        'body-1-1':f"user_mapping: {n(s['train_users'])} visitor.\nitem_mapping: {n(s['train_items'])} item train.\nPAD=0, UNK=1; ID đã biết bắt đầu từ 2."})
    patch(15,{2:'read_properties(), build_catalog(), build_category_tree(); số lượng xác nhận từ lần chạy CSV thật.',
        'body-0-1':f"{n(s['retained_property_rows'])} bản cập nhật hợp lệ; available chỉ 0/1. Hiệu lực [property_time, valid_to), không lấp bằng dữ liệu tương lai.",
        'body-1-0':f"{n(s['catalog_items'])} item từ hợp log và thuộc tính. known_at là lần quan sát đầu; seen_in_train chỉ item có tương tác train."})
    patch(16,{2:'lookup_metadata() tra bản cập nhật gần nhất không vượt query_time; đã xác minh trên dữ liệu thật.',
        'body-1-0':f"{n(len(snapshot))} item đã biết tại train_end. Snapshot chỉ mô tả cutoff; không thay metadata từng query validation/test.",
        'body-1-1':f"Trong item train ở snapshot: {available_unknown} chưa rõ availability, {category_unknown} chưa rõ danh mục. Giữ các cờ unknown/missing/invalid."})
    patch(19,{2:'Cutoff thực của lần chạy được ghi trong manifest; tỷ lệ query sau lọc không bắt buộc 80/10/10.',
        'body-0-0':m['cutoffs']['train_end']+' UTC' if 'UTC' not in m['cutoffs']['train_end'] else m['cutoffs']['train_end'],
        'body-0-1':m['cutoffs']['validation_end']+' UTC' if 'UTC' not in m['cutoffs']['validation_end'] else m['cutoffs']['validation_end'],
        'body-1-0':f"Phiên có min_time ≤ cutoff < max_time mang boundary_drop. {n(s['events_by_split']['boundary_drop'])} sự kiện thuộc phiên này bị loại khỏi query, prefix và target.",
        'body-1-1':f"Train/validation/test có {n(s['split_summary']['train']['queries'])} / {n(s['split_summary']['validation']['queries'])} / {n(s['split_summary']['test']['queries'])} query; target luôn ở cùng session và split."})
    patch(20,{2:f"Đã tạo {n(s['queries'])} query và {n(s['nextstep_target_pairs'])} cặp target từ toàn bộ dữ liệu thật."})
    patch(21,{2:'build_prefix_features() và sample_train_negatives(); feature chỉ lấy cùng session, không dùng phiên trước.',
        'body-0-0':f"{n(s['queries'])} dòng feature: số nhóm/sự kiện prefix, đếm hành vi, last_*_time và *_recency_seconds. Không có target_time hoặc target_count.",
        'body-1-0':'Mặc định tắt (k=0), tệp âm hiện có 0 dòng. Khi bật, lấy tối đa k item train hợp lệ tại t, loại toàn bộ tập target của nhóm kế.'})
    patch(22,{0:'04 / KẾT QUẢ TRÊN DỮ LIỆU THẬT',1:'Đã tái tạo toàn bộ tập next-step RetailRocket.',2:'Số liệu từ summary và manifest của lần chạy bốn CSV thật; chưa có kết quả xếp hạng mô hình.',
        4:n(s['clean_events']),5:'sự kiện sạch',6:n(s['queries']),7:'query next-step',8:n(s['nextstep_target_pairs']),9:'cặp query–item target',10:str(artifacts),11:'Parquet đã xác minh',
        18:'Query next-step',23:'Cặp query–item target',28:'Phiên tạo query',33:f'{tests}/{tests} KIỂM THỬ ĐẠT',35:f'{artifacts}/{artifacts} TỆP KHỚP MANIFEST',36:f'{checks} kiểm tra dữ liệu đạt.'})
    for field,indices in [('queries',[19,20,21]),('nextstep_target_pairs',[24,25,26]),('sessions',[29,30,31])]:
        for split,index in zip(['train','validation','test'],indices):patch(22,{index:n(s['split_summary'][split][field])})
    patch(23,{2:'Kiểm chứng dữ liệu thật, kiểm thử trường hợp biên và audit độc lập; chưa đo chất lượng mô hình.',
        'heading-0-0':f'{checks} kiểm tra pipeline đạt',
        'body-0-0':'Kiểm tra toàn bộ query/target: đúng nhóm kế, đủ tập item, cùng session/split; mapping train; metadata không vượt t; kiểm tra prefix và mẫu âm.',
        'heading-0-1':f'{tests} kiểm thử tự động đạt',
        'body-0-1':f"Gồm nhóm đồng thời, item lặp, boundary và chạy xuyên suốt. Audit riêng {a['independent_sample_queries']} query / {a['independent_sample_target_pairs']} cặp target thật tái tính history/label/metadata.",
        'body-1-0':f'Schema v2, task=next_step; SHA-256 bốn raw, config, pipeline, reader và {artifacts} Parquet đã khớp. Đã thử sửa artifact để kiểm chứng phát hiện lỗi.',
        'body-1-1':'INCOMPLETE chặn nạp bộ chưa hoàn tất hoặc lần tái tạo bị lỗi. dataset_info.json xác định task/schema; verifier kiểm checksum.'})
    patch(24,{2:'Đầu ra duy nhất data/nextstep/; số dòng thực tế nằm trong manifest của lần chạy đã xác minh.',
        'body-1-0':f"queries: {n(s['queries'])}.\nnextstep_targets: {n(s['nextstep_target_pairs'])}.\nquery_features: {n(s['queries'])}.\nexcluded_steps: {n(m['artifacts']['excluded_steps.parquet']['rows'])}.\nnegative_samples_train: 0 (tắt)."})
    patch(25,{2:'Một pipeline và một cấu hình next-step; không còn notebook tiền xử lý tạo mẫu song song.',
        'body-0-0':'scripts/prepare_nextstep.py\nconfigs/nextstep.json\nscripts/dataset.py\nscripts/verify_artifacts.py\nscripts/audit_nextstep.py',
        'body-0-1':'reports/nextstep/summary.json\nreports/nextstep/manifest.json\nreports/nextstep/analysis.json\nreports/nextstep/validation.json',
        'body-1-1':'01_data_audit.ipynb: nguồn CSV.\n02_nextstep_overview.ipynb: prefix/target.\n03_xem_du_lieu_parquet.ipynb: xem bảng.\nCả ba notebook đã chạy thành công.'})
    patch(26,{2:f"Đã chạy trên toàn dữ liệu; {n(s['queries_with_multiple_targets'])} query có tập target gồm nhiều item."})
    patch(34,{2:'Tài liệu Word là nguồn phương pháp; trạng thái và số liệu hiện tại theo mã và manifest next-step.'})
    patch(38,{'body-0-1':f"Test: {pct(s['split_summary']['test']['cold_item_target_pct'])} cặp target ngoài train; {pct(s['split_summary']['test']['candidate_target_pair_coverage_pct'])} cặp target thuộc ứng viên tại t. Không bỏ target khó khỏi mẫu số.",
        'body-1-1':f"Phân tích theo prefix, hành vi, visitor mới/cũ và độ phổ biến item. Test có {pct(s['split_summary']['test']['cold_user_query_pct'])} query thuộc visitor chưa thấy trong train."})
    patch(41,{1:'Nền dữ liệu hoàn tất; tiếp theo là kiểm chứng mô hình.',2:'Bộ next-step đã xác minh đầy đủ; một mô hình nâng cao được lựa chọn là phạm vi nghiên cứu chính.',
        'heading-0-0':'Đã hoàn tất: dữ liệu next-step',
        'body-0-0':f"{n(s['queries'])} query, {n(s['nextstep_target_pairs'])} target pairs; {tests} kiểm thử, {checks} kiểm tra và {artifacts} artifact đạt. Ba notebook đã thực thi trên dữ liệu thật."})
    # Canonical naming on retained slides; notes always describe current evidence.
    for index,slide in enumerate(deck.slides,1):
        slide.notes_slide.notes_text_frame.text=(
            f'Trang {index}/41 — next-step theo nhóm timestamp, cập nhật 29/10/2026. '
            f'Nguồn: reports/nextstep/summary.json, manifest.json, analysis.json, validation.json. '
            f'Đã xử lý bốn CSV thật: {n(s["clean_events"])} sự kiện sạch, {n(s["queries"])} query, '
            f'{n(s["nextstep_target_pairs"])} cặp target; {tests} kiểm thử và {checks} kiểm tra dữ liệu đạt; '
            f'{artifacts} Parquet đã xác minh checksum. Audit độc lập {a["independent_sample_queries"]} query và '
            f'{a["independent_sample_target_pairs"]} cặp target; audit feature/metadata là lấy mẫu, không phải tái tính độc lập mọi ô. '
            'Prefix chỉ thuộc cùng phiên và không vượt t; target là tập item đúng nhóm kế tiếp, cả xem/giỏ/mua. '
            'Không loại mặc định item đã xem hoặc target ngoài train/ứng viên. Độ phủ ứng viên không phải Recall của mô hình. '
            'Các mô hình, bộ đánh giá ranking và demo vẫn là đề xuất; không có kết quả huấn luyện được tự tạo. '
            'Ví dụ trên các trang kịch bản/demo/tính chỉ số là minh họa, không phải kết quả mô hình. '
            'Tham khảo: Hidasi et al. ICLR 2016; Wu et al. AAAI 2019; Ludewig & Jannach UMUAI 2018; '
            'Järvelin & Kekäläinen TOIS 2002; Krichene & Rendle KDD 2020. '
            'Dữ liệu: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset .')
    deck.core_properties.title='RetailRocket next-step — dữ liệu đã xác minh và phương pháp đề xuất'
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    deck.save(output)
    with ZipFile(output) as z:parts={k:z.read(k) for k in z.namelist()}
    parts.update(original)
    with ZipFile(output,'w',ZIP_DEFLATED) as z:
        for key,data in parts.items():z.writestr(key,data)
    print('Updated 41-slide HUST-template report:',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=TEMPLATE)
    build(parser.parse_args().output)

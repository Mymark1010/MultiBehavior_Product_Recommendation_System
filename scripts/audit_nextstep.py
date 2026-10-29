"""Independently audit a completed next-step dataset and write compact statistics."""
import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from scripts.dataset import require_complete
except ModuleNotFoundError:
    from dataset import require_complete


def audit(root, sample_size=240):
    root = Path(root).resolve()
    folder = require_complete(root / 'data/nextstep')
    events = pd.read_parquet(folder / 'events.parquet')
    queries = pd.read_parquet(folder / 'queries.parquet')
    targets = pd.read_parquet(folder / 'nextstep_targets.parquet')
    features = pd.read_parquet(folder / 'query_features.parquet')
    properties = pd.read_parquet(folder / 'property_history.parquet')
    catalog = pd.read_parquet(folder / 'catalog.parquet').set_index('itemid')
    tree = pd.read_parquet(folder / 'category_tree.parquet')
    valid_categories = set(tree.categoryid)
    config = json.loads((root / 'configs/nextstep.json').read_text())
    sampled = queries.sample(min(sample_size, len(queries)), random_state=2026)
    sampled_targets = targets[targets.query_id.isin(sampled.query_id)]
    histories = {sid: frame for sid, frame in events[events.session_id.isin(sampled.session_id)].groupby('session_id')}
    property_rows = {item: frame for item, frame in properties[properties.itemid.isin(sampled_targets.itemid)].groupby('itemid')}
    label_rows = {qid: frame for qid, frame in sampled_targets.groupby('query_id')}
    feature_rows = features.set_index('query_id')
    verified_targets = 0
    for q in sampled.itertuples():
        session = histories[q.session_id]
        observed = session[session.event_time.le(q.query_time)]
        next_time = session.loc[session.event_time.gt(q.query_time), 'event_time'].min()
        expected = set(session.loc[session.event_time.eq(next_time), 'itemid'])
        t = label_rows[q.query_id]
        assert next_time == q.target_time, f'Wrong next timestamp: {q.query_id}'
        assert set(t.itemid) == expected, f'Target set mismatch: {q.query_id}'
        assert len(observed) == q.prefix_event_count
        assert observed.event_time.nunique() == q.prefix_group_count
        f = feature_rows.loc[q.query_id]
        for behavior in ('view','addtocart','transaction'):
            h = observed[observed.event.eq(behavior)]
            assert int(f[f'{behavior}_count']) == len(h)
            expected_recency = (q.query_time - h.event_time.max()).total_seconds() if len(h) else None
            actual = f[f'{behavior}_recency_seconds']
            assert pd.isna(actual) if expected_recency is None else actual == expected_recency
        for item in t.itertuples():
            rows = property_rows.get(item.itemid, properties.iloc[:0])
            observed_props = rows[rows.property_time.le(q.query_time)]
            def latest(prop):
                subset = observed_props[observed_props.property.eq(prop)]
                return None if subset.empty else int(subset.sort_values('property_time').iloc[-1].numeric_value)
            available, category = latest('available'), latest('categoryid')
            assert pd.isna(item.available) if available is None else item.available == available
            expected_category = category if category in valid_categories else config['unknown_category_id']
            assert item.category_id == expected_category
            assert item.availability_unknown == (available is None)
            assert item.category_missing == (category is None)
            known = catalog.loc[item.itemid, 'known_at']
            assert bool(item.candidate_at_query) == bool(known <= q.query_time and available != 0)
            verified_targets += 1
    session_sizes = events.groupby('session_id').event_time.nunique()
    length_stats = {}
    for split, q in queries.groupby('split'):
        length_stats[split] = {
            'prefix_groups_min':int(q.prefix_group_count.min()),
            'prefix_groups_median':float(q.prefix_group_count.median()),
            'prefix_groups_p95':float(q.prefix_group_count.quantile(.95)),
            'prefix_groups_max':int(q.prefix_group_count.max()),
            'multi_target_queries':int(q.target_count.gt(1).sum())}
    result = {'task':'next_step','scope':'full_retailrocket_dataset',
              'independent_sample_queries':len(sampled), 'independent_sample_target_pairs':verified_targets,
              'sample_checks_passed':True,
              'total_sessions':len(session_sizes),
              'single_timestamp_group_sessions':int(session_sizes.eq(1).sum()),
              'single_timestamp_group_session_pct':float(session_sizes.eq(1).mean()*100),
              'prefix_length_by_split':length_stats,
              'limitations':['All query/target sets are checked by the pipeline; this separate feature/metadata audit is a deterministic sample, not an exhaustive independent audit.',
                             'No ranking model has been trained or evaluated.']}
    output = root/'reports/nextstep/analysis.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--sample-size',type=int,default=240)
    args = parser.parse_args()
    if args.sample_size <= 0:
        parser.error('--sample-size must be positive')
    audit(args.root,args.sample_size)

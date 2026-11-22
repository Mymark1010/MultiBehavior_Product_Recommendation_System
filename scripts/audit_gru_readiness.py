"""Read-only audit of session learning signal; writes reports, never changes data."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'gru_readiness'

def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}

def distribution(values):
    return {
        'mean': float(values.mean()) if len(values) else None,
        'median': float(values.median()) if len(values) else None,
        'p90': float(values.quantile(.90)) if len(values) else None,
        'p95': float(values.quantile(.95)) if len(values) else None,
        'p99': float(values.quantile(.99)) if len(values) else None,
        'p999': float(values.quantile(.999)) if len(values) else None,
        'max': int(values.max()) if len(values) else None,
    }

def audit():
    paths = [ROOT / 'data/processed' / name for name in
             ['sessions.parquet', 'timestamp_groups.parquet', 'events_sessionized.parquet', 'split_config.json']]
    fingerprints = {str(p.relative_to(ROOT)): fingerprint(p) for p in paths}
    sessions = pd.read_parquet(paths[0])
    groups = pd.read_parquet(paths[1], columns=['session_id','group_id','group_index','n_items','n_events','split'])
    events = pd.read_parquet(paths[2], columns=['session_id','group_id','group_index','itemid','event','split'])
    assert sessions.session_id.is_unique and groups.group_id.is_unique
    assert len(events) == int(sessions.n_events.sum()) == int(groups.n_events.sum())
    assert len(groups) == int(sessions.n_groups.sum())
    assert (sessions.n_prefix_target_pairs == sessions.n_groups - 1).all()
    lengths = sessions.set_index('session_id').n_groups
    # One row per target group and distinct item, not one row per raw event.
    items = events[['session_id','group_id','group_index','itemid','split']].drop_duplicates(['group_id','itemid']).copy()
    items['session_length'] = items.session_id.map(lengths)
    first = items.groupby(['session_id','itemid'], observed=True).group_index.transform('min')
    items['seen_in_prefix'] = first < items.group_index
    targets = items[items.group_index > 1].copy()
    train_events = events[events.split == 'train']
    train_items = items[items.split == 'train']
    train_targets = targets[targets.split == 'train']
    vocab_all = set(train_events.itemid)
    vocab_sequences = set(train_items.loc[train_items.session_length >= 2, 'itemid'])
    vocab_inputs = set(train_items.loc[train_items.group_index < train_items.session_length, 'itemid'])
    vocab_targets = set(train_targets.itemid)
    result = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Audit existing splits, no model training, no metadata/availability filtering, no data mutation.',
        'query_definition': 'One prefix -> next timestamp group per session. Distinct item IDs retained in each target. Sessions need at least two groups.',
        'sources': fingerprints,
        'split_config': json.loads(paths[3].read_text(encoding='utf-8')),
        'splits': {},
        'train_vocabulary': {
            'all_train_items': len(vocab_all),
            'items_in_train_sessions_with_at_least_two_groups': len(vocab_sequences),
            'items_only_in_single_group_train_sessions': len(vocab_all-vocab_sequences),
            'items_observed_as_training_inputs': len(vocab_inputs),
            'items_observed_as_positive_training_targets': len(vocab_targets),
            'items_without_positive_target_in_train': len(vocab_all-vocab_targets),
        },
    }
    for split in ['train','validation','test','boundary_drop']:
        s = sessions[sessions.split == split]
        eligible = s[s.n_groups >= 2]
        t = targets[targets.split == split].copy()
        g = groups[(groups.split == split) & (groups.group_index > 1)]
        assert len(g) == int((eligible.n_groups-1).sum()) == t.group_id.nunique()
        actual_target_sizes = t.groupby('group_id', observed=True).size()
        assert actual_target_sizes.reindex(g.group_id).to_numpy().tolist() == g.n_items.tolist()
        unique_per_session = items[items.split == split].groupby('session_id', observed=True).itemid.nunique()
        result['splits'][split] = {
            'sessions': len(s),
            'events': int(s.n_events.sum()),
            'single_event_sessions': int((s.n_events == 1).sum()),
            'single_group_sessions': int((s.n_groups == 1).sum()),
            'eligible_sessions': len(eligible),
            'eligible_sessions_pct': 100*len(eligible)/len(s),
            'group_length_bins': {label:int(mask.sum()) for label,mask in {
                '1':s.n_groups.eq(1),'2':s.n_groups.eq(2),'3-5':s.n_groups.between(3,5),
                '6-10':s.n_groups.between(6,10),'11-20':s.n_groups.between(11,20),
                '21-50':s.n_groups.between(21,50),'51-100':s.n_groups.between(51,100),
                '>100':s.n_groups.gt(100)}.items()},
            'eligible_length_groups': distribution(eligible.n_groups),
            'queries': len(g),
            'queries_with_prefix_at_least_two_groups': int((eligible.n_groups-2).clip(lower=0).sum()),
            'queries_with_prefix_at_least_five_groups': int((eligible.n_groups-5).clip(lower=0).sum()),
            'distinct_target_item_occurrences': len(t),
            'queries_with_multiple_target_items': int(g.n_items.gt(1).sum()),
            'target_items_already_in_prefix': int(t.seen_in_prefix.sum()),
            'target_items_already_in_prefix_pct': float(t.seen_in_prefix.mean()*100),
            'eligible_sessions_with_only_one_distinct_item': int(unique_per_session.reindex(eligible.session_id).eq(1).sum()),
        }
        if split in ['validation','test']:
            cover = {}
            for label, vocab in [('all_train',vocab_all),('eligible_train_sessions',vocab_sequences),('positive_train_targets',vocab_targets)]:
                unknown = ~t.itemid.isin(vocab)
                by_query = t.assign(unknown=unknown).groupby('group_id',observed=True).unknown
                cover[label] = {
                    'unseen_target_occurrences': int(unknown.sum()),
                    'unseen_target_occurrences_pct': float(unknown.mean()*100),
                    'queries_with_any_unseen_target': int(by_query.any().sum()),
                    'queries_with_all_targets_unseen': int(by_query.all().sum()),
                    'unseen_distinct_target_items': int(t.loc[unknown,'itemid'].nunique()),
                    'mean_query_recall_ceiling_from_vocabulary_only': float(1-by_query.mean().mean()),
                }
            result['splits'][split]['target_coverage'] = cover

    train_sessions=sessions[sessions.split=='train']
    counts=train_targets.groupby('itemid',observed=True).size()
    counts_all=counts.reindex(sorted(vocab_all),fill_value=0)
    result['train_positive_target_support']={
        'counts_per_item_including_zero':distribution(counts_all),
        'items_with_zero_targets':int(counts_all.eq(0).sum()),
        'items_with_one_target':int(counts_all.eq(1).sum()),
        'items_with_1_to_5_targets':int(counts_all.between(1,5).sum()),
        'items_with_6_to_20_targets':int(counts_all.between(6,20).sum()),
        'items_with_more_than_20_targets':int(counts_all.gt(20).sum()),
        'target_occurrences_on_items_with_1_to_5_targets_pct':float(counts[counts.le(5)].sum()/counts.sum()*100),
    }
    active=train_sessions[train_sessions.n_groups.ge(2)].copy()
    q=active.n_groups-1
    top_n=max(1,int(len(active)*.01))
    durations=(active.end_ms-active.start_ms)/3600000
    result['train_long_session_concentration']={
        'sessions_over_100_groups':int(active.n_groups.gt(100).sum()),
        'queries_from_sessions_over_100_groups':int(q[active.n_groups.gt(100)].sum()),
        'queries_from_longest_one_pct_of_eligible_sessions_pct':float(q.nlargest(top_n).sum()/q.sum()*100),
        'max_session_duration_hours':float(durations.max()),
        'eligible_sessions_longer_than_24_hours':int(durations.gt(24).sum()),
    }
    # Diagnostic only: view-only sequences within existing sessions. Do not
    # silently re-sessionize or substitute these counts for the current protocol.
    view_groups=train_events[train_events.event=='view'][['session_id','group_id']].drop_duplicates()
    views=view_groups.groupby('session_id',observed=True).size()
    result['train_view_only_diagnostic']={
        'definition':'Keep only view groups inside existing sessions; no re-sessionization.',
        'sessions_with_at_least_two_view_groups':int(views.ge(2).sum()),
        'possible_next_view_group_queries':int((views-1).clip(lower=0).sum()),
    }
    for p in paths:
        assert fingerprint(p)==fingerprints[str(p.relative_to(ROOT))], 'Input changed during audit'
    result['inputs_unchanged']=True
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    audit()

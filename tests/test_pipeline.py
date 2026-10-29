"""Timestamp-group next-step invariants and end-to-end fixture validation."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.dataset import history_for_query, candidates_at_time, require_complete
from scripts.prepare_nextstep import (
    build_category_tree, build_prefix_features, clean_events, create_queries,
    create_targets, lookup_metadata, sample_train_negatives, run,
)
from scripts.verify_artifacts import verify
from scripts.audit_nextstep import audit

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configs/nextstep.json').read_text())


def timestamp(value):
    return pd.Timestamp(value, tz='UTC')


def fixture_events():
    rows = [
        (1, 10, 'view', '2020-01-01 10:00', 1, 'train'),
        (1, 10, 'addtocart', '2020-01-01 10:00', 1, 'train'),
        (1, 11, 'view', '2020-01-01 10:01', 1, 'train'),
        (1, 12, 'addtocart', '2020-01-01 10:01', 1, 'train'),
        (1, 11, 'transaction', '2020-01-01 10:01', 1, 'train'),
        (1, 10, 'view', '2020-01-01 10:02', 1, 'train'),
        (1, 90, 'view', '2020-01-02 10:00', 2, 'validation'),
        (1, 91, 'view', '2020-01-02 10:01', 2, 'validation'),
        (2, 20, 'view', '2020-01-03 10:00', 3, 'test'),
        (3, 30, 'view', '2020-01-04 10:00', 4, 'boundary_drop'),
        (3, 31, 'view', '2020-01-04 10:01', 4, 'boundary_drop'),
    ]
    e = pd.DataFrame(rows, columns=['visitorid', 'itemid', 'event', 'event_time', 'session_id', 'split'])
    e['event_time'] = pd.to_datetime(e.event_time, utc=True).astype('datetime64[ns, UTC]')
    e['event_id'] = np.arange(len(e))
    return e


def write_fixture(root):
    raw = root / 'data/raw'
    raw.mkdir(parents=True)
    rows = []
    for session in range(5):
        start = timestamp('2020-01-01') + pd.Timedelta(hours=session)
        for minute, item, behavior in [(0, 10, 'view'), (1, 11, 'view'), (1, 11, 'addtocart'), (2, 12, 'transaction')]:
            rows.append((int((start + pd.Timedelta(minutes=minute)).timestamp()*1000),
                         session+1, behavior, item+session*10,
                         session+100 if behavior == 'transaction' else None))
    events = pd.DataFrame(rows, columns=['timestamp', 'visitorid', 'event', 'itemid', 'transactionid'])
    pd.concat([events, events.iloc[[0]]]).to_csv(raw/'events.csv', index=False)
    baseline = int((timestamp('2019-12-31')).timestamp()*1000)
    props = [(baseline, item, prop, value) for item in sorted(events.itemid.unique())
             for prop, value in [('available', 1), ('categoryid', 1)]]
    pd.DataFrame(props[::2], columns=['timestamp','itemid','property','value']).to_csv(raw/'item_properties_part1.csv', index=False)
    pd.DataFrame(props[1::2], columns=['timestamp','itemid','property','value']).to_csv(raw/'item_properties_part2.csv', index=False)
    pd.DataFrame({'categoryid':[1], 'parentid':[None]}).to_csv(raw/'category_tree.csv', index=False)
    config = dict(CONFIG, train_quantile=.6, validation_quantile=.8, negative_samples_per_query=2)
    (root/'configs').mkdir()
    (root/'configs/nextstep.json').write_text(json.dumps(config))
    return root/'configs/nextstep.json'


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.events = fixture_events()
        self.q, self.excluded = create_queries(self.events)
        self.q, self.targets = create_targets(self.q, self.events)

    def test_only_exact_duplicates_removed_and_behaviors_preserved(self):
        raw = pd.DataFrame({'timestamp':[1000,1000,1000,2000,2000], 'visitorid':[1]*5,
                            'event':['view','view','addtocart','transaction','view'],
                            'itemid':[10]*5, 'transactionid':[None,None,None,5,None]})
        clean, _ = clean_events(raw, CONFIG)
        self.assertEqual(len(clean), 4)
        self.assertEqual(clean.event.value_counts().to_dict(), {'view':2,'addtocart':1,'transaction':1})

    def test_each_non_terminal_group_has_one_query(self):
        self.assertEqual(len(self.q), 3)
        self.assertEqual(self.q.groupby('session_id').size().to_dict(), {1:2,2:1})
        self.assertEqual(self.q.query_id.nunique(), 3)

    def test_single_group_sessions_and_terminal_groups_are_excluded(self):
        single = self.excluded[self.excluded.session_id.eq(3)]
        self.assertEqual(single.exclusion_reason.tolist(), ['no_next_group'])
        self.assertEqual(len(self.excluded[self.excluded.exclusion_reason.eq('no_next_group')]), 3)

    def test_boundary_sessions_never_create_queries_or_history(self):
        self.assertNotIn(4, self.q.session_id.tolist())
        self.assertTrue(self.excluded.loc[self.excluded.session_id.eq(4), 'exclusion_reason'].eq('boundary_session').all())
        query = {'session_id':4, 'query_time':timestamp('2020-01-04 10:01')}
        self.assertTrue(history_for_query(self.events, query).empty)

    def test_targets_are_immediate_group_and_all_behavior_types(self):
        first = self.q.iloc[0]
        t = self.targets[self.targets.query_id.eq(first.query_id)]
        self.assertEqual(set(t.itemid), {11,12})
        self.assertEqual(t.target_time.unique().tolist(), [timestamp('2020-01-01 10:01')])
        self.assertNotIn(10, t.itemid.tolist())

    def test_same_item_in_next_group_is_one_target_with_event_count(self):
        t = self.targets[self.targets.query_id.eq(self.q.iloc[0].query_id)].set_index('itemid')
        self.assertEqual(t.loc[11,'target_event_count'], 2)
        self.assertEqual(self.q.iloc[0].target_count, 2)

    def test_repeat_item_is_a_valid_target(self):
        t = self.targets[self.targets.query_id.eq(self.q.iloc[1].query_id)]
        self.assertEqual(t.itemid.tolist(), [10])

    def test_targets_never_cross_sessions_of_same_visitor(self):
        t = self.targets[self.targets.query_id.eq(self.q.iloc[1].query_id)]
        self.assertNotIn(90, t.itemid.tolist())
        self.assertNotIn(91, t.itemid.tolist())

    def test_prefix_observes_entire_simultaneous_group(self):
        h = history_for_query(self.events, self.q.iloc[0])
        self.assertEqual(h.event.tolist(), ['view','addtocart'])
        self.assertEqual(len(history_for_query(self.events, self.q.iloc[1])), 5)

    def test_prefix_is_session_local_even_for_returning_visitor(self):
        q = self.q[self.q.session_id.eq(2)].iloc[0]
        self.assertEqual(history_for_query(self.events, q).itemid.tolist(), [90])
        f = build_prefix_features(self.q, self.events).set_index('query_id').loc[q.query_id]
        self.assertEqual(f.view_count, 1)
        self.assertEqual(f.addtocart_count, 0)
        self.assertEqual(f.transaction_count, 0)

    def test_prefix_features_include_counts_lengths_and_recency(self):
        f = build_prefix_features(self.q, self.events).iloc[1]
        self.assertEqual((f.view_count,f.addtocart_count,f.transaction_count), (2,2,1))
        self.assertEqual((f.prefix_group_count,f.prefix_event_count), (2,5))
        self.assertEqual(f.view_recency_seconds, 0)

    def test_absent_behavior_recency_is_missing_not_zero(self):
        f = build_prefix_features(self.q, self.events).iloc[0]
        self.assertTrue(pd.isna(f.transaction_recency_seconds))
        self.assertFalse(f.has_transaction_history)

    def test_feature_table_does_not_expose_label_columns(self):
        f = build_prefix_features(self.q, self.events)
        self.assertTrue({'target_count','target_time','eligible_target_count','split'}.isdisjoint(f.columns))

    def test_future_events_cannot_change_existing_prefix_features(self):
        before = build_prefix_features(self.q, self.events)
        extra = self.events.iloc[[0]].copy()
        extra['event_time'] = timestamp('2020-01-01 10:03')
        extra['event'] = 'transaction'
        extra['event_id'] = 999
        after = build_prefix_features(self.q, pd.concat([self.events,extra]))
        pd.testing.assert_frame_equal(before,after)

    def test_permuting_simultaneous_events_preserves_queries_targets_features(self):
        shuffled = self.events.sample(frac=1, random_state=7)
        q,_ = create_queries(shuffled)
        q,t = create_targets(q,shuffled)
        pd.testing.assert_frame_equal(q,self.q)
        pd.testing.assert_frame_equal(t,self.targets)
        pd.testing.assert_frame_equal(build_prefix_features(q,shuffled), build_prefix_features(self.q,self.events))

    def test_empty_query_set_is_supported_by_helpers(self):
        e = self.events[self.events.session_id.eq(3)]
        q,_ = create_queries(e)
        q,t = create_targets(q,e)
        self.assertTrue(q.empty and t.empty and build_prefix_features(q,e).empty)

    def test_category_cycle_and_missing_parent_rejected(self):
        for parents in ([2,1],[99,None]):
            with self.assertRaises(ValueError):
                build_category_tree(pd.DataFrame({'categoryid':[1,2],'parentid':parents}))

    def test_asof_metadata_unknown_and_future_catalog_policy(self):
        tree = build_category_tree(pd.DataFrame({'categoryid':[1],'parentid':[None]}))
        props = pd.DataFrame({'itemid':[10,10,10], 'property':['available','available','categoryid'],
                              'property_time':[timestamp('2020-01-01'),timestamp('2020-01-05'),timestamp('2020-01-01')],
                              'numeric_value':[1,0,999]})
        catalog = pd.DataFrame({'itemid':[10,11,12], 'known_at':[timestamp('2020-01-01'),timestamp('2020-01-01'),timestamp('2020-01-10')]})
        req = pd.DataFrame({'itemid':[10,10,11,12], 'query_time':[timestamp('2020-01-02'),timestamp('2020-01-06'),timestamp('2020-01-02'),timestamp('2020-01-02')]})
        f = lookup_metadata(req,props,catalog,tree,CONFIG)
        self.assertEqual(f.candidate_at_query.tolist(), [True,False,True,False])
        self.assertTrue(f.loc[0,'category_invalid'])
        self.assertEqual(f.loc[0,'category_id'], -1)
        self.assertEqual(f.loc[0,'available'], 1)

    def negative_fixture(self):
        q = pd.DataFrame({'query_id':[1,2], 'query_time':[timestamp('2020-01-02')]*2, 'split':['train','test']})
        t = pd.DataFrame({'query_id':[1,1,2], 'itemid':[10,11,10]})
        items = pd.DataFrame({'itemid':[10,11,12,13,14]})
        props = pd.DataFrame({'itemid':[10], 'property':['available'], 'property_time':[timestamp('2020-01-01')], 'numeric_value':[1]})
        catalog = items.assign(known_at=[timestamp('2020-01-01')]*4+[timestamp('2020-01-10')])
        tree = build_category_tree(pd.DataFrame({'categoryid':[1], 'parentid':[None]}))
        return q,t,items,props,catalog,tree

    def test_negatives_exclude_entire_target_set_and_are_reproducible(self):
        args = self.negative_fixture()
        c = dict(CONFIG,negative_samples_per_query=2)
        first = sample_train_negatives(*args,c)
        self.assertEqual(set(first.itemid), {12,13})
        pd.testing.assert_frame_equal(first,sample_train_negatives(*args,c))

    def test_negatives_are_train_only_and_cannot_invent_future_items(self):
        n = sample_train_negatives(*self.negative_fixture(),dict(CONFIG,negative_samples_per_query=5))
        self.assertEqual(n.query_id.unique().tolist(), [1])
        self.assertNotIn(14,n.itemid.tolist())
        self.assertEqual(len(n),2)

    def test_optional_negative_sampling_can_be_disabled(self):
        result = sample_train_negatives(*self.negative_fixture(),dict(CONFIG,negative_samples_per_query=0))
        self.assertTrue(result.empty)
        self.assertEqual(str(result.query_id.dtype), 'int64')
        self.assertEqual(str(result.label.dtype), 'int8')

    def test_missing_raw_columns_report_schema_error(self):
        with self.assertRaisesRegex(ValueError, 'Missing event columns'):
            clean_events(pd.DataFrame({'timestamp':[0]}), CONFIG)

    def test_cleaning_rejects_source_without_valid_events(self):
        raw = pd.DataFrame({'timestamp':[0], 'visitorid':[1], 'event':['invalid'],
                            'itemid':[10], 'transactionid':[None]})
        with self.assertRaisesRegex(ValueError, 'No valid events'):
            clean_events(raw, CONFIG)

    def test_transaction_without_id_is_rejected_early(self):
        raw = pd.DataFrame({'timestamp':[0], 'visitorid':[1], 'event':['transaction'],
                            'itemid':[10], 'transactionid':[None]})
        with self.assertRaisesRegex(ValueError, 'require transactionid'):
            clean_events(raw, CONFIG)

    def test_real_cutoff_through_session_excludes_entire_session(self):
        raw = pd.DataFrame({'timestamp':np.arange(9)*1000, 'visitorid':np.repeat([1,2,3],3),
                            'event':['view']*9, 'itemid':np.arange(9), 'transactionid':[None]*9})
        e,_ = clean_events(raw, dict(CONFIG,train_quantile=.5,validation_quantile=.8))
        self.assertTrue(e.loc[e.visitorid.isin([2,3]), 'split'].eq('boundary_drop').all())
        q,_ = create_queries(e)
        self.assertEqual(q.visitorid.unique().tolist(),[1])

    def test_failed_regeneration_blocks_reading_previous_successful_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            config=write_fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                run(root,config)
                path=root/'data/raw/item_properties_part1.csv'
                props=pd.read_csv(path)
                props.loc[0,'value']=2
                props.to_csv(path,index=False)
                with self.assertRaisesRegex(ValueError,'Availability'):
                    run(root,config)
            with self.assertRaisesRegex(RuntimeError,'incomplete'):
                require_complete(root/'data/nextstep')

    def test_session_gap_threshold_and_temporal_boundary(self):
        raw = pd.DataFrame({'timestamp':[0,30*60*1000,61*60*1000], 'visitorid':[1]*3,
                            'event':['view']*3,'itemid':[1,2,3],'transactionid':[None]*3})
        e,_ = clean_events(raw,CONFIG)
        self.assertEqual(e.session_id.tolist(),[1,1,2])
        self.assertTrue(e.loc[e.session_id.eq(1),'split'].eq('train').all())

    def test_end_to_end_write_verify_read_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = write_fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                summary,manifest = run(root,config)
                verify(root)
                alternate = root/'alternate_config.json'
                alternate.write_bytes(config.read_bytes())
                verify(root, alternate)
                # Verify repeated installation does not change query/target content.
                first = pd.read_parquet(root/'data/nextstep/nextstep_targets.parquet')
                run(root,config)
                verify(root)
            pd.testing.assert_frame_equal(first,pd.read_parquet(root/'data/nextstep/nextstep_targets.parquet'))
            self.assertEqual(manifest['task'],'next_step')
            self.assertTrue(all(summary['checks'].values()))
            self.assertEqual(len(manifest['artifacts']),20)
            self.assertEqual({k:v['queries'] for k,v in summary['split_summary'].items()}, {'train':6,'validation':2,'test':2})
            output = root/'data/nextstep'
            require_complete(output)
            q = pd.read_parquet(output/'queries.parquet').iloc[0]
            e = pd.read_parquet(output/'events.parquet')
            self.assertEqual(len(history_for_query(e,q)),1)
            self.assertFalse(candidates_at_time(output,q.query_time,CONFIG).empty)
            changed = pd.read_parquet(output/'nextstep_targets.parquet')
            changed.loc[0,'itemid'] = 99999
            changed.to_parquet(output/'nextstep_targets.parquet',index=False)
            with self.assertRaisesRegex(AssertionError,'Artifact changed'):
                verify(root)

    def test_reader_refuses_missing_or_incomplete_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            with self.assertRaisesRegex(RuntimeError,'schema-v2'):
                require_complete(p)
            (p/'INCOMPLETE').write_text('running')
            with self.assertRaisesRegex(RuntimeError,'incomplete'):
                require_complete(p)

    def test_independent_audit_recomputes_labels_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            config=write_fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                run(root,config)
                result=audit(root,sample_size=10)
                self.assertEqual(result['independent_sample_queries'],10)
                self.assertTrue(result['sample_checks_passed'])
                path=root/'data/nextstep/nextstep_targets.parquet'
                labels=pd.read_parquet(path)
                labels.loc[0,'itemid']=99999
                labels.to_parquet(path,index=False)
                with self.assertRaisesRegex(AssertionError,'Target set mismatch'):
                    audit(root,sample_size=10)


if __name__ == '__main__':
    unittest.main()

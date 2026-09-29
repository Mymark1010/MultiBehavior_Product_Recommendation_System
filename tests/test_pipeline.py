"""Small adversarial fixtures; no Retailrocket download needed for CI."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_multibehavior import (
    build_category_tree, build_user_features, clean_events, create_queries,
    create_targets, lookup_metadata, sample_train_negatives,
)

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/multibehavior.json").read_text())


def timestamp(value):
    return pd.Timestamp(value, tz="UTC")


def fixture_events():
    rows = [
        (1, 10, "view", "2020-01-01 10:00", 1),
        (1, 10, "addtocart", "2020-01-01 10:00", 1),
        (1, 10, "transaction", "2020-01-01 10:01", 1),
        (1, 11, "transaction", "2020-01-03 11:00", 2),
        (1, 11, "transaction", "2020-01-03 11:01", 2),
        (2, 12, "view", "2020-01-02 10:00", 3),
        (1, 15, "view", "2020-01-29 10:00", 4),
    ]
    e = pd.DataFrame(rows, columns=["visitorid", "itemid", "event", "event_time", "session_id"])
    e["event_time"] = pd.to_datetime(e.event_time, utc=True).astype("datetime64[ns, UTC]")
    e["event_id"] = np.arange(len(e))
    e["split"] = "train"
    return e.sort_values(["visitorid", "event_time", "event_id"])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(CONFIG)
        self.events = fixture_events()
        self.cutoffs = {"train_end": timestamp("2020-02-01"), "validation_end": timestamp("2020-02-15"), "observation_end": timestamp("2020-03-01")}

    def test_distinct_behaviors_preserved_and_only_exact_duplicates_removed(self):
        raw = pd.DataFrame({"timestamp": [1000, 1000, 1000, 2000, 2000], "visitorid": [1]*5,
                            "event": ["view", "view", "addtocart", "transaction", "view"],
                            "itemid": [10]*5, "transactionid": [np.nan, np.nan, np.nan, 5, np.nan]})
        clean, _ = clean_events(raw, self.config)
        self.assertEqual(len(clean), 4)
        self.assertEqual(clean.event.value_counts().to_dict(), {"view": 2, "addtocart": 1, "transaction": 1})

    def test_incomplete_horizon_excluded_but_zero_purchase_queries_retained(self):
        q, excluded = create_queries(self.events, self.cutoffs, self.config)
        q, targets = create_targets(q, self.events)
        self.assertIn(4, excluded.query_id.tolist())
        self.assertIn(3, q.query_id.tolist())
        self.assertFalse(q.set_index("query_id").loc[3, "has_purchase_target"])

    def test_target_window_strictly_future_and_deduplicates_item(self):
        q, _ = create_queries(self.events, self.cutoffs, self.config)
        q, targets = create_targets(q, self.events)
        self.assertEqual(set(targets[targets.query_id.eq(1)].itemid), {10, 11})
        # Purchase at the query timestamp is already observed, not a future target.
        self.assertEqual(targets[targets.query_id.eq(2)].purchase_event_count.sum(), 1)
        self.assertEqual(q.set_index("query_id").loc[1, "target_count"], 2)

    def test_history_observes_timestamp_bucket_without_future_purchase(self):
        q, _ = create_queries(self.events, self.cutoffs, self.config)
        q, _ = create_targets(q, self.events)
        features = build_user_features(q, self.events, self.config).set_index("query_id")
        first = features.loc[1]
        self.assertEqual(first.view_count, 1)
        self.assertEqual(first.addtocart_count, 1)
        self.assertEqual(first.transaction_count, 0)
        self.assertTrue(pd.isna(first.transaction_recency_hours))
        self.assertEqual(features.loc[2].transaction_count, 2)

    def test_future_events_cannot_change_existing_features(self):
        q, _ = create_queries(self.events, self.cutoffs, self.config)
        before = build_user_features(q, self.events, self.config)
        extra = self.events.iloc[[0]].copy()
        extra["event_time"] = timestamp("2020-02-01")
        extra["event"] = "transaction"
        extra["event_id"] = 100
        after = build_user_features(q, pd.concat([self.events, extra]), self.config)
        pd.testing.assert_frame_equal(before, after)

    def test_feature_table_does_not_contain_labels(self):
        q, _ = create_queries(self.events, self.cutoffs, self.config)
        q, _ = create_targets(q, self.events)
        features = build_user_features(q, self.events, self.config)
        self.assertTrue({"target_count", "has_purchase_target", "label_end", "split"}.isdisjoint(features.columns))

    def test_rolling_window_excludes_left_boundary(self):
        q = pd.DataFrame({"query_id": [99], "visitorid": [1], "query_time": [timestamp("2020-01-08 10:00")]})
        f = build_user_features(q, self.events, self.config).iloc[0]
        self.assertEqual(f.view_count, 1)
        self.assertEqual(f.view_count_7d, 0)
        self.assertEqual(f.transaction_count_7d, 3)

    def test_category_cycle_and_missing_parent_rejected(self):
        for parents in ([2, 1], [99, None]):
            with self.assertRaises(ValueError):
                build_category_tree(pd.DataFrame({"categoryid": [1, 2], "parentid": parents}))

    def test_asof_metadata_and_unknown_candidate_policy(self):
        tree = build_category_tree(pd.DataFrame({"categoryid": [1], "parentid": [None]}))
        props = pd.DataFrame({"itemid": [10, 10, 10], "property": ["available", "available", "categoryid"],
                              "property_time": [timestamp("2020-01-01"), timestamp("2020-01-05"), timestamp("2020-01-01")],
                              "numeric_value": [1, 0, 999]})
        catalog = pd.DataFrame({"itemid": [10, 11, 12], "known_at": [timestamp("2020-01-01"), timestamp("2020-01-01"), timestamp("2020-01-10")]})
        requests = pd.DataFrame({"itemid": [10, 10, 11, 12], "query_time": [timestamp("2020-01-02"), timestamp("2020-01-06"), timestamp("2020-01-02"), timestamp("2020-01-02")]})
        f = lookup_metadata(requests, props, catalog, tree, self.config)
        self.assertEqual(f.candidate_at_query.tolist(), [True, False, True, False])
        self.assertTrue(f.loc[0, "category_invalid"])
        self.assertEqual(f.loc[0, "category_id"], -1)
        self.assertEqual(f.loc[0, "available"], 1)

    def test_negative_sampling_excludes_targets_and_is_reproducible(self):
        config = dict(self.config, negative_samples_per_positive_query=2)
        q = pd.DataFrame({"query_id": [1], "query_time": [timestamp("2020-01-02")], "split": ["train"], "has_purchase_target": [True]})
        target = pd.DataFrame({"query_id": [1], "itemid": [10]})
        items = pd.DataFrame({"itemid": [10, 11, 12]})
        props = pd.DataFrame({"itemid": [10, 10], "property": ["available", "categoryid"],
                              "property_time": [timestamp("2020-01-01")]*2, "numeric_value": [1, 1]})
        catalog = items.assign(known_at=timestamp("2020-01-01"))
        tree = build_category_tree(pd.DataFrame({"categoryid": [1], "parentid": [None]}))
        first = sample_train_negatives(q, target, items, props, catalog, tree, config)
        second = sample_train_negatives(q, target, items, props, catalog, tree, config)
        self.assertEqual(set(first.itemid), {11, 12})
        pd.testing.assert_frame_equal(first, second)

    def test_negative_sampling_does_not_invent_future_candidates(self):
        config = dict(self.config, negative_samples_per_positive_query=2)
        q = pd.DataFrame({"query_id": [1], "query_time": [timestamp("2020-01-02")], "split": ["train"], "has_purchase_target": [True]})
        target = pd.DataFrame({"query_id": [1], "itemid": [10]})
        items = pd.DataFrame({"itemid": [10, 11, 12]})
        props = pd.DataFrame({"itemid": [10, 10], "property": ["available", "categoryid"],
                              "property_time": [timestamp("2020-01-01")]*2, "numeric_value": [1, 1]})
        catalog = items.assign(known_at=[timestamp("2020-01-01"), timestamp("2020-01-01"), timestamp("2020-01-10")])
        tree = build_category_tree(pd.DataFrame({"categoryid": [1], "parentid": [None]}))
        negatives = sample_train_negatives(q, target, items, props, catalog, tree, config)
        self.assertEqual(negatives.itemid.tolist(), [11])


if __name__ == "__main__":
    unittest.main()

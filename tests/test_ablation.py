import numpy as np
import pandas as pd
import pytest

from rr.ablation import (assign_session_splits, build_samples, flag_heavy_visitors, heavy_visitor_thresholds,
                         known_at_for, merge_repeated_groups, sessionize, topk_matrix)

MIN = 60_000


def clean(rows):
    """rows: (visitorid, timestamp, itemid, event)."""
    v, t, i, e = zip(*rows)
    return pd.DataFrame({"event_id": np.arange(len(rows)), "timestamp": np.array(t, dtype=np.int64),
                         "visitorid": np.array(v, dtype=np.int64), "itemid": np.array(i, dtype=np.int64),
                         "event": pd.Categorical(e, categories=["view", "addtocart", "transaction"])})


def test_sessionize_threshold_groups_and_split():
    ev = clean([(1, 0, 10, "view"), (1, 0, 11, "view"), (1, 30 * MIN, 12, "view"),   # gap = 30 min: same session
                (1, 61 * MIN, 13, "view"),                                           # gap 31 min: new at 30, same at 60
                (2, 0, 10, "view"), (2, 200 * MIN, 10, "view")])
    s30 = sessionize(ev, 30 * MIN, (100 * MIN, 150 * MIN)).sort_values("event_id")
    assert s30["session_id"].tolist() == [1, 1, 1, 2, 3, 4]
    assert s30["group_index"].tolist() == [1, 1, 2, 1, 1, 1]  # same timestamp -> one group
    assert s30["split"].astype(str).tolist() == ["train", "train", "train", "train", "train", "test"]
    s60 = sessionize(ev, 60 * MIN, (100 * MIN, 150 * MIN)).sort_values("event_id")
    assert s60["session_id"].tolist() == [1, 1, 1, 1, 2, 3]
    assert s60["group_index"].tolist() == [1, 1, 2, 3, 1, 1]
    # a longer threshold can make a session cross a cutoff
    s_long = sessionize(ev, 300 * MIN, (100 * MIN, 150 * MIN)).sort_values("event_id")
    assert s_long["split"].astype(str).tolist()[-2:] == ["boundary_drop", "boundary_drop"]


def test_split_rules_match_notebook_02_examples():
    start = [0, 50, 100, 101, 190, 200, 201, 0]
    end = [100, 100, 101, 200, 201, 200, 230, 230]
    assert assign_session_splits(start, end, 100, 200).tolist() == [
        "train", "train", "boundary_drop", "validation", "boundary_drop", "validation", "test", "boundary_drop"]


def test_merge_repeated_groups_drops_only_consecutive_identical_sets():
    # g1 {A view}, g2 {A view} (dup), g3 {B view}, g4 {A view}, g5 {A view, B cart}, g6 {B cart, A view} (dup, other order)
    ev = clean([(1, 0, 1, "view"), (1, MIN, 1, "view"), (1, 2 * MIN, 2, "view"), (1, 3 * MIN, 1, "view"),
                (1, 4 * MIN, 1, "view"), (1, 4 * MIN, 2, "addtocart"),
                (1, 5 * MIN, 2, "addtocart"), (1, 5 * MIN, 1, "view"),
                (2, 0, 1, "view"), (2, MIN, 1, "addtocart")])          # same item, other behavior: kept
    s = sessionize(ev, 30 * MIN, (10 * MIN, 20 * MIN))
    merged, stats = merge_repeated_groups(s)
    one = merged[merged["session_id"] == 1].sort_values("event_id")
    assert one["timestamp"].drop_duplicates().tolist() == [0, 2 * MIN, 3 * MIN, 4 * MIN]
    assert one["group_index"].drop_duplicates().tolist() == [1, 2, 3, 4]
    assert len(merged[merged["session_id"] == 2]) == 2
    assert stats["groups_dropped"] == 2 and stats["events_dropped"] == 3
    q, t = build_samples(merged, "train")
    assert len(q) == 3 + 1
    first = t[t["query_id"] == q["query_id"].iloc[0]]
    assert first["itemid"].tolist() == [2] and not first["target_is_repeat"].any()


def test_heavy_visitors_use_daily_counts():
    day = 86_400_000
    train = clean([(1, k, 5, "view") for k in range(10)] + [(2, 0, 5, "view"), (2, day, 5, "view")]
                  + [(3, 0, 5, "transaction"), (3, 1, 6, "transaction")])
    thr = heavy_visitor_thresholds(train, 0.5)
    # visitor-days: events [10, 1, 1, 2] -> median 1.5; transactions [0, 0, 0, 2] -> median 0
    assert thr == {"quantile": 0.5, "events_per_day": 1.5, "transactions_per_day": 0.0}
    other = clean([(7, 0, 5, "view"), (8, 0, 5, "view"), (8, 1, 5, "view"), (9, 0, 5, "transaction")])
    assert flag_heavy_visitors(other, thr).tolist() == [8, 9]
    # the same two events on two different days are not heavy
    assert flag_heavy_visitors(clean([(8, 0, 5, "view"), (8, day, 5, "view")]), thr).tolist() == []


def test_known_at_uses_fit_events_and_first_property():
    catalog = pd.DataFrame({"itemid": [1, 2, 3], "first_property_ms": pd.array([50, pd.NA, 5], dtype="Int64")})
    fit = pd.DataFrame({"itemid": [1, 2, 2], "timestamp": [100, 30, 20]})
    out = known_at_for(catalog, fit).set_index("itemid")["known_at_fit_ms"]
    assert out.tolist() == [50, 20, 5]
    assert str(out.dtype) == "Int64"


def test_topk_matrix_aligns_and_pads():
    long = pd.DataFrame({"query_id": [9, 9, 4], "rank": [1, 2, 1], "itemid": [70, 71, 72]})
    m = topk_matrix(long, [4, 9, 5], k=3)
    assert m.tolist() == [[72, -1, -1], [70, 71, -1], [-1, -1, -1]]
    with pytest.raises(ValueError):
        topk_matrix(long, [4], k=3)

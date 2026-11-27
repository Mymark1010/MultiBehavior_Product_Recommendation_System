import pandas as pd

from rr.samples import (UNK_IDX, asof_property, build_queries, build_targets, build_vocab, encode,
                        get_prefix, prefix_events)


def toy():
    # Session 1: g1 = view A, g2 = view B, g3 = addtocart B + view C (same timestamp).
    # Session 2: single group -> no query.
    events = pd.DataFrame({
        "event_id": [0, 1, 2, 3, 4],
        "session_id": [1, 1, 1, 1, 2],
        "visitorid": [7, 7, 7, 7, 8],
        "group_id": [10, 11, 12, 12, 13],
        "group_index": [1, 2, 3, 3, 1],
        "timestamp": [100, 200, 300, 300, 50],
        "itemid": [65, 66, 66, 67, 65],
        "event": pd.Categorical(["view", "view", "addtocart", "view", "view"],
                                categories=["view", "addtocart", "transaction"]),
    })
    groups = events.drop_duplicates("group_id")[["group_id", "session_id", "visitorid", "timestamp", "group_index"]]
    return events, groups


def test_queries_and_targets_match_readme_example():
    events, groups = toy()
    q = build_queries(groups, "train", first_query_id=5)
    assert q["query_id"].tolist() == [5, 6]
    assert q["prefix_len_groups"].tolist() == [1, 2]
    assert q["query_time_ms"].tolist() == [100, 200] and q["target_time_ms"].tolist() == [200, 300]
    assert (q["target_time_ms"] > q["query_time_ms"]).all()
    t = build_targets(q, events)
    second = t[t["query_id"] == 6].set_index("itemid")
    assert set(second.index) == {66, 67}  # target {B, C}
    assert second.loc[66, "behaviors"] == ["addtocart"] and bool(second.loc[66, "target_is_repeat"])
    assert second.loc[67, "behaviors"] == ["view"] and not bool(second.loc[67, "target_is_repeat"])
    first = t[t["query_id"] == 5]
    assert first["itemid"].tolist() == [66] and not first["target_is_repeat"].any()


def test_get_prefix_excludes_target_group():
    events, groups = toy()
    q = build_queries(groups, "train").iloc[1]
    prefix = get_prefix(events, q)
    assert prefix["group_id"].tolist() == [10, 11]
    assert not prefix["group_id"].isin([q["target_group_id"]]).any()
    pe = prefix_events(build_queries(groups, "train"), events)
    assert pe.groupby("query_id").size().tolist() == [1, 2]
    assert pe[pe["query_id"] == 1].sort_values("group_index")["distance"].tolist() == [1, 0]


def test_vocab_and_encode():
    vocab = build_vocab(pd.DataFrame({"itemid": [30, 10, 10, 20]}))
    assert vocab["itemid"].tolist() == [10, 20, 30] and vocab["item_idx"].tolist() == [2, 3, 4]
    assert encode([20, 99], vocab).tolist() == [3, UNK_IDX]


def test_asof_never_uses_future():
    history = pd.DataFrame({"itemid": [1, 1, 2], "timestamp": [10, 20, 30],
                            "property": ["available", "available", "available"], "value": ["1", "0", "1"]})
    keys = pd.DataFrame({"itemid": [1, 1, 1, 2], "query_time_ms": [5, 10, 19, 29]}, index=[3, 1, 2, 0])
    out = asof_property(keys, history, "available")
    assert out.loc[[3, 1, 2, 0], "available_asof"].tolist() == [pd.NA, "1", "1", pd.NA]
    known = out["available_asof_ms"].notna()
    assert (out.loc[known, "available_asof_ms"] <= keys.loc[known, "query_time_ms"]).all()


def test_prefix_events_max_groups_and_chunks():
    events, groups = toy()
    q = build_queries(groups, "train")
    full = prefix_events(q, events)
    chunked = prefix_events(q, events, chunk=1)
    pd.testing.assert_frame_equal(full.sort_values(["query_id", "group_index"]).reset_index(drop=True),
                                  chunked.sort_values(["query_id", "group_index"]).reset_index(drop=True))
    last = prefix_events(q, events, max_groups=1)
    assert last.groupby("query_id")["distance"].max().tolist() == [0, 0]
    assert last[last["query_id"] == 1]["group_index"].tolist() == [2]

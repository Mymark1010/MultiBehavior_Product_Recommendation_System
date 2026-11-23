import numpy as np
import pandas as pd

from rr.rerank import add_features, candidate_table, feature_columns, label, ranked_lists, session_folds


def test_candidates_features_and_labels():
    queries = pd.DataFrame({"query_id": [10, 11], "session_id": [1, 2], "prefix_len_groups": [2, 1],
                            "query_time_ms": [100, 100]})
    prefix = pd.DataFrame({"query_id": [10, 10, 11], "itemid": [5, 6, 7], "event": ["view", "addtocart", "view"],
                           "distance": [1, 0, 0]})
    sources = {"knn": (np.array([[8, 5, -1], [9, -1, -1]]), np.array([[0.9, 0.5, np.nan], [0.3, np.nan, np.nan]])),
               "gru": (np.array([[5, 9, 8], [7, 8, -1]]), np.array([[2.0, 1.0, 0.5], [3.0, 1.0, np.nan]]))}
    t = candidate_table(queries["query_id"].to_numpy(), sources, prefix, top_n=3)
    q10 = t[t["query_id"] == 10].set_index("itemid")
    assert sorted(q10.index) == [5, 6, 8, 9]                 # union + prefix item 6
    assert q10.loc[5, "knn_rank"] == 2 and q10.loc[5, "gru_rank"] == 1 and q10.loc[5, "n_lists"] == 2
    assert q10.loc[6, "knn_rank"] == 4 and q10.loc[6, "n_lists"] == 0  # missing -> top_n + 1
    fit_events = pd.DataFrame({"itemid": [5, 5, 8], "event": ["view", "transaction", "view"]})
    history = pd.DataFrame({"itemid": [5, 6, 8, 9], "timestamp": [0, 0, 0, 500], "property": "categoryid",
                            "value": ["1", "1", "2", "1"]})
    f = add_features(t, queries, prefix, fit_events, history).set_index(["query_id", "itemid"])
    assert f.loc[(10, 6), "prefix_addtocart"] == 1 and f.loc[(10, 6), "in_last_group"] == 1
    assert f.loc[(10, 5), "prefix_last_distance"] == 1 and f.loc[(10, 8), "in_prefix"] == 0
    assert f.loc[(10, 5), "pop_transaction"] == np.log1p(1)
    assert f.loc[(10, 5), "same_category_as_last"] == 1      # cat 1 == cat of item 6 (last group)
    assert f.loc[(10, 8), "same_category_as_last"] == 0
    assert np.isnan(f.loc[(10, 9), "same_category_as_last"])  # category of 9 only known after t
    assert len(feature_columns(t, sources)) == 4 + 14
    y = label(t, pd.DataFrame({"query_id": [10, 11], "itemid": [8, 123]}))
    assert y.sum() == 1 and t.loc[y == 1, "itemid"].tolist() == [8]
    items, _ = ranked_lists(t, -t["itemid"].to_numpy(dtype=float), [11, 10], width=3)
    assert items[1].tolist() == [5, 6, 8] and items[0].tolist()[:2] == [7, 8]


def test_session_folds_keep_sessions_together():
    q = pd.DataFrame({"session_id": [1, 1, 2, 3, 3, 3]})
    f = session_folds(q, 2, seed=0)
    assert f[0] == f[1] and f[3] == f[4] == f[5]

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from rr.models import ItemKNN, SessionKNN, popularity, prefix_matrix, rank_long, repeat_last, topn_csr


def ev(rows):
    """rows: (session_id, group_index, itemid, event, timestamp)."""
    return pd.DataFrame(rows, columns=["session_id", "group_index", "itemid", "event", "timestamp"])


def pre(rows):
    """rows: (query_id, itemid, event, distance)."""
    return pd.DataFrame(rows, columns=["query_id", "itemid", "event", "distance"])


def test_rank_long_and_topn_csr():
    items, scores = rank_long([0, 0, 0, 1], [5, 3, 4, 9], [1.0, 2.0, 2.0, 0.5], 3, 2)
    assert items.tolist() == [[3, 4], [9, -1], [-1, -1]]  # tie 3/4 -> lower itemid first
    m = sp.csr_matrix(np.array([[0, 3, 1, 3], [2, 0, 0, 0]], dtype=float))
    assert topn_csr(m, 2).toarray().tolist() == [[0, 3, 0, 3], [2, 0, 0, 0]]


def test_popularity_weights():
    e = ev([(1, 1, 10, "view", 0), (1, 2, 10, "view", 1), (2, 1, 20, "addtocart", 2)])
    assert popularity(e, {"view": 1, "addtocart": 1, "transaction": 1})[0].tolist() == [10, 20]
    assert popularity(e, {"view": 1, "addtocart": 5, "transaction": 1})[0].tolist() == [20, 10]


def test_repeat_last_order():
    p = pre([(7, 1, "view", 1), (7, 2, "view", 0), (7, 3, "addtocart", 0), (7, 1, "view", 2), (8, 4, "view", 0)])
    items, _ = repeat_last(p, [7, 8, 9], width=4)
    assert items.tolist() == [[3, 2, 1, -1], [4, -1, -1, -1], [-1, -1, -1, -1]]
    assert prefix_matrix(p, [7, 8]).tolist() == [[2, 3, 1], [4, -1, -1]]


def test_itemknn_cosine_by_hand():
    e = ev([(1, 1, 1, "view", 0), (1, 2, 2, "view", 1), (2, 1, 1, "view", 2), (2, 1, 2, "view", 2),
            (3, 1, 1, "view", 3), (3, 2, 3, "view", 4)])
    model = ItemKNN(n_neighbors=10).fit(e)
    items, scores = model.recommend(pre([(0, 1, "view", 0)]), [0], width=3)
    assert items[0].tolist() == [2, 3, -1]
    np.testing.assert_allclose(scores[0, :2], [2 / np.sqrt(6), 1 / np.sqrt(3)])
    # n_neighbors=1 keeps only the best neighbour of item 1
    items, _ = ItemKNN(n_neighbors=1).fit(e).recommend(pre([(0, 1, "view", 0)]), [0], width=3)
    assert items[0].tolist() == [2, -1, -1]


def test_sknn_by_hand():
    e = ev([(1, 1, 1, "view", 0), (1, 2, 2, "view", 1),
            (2, 1, 1, "view", 2), (2, 2, 3, "view", 3), (2, 3, 4, "view", 4),
            (3, 1, 5, "view", 5), (3, 2, 6, "view", 6),
            (4, 1, 7, "view", 7)])  # single-item session not stored
    model = SessionKNN("sknn", k=10, sample_size=10).fit(e)
    assert len(model.sessions) == 3
    items, scores = model.recommend(pre([(0, 1, "view", 0)]), [0], [100], width=5)
    assert items[0].tolist() == [1, 2, 3, 4, -1]
    np.testing.assert_allclose(scores[0, :4], [1 / np.sqrt(2) + 1 / np.sqrt(3), 1 / np.sqrt(2),
                                               1 / np.sqrt(3), 1 / np.sqrt(3)])
    items, _ = SessionKNN("sknn", k=1, sample_size=10).fit(e).recommend(pre([(0, 1, "view", 0)]), [0], [100], width=5)
    assert items[0].tolist() == [1, 2, -1, -1, -1]
    # sample_size=1 keeps only the most recent session containing item 1 (session 2)
    items, _ = SessionKNN("sknn", k=10, sample_size=1).fit(e).recommend(pre([(0, 1, "view", 0)]), [0], [100], width=5)
    assert items[0].tolist() == [1, 3, 4, -1, -1]


def test_stan_position_factor_prefers_items_near_match():
    e = ev([(1, 1, 1, "view", 0), (1, 2, 2, "view", 1), (1, 3, 3, "view", 2), (1, 4, 4, "view", 3)])
    model = SessionKNN("stan", k=10, sample_size=10, lambda3=0.5).fit(e)
    items, _ = model.recommend(pre([(0, 3, "view", 0)]), [0], [100], width=4)
    assert items[0, 0] == 3 and set(items[0, 1:3]) == {2, 4} and items[0, 3] == 1


def test_behavior_weights_change_item_scores():
    e = ev([(1, 1, 1, "view", 0), (1, 2, 2, "view", 1), (2, 1, 1, "view", 2), (2, 2, 3, "transaction", 3),
            (2, 3, 9, "view", 4)])
    uniform = SessionKNN("sknn", k=10, sample_size=10).fit(e)
    weighted = SessionKNN("sknn", k=10, sample_size=10,
                          weights={"view": 1, "addtocart": 1, "transaction": 10}).fit(e)
    q = pre([(0, 1, "view", 0)])
    assert uniform.recommend(q, [0], [100], width=4)[0][0, 1] == 2
    assert weighted.recommend(q, [0], [100], width=4)[0][0, 0] == 3


def test_unknown_prefix_items_are_ignored():
    e = ev([(1, 1, 1, "view", 0), (1, 2, 2, "view", 1)])
    items, _ = SessionKNN("stan", k=5, sample_size=5).fit(e).recommend(pre([(0, 99, "view", 0)]), [0], [10], width=3)
    assert (items == -1).all()
    for removed in ["other", "vsknn"]:  # V-SKNN removed in protocol v1.3.2
        with pytest.raises(ValueError):
            SessionKNN(removed)


def test_vstan_idf_downweights_ubiquitous_items():
    rows = [(s, 1, 1, "view", s) for s in range(1, 6)]
    rows += [(s, 2, 2, "view", s) for s in range(1, 6)]      # item 2 in every session
    rows += [(1, 3, 3, "view", 1)]                            # item 3 only in session 1
    e = ev(rows)
    q = pre([(0, 1, "view", 0)])
    stan = SessionKNN("stan", k=10, sample_size=10, lambda2_ms=1e15).fit(e)
    vstan = SessionKNN("vstan", k=10, sample_size=10, lambda2_ms=1e15, lambda_idf=1.0).fit(e)
    s_items, _ = stan.recommend(q, [0], [100], width=3)
    v_items, v_scores = vstan.recommend(q, [0], [100], width=3)
    assert s_items[0, 1] == 2           # STAN: popular item 2 beats rare item 3
    assert v_items[0, 0] == 3           # VSTAN: idf(2) = log(5/5) ~ 0 pushes item 2 down

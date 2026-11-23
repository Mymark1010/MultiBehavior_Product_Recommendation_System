import math

import numpy as np
import pandas as pd
import pytest

from rr.metrics import bootstrap_ci, evaluate, per_query_metrics, prefix_bin


def run(topk, targets, ks=(3,)):
    """targets: list of target sets, one per query; query ids 0..n-1."""
    rows = [(q, item) for q, items in enumerate(targets) for item in items]
    frame = pd.DataFrame(rows, columns=["query_id", "itemid"])
    return per_query_metrics(np.array(topk), np.arange(len(targets)), frame, ks)


def d(rank):
    return 1 / math.log2(rank + 1)


def test_no_hit():
    m = run([[1, 2, 3]], [{9}]).iloc[0]
    assert m["hit@3"] == 0 and m["recall@3"] == 0 and m["mrr@3"] == 0 and m["ndcg@3"] == 0


def test_hit_at_rank_one():
    m = run([[5, 2, 3]], [{5}]).iloc[0]
    assert m["hit@3"] == 1 and m["recall@3"] == 1 and m["mrr@3"] == 1 and m["ndcg@3"] == pytest.approx(1)


def test_hit_at_rank_k():
    m = run([[1, 2, 7]], [{7}]).iloc[0]
    assert m["hit@3"] == 1
    assert m["mrr@3"] == pytest.approx(1 / 3)
    assert m["ndcg@3"] == pytest.approx(d(3) / d(1))  # 0.5


def test_multiple_targets():
    # G = {2, 8, 9}; hits at rank 2 only; IDCG uses min(|G|, K) = 3 positions.
    m = run([[1, 2, 3]], [{2, 8, 9}]).iloc[0]
    assert m["hit@3"] == 1
    assert m["recall@3"] == pytest.approx(1 / 3)
    assert m["mrr@3"] == pytest.approx(1 / 2)
    assert m["ndcg@3"] == pytest.approx(d(2) / (d(1) + d(2) + d(3)))


def test_multiple_targets_more_than_k():
    # |G| = 4 > K = 2: IDCG has only K terms; two hits = perfect NDCG, recall 2/4.
    m = run([[1, 2]], [{1, 2, 3, 4}], ks=(2,)).iloc[0]
    assert m["ndcg@2"] == pytest.approx(1) and m["recall@2"] == pytest.approx(0.5)


def test_duplicate_item_rejected():
    with pytest.raises(ValueError, match="trùng"):
        run([[1, 1, 3]], [{1}])


def test_k_larger_than_candidates():
    # Only two candidates for K = 3: padding is a miss, list is still valid.
    m = run([[4, -1, -1]], [{4, 5}]).iloc[0]
    assert m["hit@3"] == 1 and m["recall@3"] == pytest.approx(0.5)
    assert m["ndcg@3"] == pytest.approx(d(1) / (d(1) + d(2)))
    # Shorter matrix than K is padded automatically.
    m2 = run([[4]], [{4, 5}]).iloc[0]
    assert m2["ndcg@3"] == pytest.approx(m["ndcg@3"])


def test_padding_must_be_trailing():
    with pytest.raises(ValueError, match="Padding"):
        run([[-1, 2, 3]], [{2}])


def test_oov_target_counts_in_denominator():
    # Query 1 has only an out-of-vocabulary target: it scores 0 but stays in the mean.
    per = run([[5, 2, 3], [1, 2, 3]], [{5}, {999999}])
    assert per["hit@3"].mean() == pytest.approx(0.5)


def test_empty_target_rejected():
    frame = pd.DataFrame({"query_id": [0], "itemid": [1]})
    with pytest.raises(ValueError, match="không rỗng"):
        per_query_metrics(np.array([[1], [2]]), np.array([0, 1]), frame, ks=(1,))


def test_evaluate_splits_new_and_repeat():
    queries = pd.DataFrame({"query_id": [10, 11], "prefix_len_groups": [1, 5]})
    targets = pd.DataFrame({"query_id": [10, 10, 11], "itemid": [1, 2, 3],
                            "target_is_repeat": [True, False, False]})
    report, per = evaluate(np.array([[1, 9], [3, 9]]), queries, targets, ks=(2,))
    assert report["repeat_targets"]["hit@2"] == 1 and report["repeat_targets"]["n_queries"] == 1
    assert report["new_targets"]["hit@2"] == pytest.approx(0.5) and report["new_targets"]["n_queries"] == 2
    assert set(report["by_prefix_len"]) == {"1", "5-9"}


def test_prefix_bins():
    assert prefix_bin([1, 2, 4, 5, 9, 10, 300]).tolist() == ["1", "2-4", "2-4", "5-9", "5-9", "10+", "10+"]


def test_bootstrap_constant():
    ci = bootstrap_ci(np.ones(50), n_resamples=20)
    assert ci["low"] == ci["high"] == 1

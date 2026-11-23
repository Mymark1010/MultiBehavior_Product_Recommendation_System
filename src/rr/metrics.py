"""Top-K ranking metrics for next-step recommendation with set-valued targets.

For one query with a de-duplicated Top-K list and target set G (original item IDs):

* hit@K    = 1 if |TopK ∩ G| > 0
* recall@K = |TopK ∩ G| / |G|
* mrr@K    = 1 / rank of the first hit, 0 without a hit
* ndcg@K   = DCG / IDCG, DCG = Σ_{hit at rank r} 1/log2(r+1), IDCG = Σ_{r=1}^{min(|G|,K)} 1/log2(r+1)

Averages are over every query with non-empty G, including queries whose targets are
outside the vocabulary (they score 0). Lists shorter than K are padded with -1 (miss).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAD_ITEM = -1
KS = (10, 20)
METRICS = ("hit", "recall", "mrr", "ndcg")
PREFIX_BINS = [(1, 1, "1"), (2, 4, "2-4"), (5, 9, "5-9"), (10, None, "10+")]


def validate_topk(topk: np.ndarray) -> None:
    """Reject duplicated items in a row and non-trailing padding."""
    topk = np.asarray(topk)
    if topk.ndim != 2:
        raise ValueError("topk phải là mảng 2 chiều [query, rank].")
    valid = topk != PAD_ITEM
    if (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("Padding -1 chỉ được nằm ở cuối danh sách.")
    ordered = np.sort(np.where(valid, topk, np.iinfo(np.int64).min), axis=1)
    duplicate = (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] != np.iinfo(np.int64).min)
    if duplicate.any():
        raise ValueError("Top-K có item trùng trong cùng một query.")


def per_query_metrics(topk: np.ndarray, query_ids, targets: pd.DataFrame, ks=KS) -> pd.DataFrame:
    """Metrics per query.

    topk:      int64 array [n_queries, Kmax] of original item IDs, -1 padding at the end.
    query_ids: query id for each row of ``topk``.
    targets:   long table with columns ``query_id, itemid`` (the full target set).
    """
    topk = np.asarray(topk, dtype=np.int64)
    validate_topk(topk)
    query_ids = np.asarray(query_ids, dtype=np.int64)
    if len(query_ids) != len(topk):
        raise ValueError("Số dòng topk khác số query.")
    if len(np.unique(query_ids)) != len(query_ids):
        raise ValueError("query_id trùng.")
    if max(ks) > topk.shape[1]:
        pad = np.full((len(topk), max(ks) - topk.shape[1]), PAD_ITEM, dtype=np.int64)
        topk = np.hstack([topk, pad])

    row_of = pd.Series(np.arange(len(query_ids)), index=query_ids)
    pairs = targets[["query_id", "itemid"]].drop_duplicates()
    rows = row_of.reindex(pairs["query_id"].to_numpy()).to_numpy()
    if np.isnan(rows.astype(float)).any():
        raise ValueError("targets chứa query không có trong topk.")
    rows = rows.astype(np.int64)
    n_targets = np.bincount(rows, minlength=len(query_ids))
    if (n_targets == 0).any():
        raise ValueError("Mọi query phải có tập target không rỗng.")

    width = int(max(pairs["itemid"].max(), topk.max(), 0)) + 2
    target_keys = rows * width + pairs["itemid"].to_numpy(dtype=np.int64)
    topk_keys = np.arange(len(topk), dtype=np.int64)[:, None] * width + topk
    hits = np.isin(topk_keys, target_keys) & (topk != PAD_ITEM)

    out = {"query_id": query_ids, "n_targets": n_targets}
    for k in ks:
        hk = hits[:, :k]
        n_hit = hk.sum(axis=1)
        first = np.where(n_hit > 0, hk.argmax(axis=1) + 1, 0)
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = (hk * discounts).sum(axis=1)
        idcg = np.cumsum(discounts)[np.minimum(n_targets, k) - 1]
        out[f"hit@{k}"] = (n_hit > 0).astype(float)
        out[f"recall@{k}"] = n_hit / n_targets
        out[f"mrr@{k}"] = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
        out[f"ndcg@{k}"] = dcg / idcg
    return pd.DataFrame(out)


def metric_columns(ks=KS):
    return [f"{m}@{k}" for k in ks for m in METRICS]


def summarize(per_query: pd.DataFrame, ks=KS) -> dict:
    cols = metric_columns(ks)
    result = {c: float(per_query[c].mean()) for c in cols}
    result["n_queries"] = int(len(per_query))
    return result


def prefix_bin(prefix_len) -> pd.Series:
    prefix_len = pd.Series(prefix_len)
    labels = pd.Series("", index=prefix_len.index, dtype=object)
    for low, high, label in PREFIX_BINS:
        mask = prefix_len.ge(low) & (prefix_len.le(high) if high is not None else True)
        labels[mask] = label
    return labels


def evaluate(topk, queries: pd.DataFrame, targets: pd.DataFrame, ks=KS) -> dict:
    """Full report: all targets / new-only / repeat-only, and by prefix-length bin.

    queries: rows aligned with ``topk``; needs ``query_id, prefix_len_groups``.
    targets: needs ``query_id, itemid, target_is_repeat``.
    """
    topk = np.asarray(topk)
    query_ids = queries["query_id"].to_numpy()
    base = per_query_metrics(topk, query_ids, targets, ks)
    base["prefix_bin"] = prefix_bin(queries["prefix_len_groups"].to_numpy()).to_numpy()
    report = {"all": summarize(base, ks), "by_prefix_len": {}}
    for _, _, label in PREFIX_BINS:
        part = base[base["prefix_bin"] == label]
        if len(part):
            report["by_prefix_len"][label] = summarize(part, ks)
    for name, flag in [("new_targets", False), ("repeat_targets", True)]:
        subset = targets[targets["target_is_repeat"] == flag]
        keep = np.isin(query_ids, subset["query_id"].unique())
        if keep.any():
            report[name] = summarize(per_query_metrics(topk[keep], query_ids[keep], subset, ks), ks)
    return report, base


def bootstrap_ci(values, n_resamples: int = 1000, seed: int = 0, level: float = 0.95) -> dict:
    """Percentile bootstrap CI of the mean, resampling queries."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = values[rng.integers(0, len(values), len(values))].mean()
    alpha = (1 - level) / 2
    return {"mean": float(values.mean()), "low": float(np.quantile(means, alpha)),
            "high": float(np.quantile(means, 1 - alpha)), "n_resamples": n_resamples, "seed": seed}

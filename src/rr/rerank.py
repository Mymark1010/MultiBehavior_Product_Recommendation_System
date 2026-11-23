"""HySAR-style reranker (protocol v1.2.0 -> methods.reranker).

Candidates of a query = union of the top-N lists of the base models (best kNN, GRU4Rec) and the
prefix items. A LightGBM lambdarank model reorders them using:

* base model scores and ranks (missing -> score NaN, rank = N + 1), and in how many lists the item is;
* prefix behavior features of the item: counts of view / addtocart / transaction events, distance
  (in groups) of its most recent occurrence, whether it is in the last group;
* item popularity in the fitting data, overall and per behavior (log1p counts);
* query context: prefix length in groups, distinct prefix items;
* category at t: whether the item's category as of query time equals a category of the
  last-group items (unknown -> NaN), using only property updates with timestamp <= t.

Labels: 1 if the item is in the query's target set (task 1). The reranked list then goes through
the same candidate filter and Popularity fallback as every other model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rr.models import event_codes
from rr.samples import BEHAVIORS, asof_property

FEATURES_BASE = ["n_lists", "in_prefix", "prefix_view", "prefix_addtocart", "prefix_transaction",
                 "prefix_last_distance", "in_last_group", "pop_all", "pop_view", "pop_addtocart",
                 "pop_transaction", "prefix_len_groups", "prefix_items", "same_category_as_last"]


def candidate_table(query_ids, sources: dict, prefix: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """Long table (query_id, itemid, <name>_score, <name>_rank, ...) for all candidate sources.

    sources: name -> (items [n, W], scores [n, W]) aligned with ``query_ids`` (-1 padding).
    Prefix items are always added as candidates.
    """
    query_ids = np.asarray(query_ids)
    parts = []
    for name, (items, scores) in sources.items():
        items, scores = items[:, :top_n], scores[:, :top_n]
        rows, cols = np.nonzero(items != -1)
        parts.append(pd.DataFrame({"query_id": query_ids[rows], "itemid": items[rows, cols],
                                   f"{name}_score": scores[rows, cols], f"{name}_rank": cols + 1}))
    pre = prefix[["query_id", "itemid"]].drop_duplicates()
    table = pre
    for part in parts:
        table = table.merge(part, on=["query_id", "itemid"], how="outer")
    for name in sources:
        table[f"{name}_rank"] = table[f"{name}_rank"].fillna(top_n + 1)
    table["n_lists"] = sum(table[f"{name}_rank"].le(top_n).astype(int) for name in sources)
    return table.sort_values(["query_id", "itemid"]).reset_index(drop=True)


def add_features(table: pd.DataFrame, queries: pd.DataFrame, prefix: pd.DataFrame, fit_events: pd.DataFrame,
                 history: pd.DataFrame) -> pd.DataFrame:
    t = table
    # prefix behavior features
    p = prefix[["query_id", "itemid", "distance"]].copy()
    code = event_codes(prefix["event"])
    for k, b in enumerate(BEHAVIORS):
        p[f"prefix_{b}"] = (code == k).astype(np.int32)
    agg = p.groupby(["query_id", "itemid"]).agg(prefix_view=("prefix_view", "sum"), prefix_addtocart=("prefix_addtocart", "sum"),
                                                prefix_transaction=("prefix_transaction", "sum"),
                                                prefix_last_distance=("distance", "min")).reset_index()
    t = t.merge(agg, on=["query_id", "itemid"], how="left")
    t["in_prefix"] = t["prefix_last_distance"].notna().astype(int)
    t["in_last_group"] = t["prefix_last_distance"].eq(0).astype(int)
    for b in BEHAVIORS:
        t[f"prefix_{b}"] = t[f"prefix_{b}"].fillna(0)
    t["prefix_last_distance"] = t["prefix_last_distance"].fillna(999)
    # popularity in the fitting data
    fe = pd.DataFrame({"itemid": fit_events["itemid"].to_numpy(), "code": event_codes(fit_events["event"])})
    pop = pd.DataFrame({"pop_all": fe.groupby("itemid").size()})
    for k, b in enumerate(BEHAVIORS):
        pop[f"pop_{b}"] = fe[fe["code"] == k].groupby("itemid").size()
    pop = np.log1p(pop.fillna(0))
    t = t.join(pop, on="itemid")
    for c in pop.columns:
        t[c] = t[c].fillna(0)
    # query context
    ctx = queries[["query_id", "prefix_len_groups", "query_time_ms"]].copy()
    ctx["prefix_items"] = ctx["query_id"].map(prefix.groupby("query_id")["itemid"].nunique()).fillna(0)
    t = t.merge(ctx, on="query_id", how="left")
    # category of the candidate vs categories of the last group, both as of t
    cand_cat = asof_property(t[["itemid", "query_time_ms"]], history, "categoryid")["categoryid_asof"]
    last = prefix[prefix["distance"] == 0][["query_id", "itemid"]].drop_duplicates()
    last = last.merge(ctx[["query_id", "query_time_ms"]], on="query_id")
    last["cat"] = asof_property(last[["itemid", "query_time_ms"]], history, "categoryid")["categoryid_asof"].to_numpy()
    last_cats = last.dropna(subset=["cat"])[["query_id", "cat"]].drop_duplicates().assign(hit=1)
    key = pd.DataFrame({"query_id": t["query_id"].to_numpy(), "cat": cand_cat.to_numpy()})
    same = key.merge(last_cats, on=["query_id", "cat"], how="left")["hit"].to_numpy()
    known = cand_cat.notna().to_numpy() & t["query_id"].isin(last_cats["query_id"]).to_numpy()
    t["same_category_as_last"] = np.where(known, np.nan_to_num(same, nan=0.0), np.nan)
    return t.drop(columns=["query_time_ms"])


def label(table: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray:
    key = targets[["query_id", "itemid"]].drop_duplicates().assign(y=1)
    return table[["query_id", "itemid"]].merge(key, on=["query_id", "itemid"], how="left")["y"].fillna(0).to_numpy(dtype=int)


def feature_columns(table: pd.DataFrame, sources) -> list:
    return [c for name in sources for c in (f"{name}_score", f"{name}_rank")] + FEATURES_BASE


def ranked_lists(table: pd.DataFrame, scores: np.ndarray, query_ids, width: int = 200):
    """Order candidates by reranker score -> (items [n, width], scores [n, width]) aligned with query_ids."""
    from rr.models import rank_long
    row_of = pd.Series(np.arange(len(query_ids)), index=np.asarray(query_ids))
    rows = row_of.reindex(table["query_id"].to_numpy()).to_numpy()
    return rank_long(rows, table["itemid"].to_numpy(), scores, len(query_ids), width)


def session_folds(queries: pd.DataFrame, n_folds: int = 2, seed: int = 0) -> np.ndarray:
    """Fold id per query, all queries of one session in the same fold."""
    sessions = queries["session_id"].unique()
    rng = np.random.default_rng(seed)
    fold_of = pd.Series(rng.permutation(len(sessions)) % n_folds, index=sessions)
    return fold_of.reindex(queries["session_id"].to_numpy()).to_numpy()

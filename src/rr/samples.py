"""Prefix–target samples, vocabulary and as-of metadata lookup.

A query is the end of prefix (g1..gt) of a session with >= 2 timestamp groups; its target is
the set of distinct items of g(t+1). Prefixes are not materialised: :func:`get_prefix` reads
them back from the sessionized events by ``session_id`` and ``group_index <= prefix_len_groups``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BEHAVIORS = ["view", "addtocart", "transaction"]
PAD_IDX, UNK_IDX, FIRST_IDX = 0, 1, 2


def build_vocab(events: pd.DataFrame) -> pd.DataFrame:
    """D5: every item of the given events, sorted by itemid, item_idx from 2 (PAD=0, UNK=1)."""
    items = np.sort(events["itemid"].unique()).astype(np.int64)
    return pd.DataFrame({"itemid": items, "item_idx": np.arange(FIRST_IDX, FIRST_IDX + len(items), dtype=np.int64)})


def encode(itemids, vocab: pd.DataFrame) -> np.ndarray:
    """Map itemid -> item_idx; unknown items -> UNK_IDX."""
    lookup = pd.Series(vocab["item_idx"].to_numpy(), index=vocab["itemid"].to_numpy())
    return lookup.reindex(np.asarray(itemids)).fillna(UNK_IDX).to_numpy(dtype=np.int64)


def build_queries(groups: pd.DataFrame, split: str, first_query_id: int = 0) -> pd.DataFrame:
    """One query per group that has a following group in the same session."""
    g = groups.sort_values(["session_id", "group_index"]).reset_index(drop=True)
    same_next = g["session_id"].eq(g["session_id"].shift(-1))
    next_group = g["group_id"].shift(-1)
    next_index = g["group_index"].shift(-1)
    if not (next_index[same_next] == g.loc[same_next, "group_index"] + 1).all():
        raise ValueError("group_index không liên tiếp trong phiên.")
    q = g.loc[same_next, ["session_id", "visitorid", "timestamp", "group_index"]].copy()
    q["target_group_id"] = next_group[same_next].astype(np.int64)
    q["target_time_ms"] = g["timestamp"].shift(-1)[same_next].astype(np.int64)
    q = q.rename(columns={"timestamp": "query_time_ms", "group_index": "prefix_len_groups"})
    q.insert(0, "split", split)
    q.insert(0, "query_id", np.arange(first_query_id, first_query_id + len(q), dtype=np.int64))
    q["prefix_len_groups"] = q["prefix_len_groups"].astype(np.int32)
    return q.reset_index(drop=True)


def build_targets(queries: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (query, distinct target item) with behaviors and repeat flag."""
    ev = events[["session_id", "group_id", "group_index", "itemid", "event"]]
    first_seen = ev.groupby(["session_id", "itemid"], observed=True)["group_index"].min().rename("first_group_index")
    tgt = ev[ev["group_id"].isin(queries["target_group_id"])]
    bit = {name: 1 << i for i, name in enumerate(BEHAVIORS)}
    tgt = tgt.assign(_bit=tgt["event"].astype(str).map(bit).astype(np.int8))
    masks = (tgt.drop_duplicates(["group_id", "itemid", "_bit"])
             .groupby(["group_id", "itemid"], observed=True)["_bit"].sum())
    names = {m: [b for i, b in enumerate(BEHAVIORS) if m >> i & 1] for m in range(1, 2 ** len(BEHAVIORS))}
    behaviors = pd.Series([names[int(m)] for m in masks.to_numpy()], index=masks.index, name="behaviors")
    out = behaviors.reset_index().merge(
        queries[["query_id", "target_group_id", "session_id", "prefix_len_groups"]],
        left_on="group_id", right_on="target_group_id", validate="many_to_one")
    out = out.join(first_seen, on=["session_id", "itemid"])
    out["target_is_repeat"] = out["first_group_index"] <= out["prefix_len_groups"]
    out = out[["query_id", "itemid", "behaviors", "target_is_repeat"]]
    return out.sort_values(["query_id", "itemid"]).reset_index(drop=True)


def get_prefix(events: pd.DataFrame, query) -> pd.DataFrame:
    """Events of the prefix of one query (a row/dict with session_id and prefix_len_groups)."""
    mask = events["session_id"].eq(query["session_id"]) & events["group_index"].le(query["prefix_len_groups"])
    return events.loc[mask].sort_values(["group_index", "event_id"] if "event_id" in events else ["group_index"])


def prefix_events(queries: pd.DataFrame, events: pd.DataFrame, columns=("itemid", "event")) -> pd.DataFrame:
    """All prefix events of many queries: query_id, group_index, distance (0 = last prefix group), columns."""
    ev = events[["session_id", "group_index", *columns]]
    merged = queries[["query_id", "session_id", "prefix_len_groups"]].merge(ev, on="session_id")
    merged = merged[merged["group_index"] <= merged["prefix_len_groups"]]
    merged = merged.assign(distance=(merged["prefix_len_groups"] - merged["group_index"]).astype(np.int32))
    return merged.drop(columns=["session_id", "prefix_len_groups"]).reset_index(drop=True)


def asof_property(keys: pd.DataFrame, history: pd.DataFrame, prop: str, time_col: str = "query_time_ms"):
    """Latest ``prop`` value with timestamp <= time for every (itemid, time) row of ``keys``.

    Returns a frame aligned with ``keys`` (same index) with ``<prop>_asof`` and ``<prop>_asof_ms``.
    """
    right = (history.loc[history["property"].astype(str) == prop, ["itemid", "timestamp", "value"]]
             .sort_values("timestamp", kind="stable"))
    left = keys[["itemid", time_col]].reset_index().sort_values(time_col, kind="stable")
    merged = pd.merge_asof(left, right, left_on=time_col, right_on="timestamp", by="itemid",
                           direction="backward", allow_exact_matches=True)
    merged = merged.set_index("index").reindex(keys.index)
    return pd.DataFrame({f"{prop}_asof": merged["value"].astype("string"),
                         f"{prop}_asof_ms": merged["timestamp"].astype("Int64")}, index=keys.index)

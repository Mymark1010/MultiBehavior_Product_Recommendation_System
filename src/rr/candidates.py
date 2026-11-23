"""Candidate eligibility at query time t, and Top-K finalisation with Popularity fallback.

An item is a candidate for a query at time t when (configs/protocol.json -> candidates):

1. it is in the vocabulary of the fitting data (train, or train+validation);
2. ``known_at_ms <= t`` (first event in the fitting data or first property record);
3. only in the ``availability_variant`` (protocol v1.2.0: secondary report): the latest
   ``available`` update with ``timestamp <= t`` is not 0 (unknown is kept).

The same filter applies to model candidates and to fallback items.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAD_ITEM = -1
TIME_BITS = 41  # millisecond timestamps of 2015 are < 2**41


class CandidateFilter:
    def __init__(self, vocab_itemids, known_at_ms, avail_itemids, avail_ts, avail_values,
                 use_availability: bool = False):
        self.use_availability = use_availability
        self.items = np.asarray(vocab_itemids, dtype=np.int64)
        if len(self.items) and not (np.diff(self.items) > 0).all():
            raise ValueError("vocab_itemids phải tăng dần và duy nhất.")
        self.known_at = np.asarray(known_at_ms, dtype=np.int64)
        if len(self.known_at) != len(self.items):
            raise ValueError("known_at_ms phải cùng độ dài vocabulary.")
        avail_itemids = np.asarray(avail_itemids, dtype=np.int64)
        avail_ts = np.asarray(avail_ts, dtype=np.int64)
        if len(avail_ts) and (avail_ts.min() < 0 or avail_ts.max() >= 2 ** TIME_BITS):
            raise ValueError("timestamp ngoài khoảng mã hóa.")
        pos, in_vocab = self._positions(avail_itemids)
        keys = (pos[in_vocab] << TIME_BITS) | avail_ts[in_vocab]
        order = np.argsort(keys, kind="stable")
        self.avail_keys = keys[order]
        self.avail_values = np.asarray(avail_values, dtype=np.int8)[in_vocab][order]
        if len(self.avail_keys) and (np.diff(self.avail_keys) == 0).any():
            raise ValueError("Có hai bản available cùng item và timestamp.")

    @classmethod
    def from_frames(cls, vocab: pd.DataFrame, catalog: pd.DataFrame, history: pd.DataFrame,
                    known_at_column: str, use_availability: bool = False):
        """vocab: ``itemid``; catalog: ``itemid`` + ``known_at_column``; history: property history."""
        items = np.sort(vocab["itemid"].to_numpy(dtype=np.int64))
        known = catalog.set_index("itemid")[known_at_column].reindex(items)
        if known.isna().any():
            raise ValueError("Item trong vocabulary thiếu known_at.")
        avail = history[history["property"].astype(str) == "available"]
        return cls(items, known.to_numpy(dtype=np.int64), avail["itemid"].to_numpy(),
                   avail["timestamp"].to_numpy(), avail["value"].astype(str).astype(int).to_numpy(), use_availability)

    def _positions(self, itemids):
        itemids = np.asarray(itemids, dtype=np.int64)
        pos = np.searchsorted(self.items, itemids)
        pos = np.minimum(pos, max(len(self.items) - 1, 0))
        in_vocab = (self.items[pos] == itemids) if len(self.items) else np.zeros(len(itemids), bool)
        return pos.astype(np.int64), in_vocab

    def availability(self, itemids, times) -> np.ndarray:
        """Latest available value with timestamp <= t: 1, 0, or -1 (unknown / not in vocab)."""
        itemids = np.asarray(itemids, dtype=np.int64)
        times = np.broadcast_to(np.asarray(times, dtype=np.int64), itemids.shape)
        flat_items, flat_times = itemids.ravel(), times.ravel()
        pos, in_vocab = self._positions(flat_items)
        out = np.full(flat_items.shape, -1, dtype=np.int8)
        if len(self.avail_keys):
            query = (pos << TIME_BITS) | np.clip(flat_times, 0, 2 ** TIME_BITS - 1)
            idx = np.searchsorted(self.avail_keys, query, side="right") - 1
            found = (idx >= 0) & in_vocab
            same_item = np.zeros_like(found)
            same_item[found] = (self.avail_keys[idx[found]] >> TIME_BITS) == pos[found]
            out[same_item] = self.avail_values[idx[same_item]]
        return out.reshape(itemids.shape)

    def eligible(self, itemids, times) -> np.ndarray:
        itemids = np.asarray(itemids, dtype=np.int64)
        times = np.broadcast_to(np.asarray(times, dtype=np.int64), itemids.shape)
        flat_items, flat_times = itemids.ravel(), times.ravel()
        pos, in_vocab = self._positions(flat_items)
        ok = in_vocab & (flat_items != PAD_ITEM)
        ok &= self.known_at[pos] <= flat_times
        if self.use_availability:
            ok &= self.availability(flat_items, flat_times) != 0
        return ok.reshape(itemids.shape)


def _dedupe_rows(candidates: np.ndarray) -> np.ndarray:
    """Mark the first occurrence of every item in each row."""
    n, width = candidates.shape
    order = np.argsort(candidates, axis=1, kind="stable")
    sorted_items = np.take_along_axis(candidates, order, axis=1)
    first_sorted = np.ones_like(sorted_items, dtype=bool)
    first_sorted[:, 1:] = sorted_items[:, 1:] != sorted_items[:, :-1]
    first = np.zeros_like(first_sorted)
    np.put_along_axis(first, order, first_sorted, axis=1)
    return first


def _compact(values, mask, k, start=None, out=None, out_mask=None):
    """Move masked entries of each row to the left, after ``start`` filled slots, up to k."""
    n = len(values)
    start = np.zeros(n, dtype=np.int64) if start is None else start
    rank = np.cumsum(mask, axis=1) - 1 + start[:, None]
    keep = mask & (rank < k)
    rows = np.broadcast_to(np.arange(n)[:, None], values.shape)[keep]
    out[rows, rank[keep]] = values[keep]
    if out_mask is not None:
        out_mask[rows, rank[keep]] = True
    return np.minimum(start + mask.sum(axis=1), k)


MAX_FALLBACK_CELLS = 60_000_000  # rows x fallback items scanned at once (memory bound)
FALLBACK_WINDOW = 64  # fallback items examined per pass; rows already full stop early


def finalize_topk(candidates, times, candidate_filter: CandidateFilter, fallback_items,
                  k: int = 20, candidate_scores=None, fallback_scores=None, chunk: int = 20000,
                  exclude=None):
    """Chunked wrapper of :func:`finalize_topk_block` (same arguments and results).

    ``fallback_items`` must already be cut to the protocol's ``fallback_top_n``; the chunk size is
    reduced automatically so that ``rows x len(fallback_items)`` stays below MAX_FALLBACK_CELLS.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    times = np.asarray(times, dtype=np.int64)
    fallback_items = np.asarray(fallback_items, dtype=np.int64)
    if exclude is not None:
        exclude = np.asarray(exclude, dtype=np.int64)
        if len(exclude) != len(candidates):
            raise ValueError("exclude phải có cùng số dòng với candidates.")
    chunk = max(1, min(chunk, MAX_FALLBACK_CELLS // max(len(fallback_items), 1)))
    parts = []
    for start in range(0, len(candidates), chunk):
        sl = slice(start, start + chunk)
        parts.append(finalize_topk_block(
            candidates[sl], times[sl], candidate_filter, fallback_items, k,
            None if candidate_scores is None else np.asarray(candidate_scores)[sl], fallback_scores,
            None if exclude is None else exclude[sl]))
    if not parts:
        return (np.empty((0, k), np.int64), np.empty((0, k)), np.empty((0, k), bool))
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def _row_member(values, row_sets):
    """values [n, w], row_sets [n, e] (-1 padded): True where values[r, j] is in row_sets[r]."""
    if row_sets is None or row_sets.shape[1] == 0:
        return np.zeros(values.shape, dtype=bool)
    width = int(max(values.max(initial=0), row_sets.max(initial=0), 0)) + 2
    n = len(values)
    keys = (np.arange(n)[:, None] * width + row_sets)[row_sets != PAD_ITEM]
    return np.isin(np.arange(n)[:, None] * width + values, keys) & (values != PAD_ITEM)


def finalize_topk_block(candidates, times, candidate_filter: CandidateFilter, fallback_items,
                        k: int = 20, candidate_scores=None, fallback_scores=None, exclude=None):
    """Filter ranked model candidates, de-duplicate, cut at k, then fill with eligible fallback items.

    exclude: optional int64 [n, E] item IDs (-1 padded) never recommended for that row, used by the
    ``exclude_seen`` variant (prefix items); applies to model candidates and to fallback items.

    candidates: int64 [n, N] original item IDs ranked best-first (-1 padding allowed).
    times: query time (ms) per row.
    fallback_items: 1-D ranked item IDs (Popularity of the fitting data).
    Returns (topk [n, k], scores [n, k], is_fallback [n, k]).
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    times = np.asarray(times, dtype=np.int64)
    n = len(candidates)
    topk = np.full((n, k), PAD_ITEM, dtype=np.int64)
    scores = np.full((n, k), np.nan)
    is_fallback = np.zeros((n, k), dtype=bool)
    filled = np.zeros(n, dtype=np.int64)
    if candidates.shape[1]:
        mask = (candidates != PAD_ITEM) & _dedupe_rows(candidates)
        mask &= candidate_filter.eligible(candidates, times[:, None])
        mask &= ~_row_member(candidates, exclude)
        filled = _compact(candidates, mask, k, out=topk)
        if candidate_scores is not None:
            _compact(np.asarray(candidate_scores, dtype=float), mask, k, out=scores)
    fallback_items = np.asarray(fallback_items, dtype=np.int64)
    # Scan the ranked fallback list in windows; only rows that still need items take the next window.
    fs_all = None if fallback_scores is None else np.asarray(fallback_scores, dtype=float)
    col = 0
    need = filled < k
    while need.any() and col < len(fallback_items):
        rows = np.flatnonzero(need)
        window = fallback_items[col:col + FALLBACK_WINDOW]
        block = np.broadcast_to(window, (len(rows), len(window)))
        mask = candidate_filter.eligible(block, times[rows][:, None])
        mask &= ~_row_member(block, topk[rows])
        if exclude is not None:
            mask &= ~_row_member(block, exclude[rows])
        sub_topk, sub_scores, sub_flag = topk[rows], scores[rows], is_fallback[rows]
        new_filled = _compact(block, mask, k, start=filled[rows], out=sub_topk, out_mask=sub_flag)
        if fs_all is not None:
            fs = np.broadcast_to(fs_all[col:col + FALLBACK_WINDOW], block.shape)
            _compact(fs, mask, k, start=filled[rows], out=sub_scores)
        topk[rows], scores[rows], is_fallback[rows] = sub_topk, sub_scores, sub_flag
        filled[rows] = new_filled
        need = filled < k
        col += FALLBACK_WINDOW
    return topk, scores, is_fallback

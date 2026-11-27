"""Non-neural recommenders: Popularity, Repeat-last, ItemKNN and the session-KNN family.

Every model returns ranked *candidate* matrices of original item IDs ([n_queries, N], -1 padding)
plus scores; :func:`rr.candidates.finalize_topk` then applies the candidate filter at t, removes
duplicates, cuts at K and fills with the Popularity fallback. Models never see validation/test
events: they are fitted on a frame of fitting events (train, or train+validation for the D2 refit).

Prefix input is the long table of :func:`rr.samples.prefix_events` (``query_id, itemid, event,
distance``), where ``distance`` = 0 for the last prefix group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from rr.samples import BEHAVIORS

PAD_ITEM = -1
STRENGTH = {b: i for i, b in enumerate(BEHAVIORS)}  # view < addtocart < transaction


def behavior_weights(weights: dict) -> np.ndarray:
    return np.array([float(weights[b]) for b in BEHAVIORS])


def event_codes(event) -> np.ndarray:
    return pd.Series(event).astype(str).map(STRENGTH).to_numpy(dtype=np.int8)


# ---------------------------------------------------------------- ranking helpers
def rank_long(row, item, score, n_rows: int, width: int):
    """Per row, keep the ``width`` best (score desc, item asc) entries -> (items, scores) matrices."""
    row = np.asarray(row, dtype=np.int64)
    item = np.asarray(item, dtype=np.int64)
    score = np.asarray(score, dtype=float)
    items = np.full((n_rows, width), PAD_ITEM, dtype=np.int64)
    scores = np.full((n_rows, width), np.nan)
    if not len(row):
        return items, scores
    order = np.lexsort((item, -score, row))
    row, item, score = row[order], item[order], score[order]
    start = np.r_[0, np.flatnonzero(np.diff(row)) + 1]
    rank = np.arange(len(row)) - np.repeat(start, np.diff(np.r_[start, len(row)]))
    keep = rank < width
    items[row[keep], rank[keep]] = item[keep]
    scores[row[keep], rank[keep]] = score[keep]
    return items, scores


def topn_csr(matrix: sp.csr_matrix, n: int) -> sp.csr_matrix:
    """Keep the n largest entries of every row of a CSR matrix (ties: lower column first)."""
    m = sp.csr_matrix(matrix)
    m.sum_duplicates()
    rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
    order = np.lexsort((m.indices, -m.data, rows))
    start = np.repeat(m.indptr[:-1], np.diff(m.indptr))
    rank = np.arange(len(order)) - start
    keep = order[rank < n]
    return sp.csr_matrix((m.data[keep], (rows[keep], m.indices[keep])), shape=m.shape)


def csr_to_ranked(m: sp.csr_matrix, col_items: np.ndarray, width: int):
    rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
    return rank_long(rows, col_items[m.indices], m.data, m.shape[0], width)


class ItemIndex:
    """Dense positions 0..n-1 for the item IDs of the fitting data."""

    def __init__(self, itemids):
        self.items = np.unique(np.asarray(itemids, dtype=np.int64))

    def __len__(self):
        return len(self.items)

    def pos(self, itemids):
        itemids = np.asarray(itemids, dtype=np.int64)
        p = np.minimum(np.searchsorted(self.items, itemids), len(self.items) - 1)
        return np.where(self.items[p] == itemids, p, -1)


# ---------------------------------------------------------------- Popularity
def popularity(events: pd.DataFrame, weights: dict):
    """Behavior-weighted event counts of the fitting events, ranked (score desc, itemid asc)."""
    w = behavior_weights(weights)[event_codes(events["event"])]
    s = pd.Series(w).groupby(events["itemid"].to_numpy()).sum()
    frame = pd.DataFrame({"itemid": s.index.to_numpy(dtype=np.int64), "score": s.to_numpy()})
    frame = frame.sort_values(["score", "itemid"], ascending=[False, True], kind="stable")
    return frame["itemid"].to_numpy(), frame["score"].to_numpy()


# ---------------------------------------------------------------- Repeat-last
def repeat_last(prefix: pd.DataFrame, query_ids, width: int = 50):
    """Prefix items by most recent group, ties by stronger behavior, then itemid."""
    row_of = pd.Series(np.arange(len(query_ids)), index=np.asarray(query_ids))
    p = pd.DataFrame({"row": row_of.reindex(prefix["query_id"].to_numpy()).to_numpy(),
                      "itemid": prefix["itemid"].to_numpy(), "distance": prefix["distance"].to_numpy(),
                      "strength": event_codes(prefix["event"])})
    agg = (p.sort_values(["row", "itemid", "distance", "strength"], ascending=[True, True, True, False])
             .drop_duplicates(["row", "itemid"]))
    # score: higher is better; distance dominates, strength breaks ties.
    score = -agg["distance"].to_numpy(dtype=float) * 10 + agg["strength"].to_numpy()
    return rank_long(agg["row"].to_numpy(), agg["itemid"].to_numpy(), score, len(query_ids), width)


# ---------------------------------------------------------------- query vectors
def query_weights(prefix: pd.DataFrame, query_ids, index: ItemIndex, scheme: str, bw: np.ndarray | None = None,
                  decay: float = 1.0, lam: float = 1.0):
    """Long (row, item_pos, weight, distance) of the query vector; one entry per (query, item).

    scheme: ``binary`` (1), ``power`` (decay**distance), ``exp`` (STAN: exp(-d / lam)). Optional behavior
    weight of the strongest prefix event of that item multiplies the weight.
    """
    row_of = pd.Series(np.arange(len(query_ids)), index=np.asarray(query_ids))
    pos = index.pos(prefix["itemid"].to_numpy())
    keep = pos >= 0
    rows = row_of.reindex(prefix["query_id"].to_numpy()[keep]).to_numpy()
    d = prefix["distance"].to_numpy()[keep].astype(float)
    if scheme == "binary":
        w = np.ones_like(d)
    elif scheme == "power":
        w = decay ** d
    elif scheme == "exp":
        w = np.exp(-d / lam)
    else:
        raise ValueError(scheme)
    if bw is not None:
        w = w * bw[event_codes(prefix["event"].to_numpy()[keep])]
    f = pd.DataFrame({"row": rows, "pos": pos[keep], "w": w, "d": d})
    f = f.sort_values(["row", "pos", "w", "d"], ascending=[True, True, False, True]).drop_duplicates(["row", "pos"])
    return f["row"].to_numpy(), f["pos"].to_numpy(), f["w"].to_numpy(), f["d"].to_numpy()


# ---------------------------------------------------------------- ItemKNN
def session_item_matrix(events: pd.DataFrame, index: ItemIndex, bw: np.ndarray):
    """Sessions x items, value = weight of the strongest behavior of that item in the session."""
    sess, sess_pos = np.unique(events["session_id"].to_numpy(), return_inverse=True)
    w = bw[event_codes(events["event"])]
    # max (not sum) over repeated (session, item) events.
    f = pd.DataFrame({"r": sess_pos, "c": index.pos(events["itemid"].to_numpy()), "w": w}).groupby(["r", "c"])["w"].max()
    return sp.csr_matrix((f.to_numpy(), (f.index.get_level_values(0), f.index.get_level_values(1))),
                         shape=(len(sess), len(index)))


class ItemKNN:
    def __init__(self, n_neighbors: int = 100, weights: dict | None = None):
        self.n_neighbors = n_neighbors
        self.weights = weights or dict.fromkeys(BEHAVIORS, 1)

    def fit(self, events: pd.DataFrame):
        self.index = ItemIndex(events["itemid"].to_numpy())
        bw = behavior_weights(self.weights)
        x = session_item_matrix(events, self.index, bw)
        x = x[np.diff(x.indptr) >= 2]  # sessions with one item add no co-occurrence
        co = (x.T @ x).tocsr()
        norms = np.sqrt(co.diagonal())
        co.setdiag(0)
        co.eliminate_zeros()
        inv = np.where(norms > 0, 1 / np.maximum(norms, 1e-12), 0)
        co = sp.diags(inv) @ co @ sp.diags(inv)
        self.sim = topn_csr(co.tocsr(), self.n_neighbors)
        return self

    def recommend(self, prefix: pd.DataFrame, query_ids, width: int = 200, decay: float = 1.0,
                  use_prefix_behavior: bool = False, chunk: int = 20_000):
        rows, pos, w, _ = query_weights(prefix, query_ids, self.index, "power",
                                        behavior_weights(self.weights) if use_prefix_behavior else None, decay)
        n = len(query_ids)
        items = np.full((n, width), PAD_ITEM, dtype=np.int64)
        scores = np.full((n, width), np.nan)
        q = sp.csr_matrix((w, (rows, pos)), shape=(n, len(self.index)))
        for start in range(0, n, chunk):
            block = (q[start:start + chunk] @ self.sim).tocsr()
            it, sc = csr_to_ranked(block, self.index.items, width)
            items[start:start + chunk], scores[start:start + chunk] = it, sc
        return items, scores


# ---------------------------------------------------------------- Session-KNN family
class SessionKNN:
    """SKNN / STAN / VSTAN over the sessions of the fitting data.

    * Store: sessions with >= 2 distinct items. For every item only the ``sample_size`` most recent
      sessions (by session end) are indexed; candidate neighbours of a query are the union over its
      prefix items (approximation of "m most recent sessions" sampling, vectorised).
    * sim(q, s) = sum_{shared items} w_q(i) / (||w_q|| * sqrt(|s|)); STAN multiplies by
      exp(-(t_q - t_end(s)) / lambda2).
    * score(i) = sum_{top-k neighbours s containing i} sim(q, s) * w_b(i, s) [* STAN position factor
      exp(-|pos_s(i) - pos_s(match)| / lambda3), match = most recent prefix item shared with s].
    * VSTAN = STAN + IDF: a shared prefix item contributes w_q(i) * idf(i)^lambda_idf to the
      similarity and a candidate score is multiplied by idf(j)^lambda_idf, idf = log(N / df).
    """

    def __init__(self, variant: str = "sknn", k: int = 100, sample_size: int = 500, weights: dict | None = None,
                 lambda1: float = 2.0, lambda2_ms: float = 7 * 86_400_000, lambda3: float = 2.0,
                 lambda_idf: float = 1.0):
        if variant not in {"sknn", "stan", "vstan"}:
            raise ValueError(variant)
        self.variant, self.k, self.sample_size = variant, k, sample_size
        self.weights = weights or dict.fromkeys(BEHAVIORS, 1)
        self.lambda1, self.lambda2_ms, self.lambda3 = lambda1, lambda2_ms, lambda3
        self.lambda_idf = lambda_idf if variant == "vstan" else 0.0
        self.positional = variant in {"stan", "vstan"}

    def fit(self, events: pd.DataFrame):
        bw = behavior_weights(self.weights)
        self.index = ItemIndex(events["itemid"].to_numpy())
        e = pd.DataFrame({"session_id": events["session_id"].to_numpy(), "pos": self.index.pos(events["itemid"].to_numpy()),
                          "w": bw[event_codes(events["event"])], "g": events["group_index"].to_numpy(),
                          "t": events["timestamp"].to_numpy()})
        per = e.groupby(["session_id", "pos"], sort=True).agg(w=("w", "max"), g=("g", "max")).reset_index()
        n_items = per.groupby("session_id").size()
        keep_sessions = n_items.index[n_items >= 2]
        per = per[per["session_id"].isin(keep_sessions)]
        end = e.groupby("session_id")["t"].max().reindex(keep_sessions)
        self.sessions = keep_sessions.to_numpy()
        self.session_end = end.to_numpy(dtype=np.int64)
        spos = np.searchsorted(self.sessions, per["session_id"].to_numpy())
        # session -> items (CSR order by session position)
        self.s_indptr = np.r_[0, np.cumsum(np.bincount(spos, minlength=len(self.sessions)))]
        self.s_items = per["pos"].to_numpy(dtype=np.int64)
        self.s_w = per["w"].to_numpy(dtype=float)
        self.s_g = per["g"].to_numpy(dtype=np.int64)
        self.s_len = np.diff(self.s_indptr).astype(float)
        # item -> most recent sessions
        inv = pd.DataFrame({"pos": self.s_items, "s": spos, "end": self.session_end[spos], "g": self.s_g})
        inv = inv.sort_values(["pos", "end", "s"], ascending=[True, False, True])
        inv = inv[inv.groupby("pos").cumcount() < self.sample_size]
        self.i_indptr = np.r_[0, np.cumsum(np.bincount(inv["pos"].to_numpy(), minlength=len(self.index)))]
        self.i_sess = inv["s"].to_numpy(dtype=np.int64)
        self.i_g = inv["g"].to_numpy(dtype=np.int64)  # group position of the item inside that session
        df = np.bincount(self.s_items, minlength=len(self.index)).astype(float)
        self.idf = np.where(df > 0, np.log(len(self.sessions) / np.maximum(df, 1)), 0.0)
        self.idf_factor = np.maximum(self.idf, 1e-6) ** self.lambda_idf if self.lambda_idf else np.ones_like(self.idf)
        return self

    def _query_vectors(self, prefix, query_ids):
        scheme = {"sknn": "binary", "stan": "exp", "vstan": "exp"}[self.variant]
        return query_weights(prefix, query_ids, self.index, scheme, lam=self.lambda1)

    def recommend(self, prefix: pd.DataFrame, query_ids, query_times, width: int = 200, chunk: int = 2_000):
        query_ids = np.asarray(query_ids)
        query_times = np.asarray(query_times, dtype=np.int64)
        n = len(query_ids)
        rows, pos, w, d = self._query_vectors(prefix, query_ids)
        items = np.full((n, width), PAD_ITEM, dtype=np.int64)
        scores = np.full((n, width), np.nan)
        order = np.argsort(rows, kind="stable")
        rows, pos, w, d = rows[order], pos[order], w[order], d[order]
        bounds = np.searchsorted(rows, np.arange(0, n + chunk, chunk))
        n_sess = len(self.sessions)
        for c, start in enumerate(range(0, n, chunk)):
            sl = slice(bounds[c], bounds[c + 1])
            r, p, wq, dq = rows[sl] - start, pos[sl], w[sl], d[sl]
            if not len(r):
                continue
            m = min(chunk, n - start)
            qnorm = np.sqrt(np.bincount(r, weights=wq ** 2, minlength=m))
            # expand prefix items to indexed sessions
            lo, cnt = self.i_indptr[p], self.i_indptr[p + 1] - self.i_indptr[p]
            rep = np.repeat(np.arange(len(r)), cnt)
            off = np.arange(len(rep)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            sess = self.i_sess[lo[rep] + off]
            key = r[rep].astype(np.int64) * n_sess + sess
            uniq, inv = np.unique(key, return_inverse=True)
            dot = np.bincount(inv, weights=wq[rep] * self.idf_factor[p[rep]], minlength=len(uniq))
            qr, qs = uniq // n_sess, uniq % n_sess
            sim = dot / (qnorm[qr] * np.sqrt(self.s_len[qs]))
            if self.positional:
                age = np.maximum(query_times[start + qr] - self.session_end[qs], 0)
                sim = sim * np.exp(-age / self.lambda2_ms)
                # match position: the shared item with the smallest prefix distance
                first = np.lexsort((dq[rep], inv))
                first = first[np.r_[True, inv[first][1:] != inv[first][:-1]]]
                match_g = np.empty(len(uniq), dtype=np.int64)
                match_g[inv[first]] = self.i_g[lo[rep] + off][first]
            # top-k neighbours per query
            o = np.lexsort((qs, -sim, qr))
            st = np.r_[0, np.flatnonzero(np.diff(qr[o])) + 1]
            rank = np.arange(len(o)) - np.repeat(st, np.diff(np.r_[st, len(o)]))
            nb = o[rank < self.k]
            nr, ns, nsim = qr[nb], qs[nb], sim[nb]
            # neighbours -> their items
            cnt2 = (self.s_indptr[ns + 1] - self.s_indptr[ns]).astype(np.int64)
            rep2 = np.repeat(np.arange(len(nb)), cnt2)
            off2 = np.arange(len(rep2)) - np.repeat(np.cumsum(cnt2) - cnt2, cnt2)
            at = self.s_indptr[ns][rep2] + off2
            contrib = nsim[rep2] * self.s_w[at]
            if self.positional:
                contrib = contrib * np.exp(-np.abs(self.s_g[at] - match_g[nb][rep2]) / self.lambda3)
            contrib = contrib * self.idf_factor[self.s_items[at]]
            key2 = nr[rep2] * len(self.index) + self.s_items[at]
            u2, inv2 = np.unique(key2, return_inverse=True)
            sc = np.bincount(inv2, weights=contrib, minlength=len(u2))
            it, s = rank_long(u2 // len(self.index), self.index.items[u2 % len(self.index)], sc, m, width)
            items[start:start + m], scores[start:start + m] = it, s
        return items, scores

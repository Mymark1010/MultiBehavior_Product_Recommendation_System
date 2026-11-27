"""GRU4Rec over timestamp groups (PyTorch, CPU), as fixed in configs/protocol.json -> gru4rec.

* Input of step t: mean embedding of the in-vocabulary items of group t + mean embedding of its
  distinct behaviors (``use_behavior=True``, the multi-behavior model), or the item part alone
  (``use_behavior=False``, the original item-only GRU4Rec). Out-of-vocabulary input items are
  dropped (PAD); the behavior part is kept.
* The output of step t predicts the item set of group t+1. Hidden state starts at zero for every
  session (or training window), so nothing flows between sessions.
* Loss: sampled softmax cross-entropy. Candidates of a batch = unique(in-batch positives ∪ shared
  negatives sampled ∝ popularity^alpha); logits get the log-Q correction; the label of a step is
  spread uniformly over its |G| target items. Because candidates are de-duplicated, a sampled
  negative that is also a positive simply becomes that positive (accidental hits never count as
  negatives).
* Training windows: sessions longer than ``max_len`` groups are cut into windows of ``max_len``
  groups overlapping by one group (D3). Evaluation always uses the full prefix.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from rr.samples import BEHAVIORS, PAD_IDX, UNK_IDX, encode

BEH_CODE = {b: i for i, b in enumerate(BEHAVIORS)}


class GroupData:
    """Sessions -> groups -> (item_idx list, behavior list) in CSR form."""

    def __init__(self, events: pd.DataFrame, vocab: pd.DataFrame):
        e = events[["session_id", "group_index", "itemid", "event"]].copy()
        e["item_idx"] = encode(e["itemid"].to_numpy(), vocab)
        e.loc[e["item_idx"] == UNK_IDX, "item_idx"] = PAD_IDX
        e["beh"] = e["event"].astype(str).map(BEH_CODE).astype(np.int64)
        e = e.sort_values(["session_id", "group_index"], kind="stable")
        g = e[["session_id", "group_index"]].drop_duplicates()
        self.group_session = g["session_id"].to_numpy()
        self.group_index = g["group_index"].to_numpy()
        gid = pd.MultiIndex.from_frame(g).get_indexer(pd.MultiIndex.from_frame(e[["session_id", "group_index"]]))
        items = pd.DataFrame({"g": gid, "i": e["item_idx"].to_numpy()})
        items = items[items["i"] != PAD_IDX].drop_duplicates().sort_values(["g", "i"])
        beh = pd.DataFrame({"g": gid, "b": e["beh"].to_numpy()}).drop_duplicates().sort_values(["g", "b"])
        n = len(g)
        self.i_ptr = np.r_[0, np.cumsum(np.bincount(items["g"], minlength=n))].astype(np.int64)
        self.i_val = items["i"].to_numpy(dtype=np.int64)
        self.b_ptr = np.r_[0, np.cumsum(np.bincount(beh["g"], minlength=n))].astype(np.int64)
        self.b_val = beh["b"].to_numpy(dtype=np.int64)
        sess, first = np.unique(self.group_session, return_index=True)
        self.sessions = sess
        self.s_start = first.astype(np.int64)
        self.s_len = np.diff(np.r_[first, n]).astype(np.int64)

    def windows(self, max_len: int | None):
        """(start_group, length) for sessions with >= 2 groups, cut into overlapping windows."""
        out = []
        for s, n in zip(self.s_start, self.s_len):
            if n < 2:
                continue
            if max_len is None or n <= max_len:
                out.append((s, n))
                continue
            start = 0
            while start < n - 1:
                length = min(max_len, n - start)
                out.append((s + start, length))
                start += max_len - 1
        return np.array(out, dtype=np.int64).reshape(-1, 2)

    def bags(self, groups: np.ndarray):
        """Concatenated item / behavior lists and offsets for EmbeddingBag."""
        def gather(ptr, val):
            lo, cnt = ptr[groups], ptr[groups + 1] - ptr[groups]
            off = np.r_[0, np.cumsum(cnt)[:-1]]
            idx = np.repeat(lo - off, cnt) + np.arange(cnt.sum())
            return torch.from_numpy(val[idx]), torch.from_numpy(off)
        return gather(self.i_ptr, self.i_val), gather(self.b_ptr, self.b_val)


class GRU4Rec(nn.Module):
    """GRU over timestamp groups.

    ``constrained=True`` follows the official GRU4Rec "constrained embedding": the item embedding
    used for the input bag is the same matrix used to score output items, and its size equals the
    hidden size (no output projection). Item embeddings and biases use sparse gradients.
    ``use_behavior=False`` removes the behavior embedding entirely (original item-only GRU4Rec).
    """

    def __init__(self, n_items: int, dim: int = 64, hidden: int = 100, dropout: float = 0.0,
                 dropout_embed: float | None = None, constrained: bool = False, use_behavior: bool = True):
        super().__init__()
        self.constrained = constrained
        self.use_behavior = use_behavior
        dim = hidden if constrained else dim
        self.item_in = nn.Embedding(n_items, dim, sparse=True)
        self.beh_in = nn.EmbeddingBag(len(BEHAVIORS), dim, mode="mean") if use_behavior else None
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.drop_embed = nn.Dropout(dropout if dropout_embed is None else dropout_embed)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Identity() if constrained else nn.Linear(hidden, dim)
        self.item_out = self.item_in if constrained else nn.Embedding(n_items, dim, sparse=True)
        self.bias = nn.Embedding(n_items, 1, sparse=True)
        self.out_dim = dim
        nn.init.normal_(self.item_in.weight, std=0.05)
        if self.beh_in is not None:  # same scale as items (the N(0, 1) default swamped the item mean)
            nn.init.normal_(self.beh_in.weight, std=0.05)
        if not constrained:
            nn.init.normal_(self.item_out.weight, std=0.05)
        nn.init.zeros_(self.bias.weight)

    def forward(self, data: GroupData, starts: np.ndarray, lengths: np.ndarray, order=None):
        """Outputs [B, L, out_dim] for sequences of consecutive groups (or explicit group orders)."""
        b, l = len(starts), int(lengths.max())
        grid = starts[:, None] + np.arange(l)[None, :] if order is None else order
        mask = np.arange(l)[None, :] < lengths[:, None]
        groups = grid[mask]
        (it, it_off), (bh, bh_off) = data.bags(groups)
        x = nn.functional.embedding_bag(it, self.item_in.weight, it_off, mode="mean", sparse=True)
        if self.beh_in is not None:
            x = x + self.beh_in(bh, bh_off)
        inp = torch.zeros(b, l, x.shape[1])
        inp[torch.from_numpy(mask)] = x
        inp = self.drop_embed(inp)
        packed = nn.utils.rnn.pack_padded_sequence(inp, torch.from_numpy(lengths), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)  # zero initial state for every sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=l)
        return self.proj(self.drop(out)), mask

    def item_scores(self, h: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return h @ self.item_out(items).T + self.bias(items).squeeze(1)

    def scores(self, h: torch.Tensor) -> torch.Tensor:
        s = h @ self.item_out.weight.T + self.bias.weight.squeeze(1)
        s[:, PAD_IDX] = -torch.inf
        s[:, UNK_IDX] = -torch.inf
        return s


def target_lists(data: GroupData, starts, lengths):
    """For every valid step (b, t) with t < length-1: row in flattened outputs and the target CSR."""
    b_idx, t_idx = [], []
    for b, n in enumerate(lengths):
        b_idx.append(np.full(n - 1, b)); t_idx.append(np.arange(n - 1))
    b_idx, t_idx = np.concatenate(b_idx), np.concatenate(t_idx)
    tgt_groups = starts[b_idx] + t_idx + 1
    lo, cnt = data.i_ptr[tgt_groups], data.i_ptr[tgt_groups + 1] - data.i_ptr[tgt_groups]
    return b_idx, t_idx, tgt_groups, lo, cnt


def sampled_loss(model, h, data, lo, cnt, neg_probs, n_neg, log_q, generator, loss: str = "cross-entropy",
                 bpreg: float = 1.0, elu: float = 0.5):
    """Sampled loss over candidates = unique(in-batch positives and shared negatives).

    cross-entropy: softmax over candidates with log-Q correction, label 1/|G| on each target item.
    bpr-max (Hidasi & Karatzoglou 2018, as in the official code): for each target item i,
    -log sum_j s_j sigmoid(r_i - r_j) + bpreg sum_j s_j r_j^2, s = softmax of the negative scores,
    negatives = candidates that are not targets of that step; scores pass through ELU(elu) first.
    """
    keep = cnt > 0
    h, lo, cnt = h[torch.from_numpy(keep)], lo[keep], cnt[keep]
    pos = data.i_val[np.repeat(lo, cnt) + np.arange(cnt.sum()) - np.repeat(np.r_[0, np.cumsum(cnt)[:-1]], cnt)]
    neg = torch.multinomial(neg_probs, n_neg, replacement=True, generator=generator).numpy()
    cand, inv = np.unique(np.r_[pos, neg], return_inverse=True)
    cand_t = torch.from_numpy(cand)
    rows = torch.from_numpy(np.repeat(np.arange(len(cnt)), cnt))
    cols = torch.from_numpy(inv[:len(pos)])
    w = torch.from_numpy(1.0 / np.repeat(cnt, cnt)).float()
    r = model.item_scores(h, cand_t)
    if loss == "cross-entropy":
        logits = r - log_q[cand_t]
        y = torch.zeros_like(logits)
        y[rows, cols] = w
        return -(y * torch.log_softmax(logits, dim=1)).sum() / len(cnt)
    if loss != "bpr-max":
        raise ValueError(loss)
    if elu:
        r = nn.functional.elu(r, alpha=elu)
    is_pos = torch.zeros_like(r, dtype=torch.bool)
    is_pos[rows, cols] = True
    s = torch.softmax(r.masked_fill(is_pos, -torch.inf), dim=1)  # 0 on the step's targets
    rp = r[rows, cols]
    term = (s[rows] * torch.sigmoid(rp[:, None] - r[rows])).sum(1)
    reg = (s * r ** 2).sum(1)[rows]
    return (w * (-torch.log(term + 1e-24) + bpreg * reg)).sum() / len(cnt)


def make_batches(windows: np.ndarray, batch_size: int, rng: np.random.Generator):
    """Length-bucketed batches in random order."""
    order = np.lexsort((rng.random(len(windows)), windows[:, 1]))
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


def train_epoch(model, optimizers, data, windows, neg_probs, log_q, batch_size=256, n_neg=2048, seed=0,
                max_batches=None, loss="cross-entropy", bpreg=1.0, elu=0.5):
    model.train()
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for i, bi in enumerate(make_batches(windows, batch_size, rng)):
        if max_batches is not None and i >= max_batches:
            break
        starts, lengths = windows[bi, 0], windows[bi, 1]
        out, _ = model(data, starts, lengths)
        b_idx, t_idx, _, lo, cnt = target_lists(data, starts, lengths)
        h = out[torch.from_numpy(b_idx), torch.from_numpy(t_idx)]
        value = sampled_loss(model, h, data, lo, cnt, neg_probs, n_neg, log_q, gen, loss, bpreg, elu)
        for opt in optimizers:
            opt.zero_grad()
        value.backward()
        for opt in optimizers:
            opt.step()
        losses.append(float(value.detach()))
    return float(np.mean(losses)), losses


def negative_distribution(data: GroupData, n_items: int, alpha: float = 0.75):
    """Shared negatives ~ popularity^alpha (``sample_alpha`` of the official code)."""
    counts = np.bincount(data.i_val, minlength=n_items).astype(float)
    counts[[PAD_IDX, UNK_IDX]] = 0
    p = counts ** alpha
    p /= p.sum()
    log_q = np.log(np.maximum(p, 1e-12))
    return torch.from_numpy(p).float(), torch.from_numpy(log_q).float()


@torch.no_grad()
def query_states(model, data: GroupData, queries: pd.DataFrame, batch_size: int = 512, mode: str = "sequential"):
    """Output vector after the prefix of every query (rows aligned with ``queries``).

    mode ``sequential``: run every session once and read step ``prefix_len_groups``.
    mode ``per_query``: run each prefix as its own sequence (same result; used as a leakage check).
    """
    model.eval()
    pos = np.searchsorted(data.sessions, queries["session_id"].to_numpy())
    if not (data.sessions[np.minimum(pos, len(data.sessions) - 1)] == queries["session_id"].to_numpy()).all():
        raise ValueError("Có query không thuộc dữ liệu GroupData.")
    t = queries["prefix_len_groups"].to_numpy(dtype=np.int64)
    start = data.s_start[pos]
    out = torch.empty(len(queries), model.out_dim)
    if mode == "sequential":
        sess_rows = np.unique(pos)
        lengths_all = data.s_len[sess_rows]
        order = np.argsort(lengths_all, kind="stable")
        result = {}
        for i in range(0, len(order), batch_size):
            sel = order[i:i + batch_size]
            o, _ = model(data, data.s_start[sess_rows[sel]], lengths_all[sel])
            for j, s in enumerate(sess_rows[sel]):
                result[int(s)] = o[j]
        for k in range(len(queries)):
            out[k] = result[int(pos[k])][t[k] - 1]
        return out
    if mode != "per_query":
        raise ValueError(f"mode không hợp lệ: {mode}")
    starts, lengths = start, t
    order_idx = np.argsort(lengths, kind="stable")
    for i in range(0, len(order_idx), batch_size):
        sel = order_idx[i:i + batch_size]
        L = int(lengths[sel].max())
        grid = starts[sel][:, None] + np.arange(L)[None, :]
        grid = np.where(np.arange(L)[None, :] < lengths[sel][:, None], grid, 0)
        o, _ = model(data, starts[sel], lengths[sel], order=grid)
        out[torch.from_numpy(sel)] = o[torch.arange(len(sel)), torch.from_numpy(lengths[sel] - 1)]
    return out


@torch.no_grad()
def topn_items(model, h: torch.Tensor, idx_to_item: np.ndarray, n: int = 200, chunk: int = 1024):
    items = np.empty((len(h), n), dtype=np.int64)
    scores = np.empty((len(h), n))
    for i in range(0, len(h), chunk):
        s = model.scores(h[i:i + chunk])
        v, ix = torch.topk(s, n, dim=1)
        items[i:i + chunk] = idx_to_item[ix.numpy()]
        scores[i:i + chunk] = v.numpy()
    return items, scores


def idx_to_itemid(vocab: pd.DataFrame) -> np.ndarray:
    n = int(vocab["item_idx"].max()) + 1
    out = np.full(n, -1, dtype=np.int64)
    out[vocab["item_idx"].to_numpy()] = vocab["itemid"].to_numpy()
    return out


def optimizers_for(model, lr):
    """SparseAdam for item embeddings/biases (sparse gradients), Adam for the rest."""
    sparse = {id(p): p for n, p in model.named_parameters() if n.startswith(("item_in", "item_out", "bias"))}
    dense = [p for p in model.parameters() if id(p) not in sparse]
    return [torch.optim.SparseAdam(list(sparse.values()), lr=lr), torch.optim.Adam(dense, lr=lr)]


def fit(model, data, windows, epochs, lr=1e-3, batch_size=256, n_neg=2048, seed=0, on_epoch=None, patience=None,
        loss="cross-entropy", bpreg=1.0, elu=0.5, sample_alpha=0.75):
    """Train for ``epochs`` (with optional early stopping on ``on_epoch(epoch) -> score``)."""
    torch.manual_seed(seed)
    n_items = model.item_in.num_embeddings
    neg_probs, log_q = negative_distribution(data, n_items, sample_alpha)
    opts = optimizers_for(model, lr)
    history, best, best_state, bad = [], -np.inf, None, 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        value, _ = train_epoch(model, opts, data, windows, neg_probs, log_q, batch_size, n_neg, seed + epoch,
                               loss=loss, bpreg=bpreg, elu=elu)
        row = {"epoch": epoch, "loss": value, "train_seconds": round(time.perf_counter() - t0, 1)}
        if on_epoch is not None:
            score = on_epoch(epoch)
            row["score"] = score
            if score > best:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                row["best"] = True
            else:
                bad += 1
        history.append(row)
        if patience is not None and bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def build_from_config(cfg: dict, n_items: int | None = None) -> GRU4Rec:
    """Model skeleton from ``models/gru4rec/config.json`` (``n_items`` overrides it for a refit vocabulary)."""
    return GRU4Rec(n_items or cfg["n_items"], dim=cfg.get("dim", 64), hidden=cfg["hidden"], dropout=cfg["dropout"],
                   dropout_embed=cfg.get("dropout_embed"), constrained=cfg["constrained"],
                   use_behavior=cfg.get("use_behavior", True))

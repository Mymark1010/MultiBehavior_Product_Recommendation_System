"""SR-GNN (Wu et al., AAAI 2019) adapted to timestamp groups, as fixed in protocol v1.2.0.

Session graph of one query (the most recent ``max_groups`` prefix groups):

* nodes = distinct prefix items (out-of-vocabulary items share the UNK node); the node input is
  its item embedding + the mean embedding of the behaviors of its prefix events;
* directed edges from every item of group g to every item of group g+1 (consecutive prefix
  groups); items inside one group are never ordered; repeated (src, dst) pairs count once and
  rows are normalised by degree as in SR-GNN;
* one gated GNN step, then the SR-GNN readout: local interest = mean state of the last-group
  nodes, global interest = soft attention over all nodes, session vector = W [local; global];
* item scores = session vector . item embedding + bias (embeddings shared with the input), trained
  with the same sampled softmax (log-Q corrected, labels 1/|G|) as GRU4Rec.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from rr.samples import BEHAVIORS, PAD_IDX, UNK_IDX, encode

BEH_CODE = {b: i for i, b in enumerate(BEHAVIORS)}


class GraphData:
    """CSR graphs for many queries; rows aligned with ``query_ids``."""

    def __init__(self, prefix: pd.DataFrame, query_ids, vocab: pd.DataFrame):
        query_ids = np.asarray(query_ids)
        row_of = pd.Series(np.arange(len(query_ids)), index=query_ids)
        p = pd.DataFrame({"q": row_of.reindex(prefix["query_id"].to_numpy()).to_numpy(),
                          "d": prefix["distance"].to_numpy(),
                          "i": encode(prefix["itemid"].to_numpy(), vocab)})
        code = prefix["event"].astype(str).map(BEH_CODE).to_numpy()
        bcols = [f"b{k}" for k in range(len(BEHAVIORS))]
        for k, c in enumerate(bcols):
            p[c] = (code == k).astype(np.float32)
        nodes = p.groupby(["q", "i"], as_index=False)[bcols].max().sort_values(["q", "i"])
        nodes["local"] = nodes.groupby("q").cumcount()
        n = len(query_ids)
        self.n_ptr = np.r_[0, np.cumsum(np.bincount(nodes["q"], minlength=n))].astype(np.int64)
        self.n_item = nodes["i"].to_numpy(dtype=np.int64)
        self.n_beh = nodes[bcols].to_numpy(dtype=np.float32)
        gi = p[["q", "d", "i"]].drop_duplicates().merge(nodes[["q", "i", "local"]], on=["q", "i"])
        older = gi.rename(columns={"local": "src"}).assign(d=lambda f: f["d"] - 1)
        edges = older[["q", "d", "src"]].merge(gi.rename(columns={"local": "dst"})[["q", "d", "dst"]], on=["q", "d"])
        edges = edges[["q", "src", "dst"]].drop_duplicates().sort_values(["q", "src", "dst"])
        self.e_ptr = np.r_[0, np.cumsum(np.bincount(edges["q"], minlength=n))].astype(np.int64)
        self.e_src = edges["src"].to_numpy(dtype=np.int64)
        self.e_dst = edges["dst"].to_numpy(dtype=np.int64)
        last = gi[gi["d"] == 0].sort_values(["q", "local"])
        self.l_ptr = np.r_[0, np.cumsum(np.bincount(last["q"], minlength=n))].astype(np.int64)
        self.l_val = last["local"].to_numpy(dtype=np.int64)
        if (np.diff(self.n_ptr) == 0).any() or (np.diff(self.l_ptr) == 0).any():
            raise ValueError("Mọi query phải có ít nhất một nút và một nút ở nhóm cuối.")

    def __len__(self):
        return len(self.n_ptr) - 1

    def batch(self, rows: np.ndarray):
        """Padded tensors for a batch of query rows."""
        counts = np.diff(self.n_ptr)[rows]
        b, m = len(rows), int(counts.max())
        node = np.zeros((b, m), dtype=np.int64)
        beh = np.zeros((b, m, len(BEHAVIORS)), dtype=np.float32)
        mask = np.zeros((b, m), dtype=bool)
        last = np.zeros((b, m), dtype=np.float32)
        a_out = np.zeros((b, m, m), dtype=np.float32)
        for j, r in enumerate(rows):
            lo, hi = self.n_ptr[r], self.n_ptr[r + 1]
            node[j, :hi - lo] = self.n_item[lo:hi]
            beh[j, :hi - lo] = self.n_beh[lo:hi]
            mask[j, :hi - lo] = True
            last[j, self.l_val[self.l_ptr[r]:self.l_ptr[r + 1]]] = 1.0
            e = slice(self.e_ptr[r], self.e_ptr[r + 1])
            a_out[j, self.e_src[e], self.e_dst[e]] = 1.0
        a_in = a_out.transpose(0, 2, 1).copy()
        a_out /= np.maximum(a_out.sum(2, keepdims=True), 1)
        a_in /= np.maximum(a_in.sum(2, keepdims=True), 1)
        last /= last.sum(1, keepdims=True)
        return tuple(torch.from_numpy(x) for x in (node, beh, mask, last, a_in, a_out))


class SRGNN(nn.Module):
    def __init__(self, n_items: int, dim: int = 100, steps: int = 1, dropout: float = 0.0):
        super().__init__()
        self.item_in = nn.Embedding(n_items, dim, sparse=True)
        self.item_out = self.item_in
        self.bias = nn.Embedding(n_items, 1, sparse=True)
        self.beh_in = nn.Linear(len(BEHAVIORS), dim, bias=False)
        self.lin_in, self.lin_out = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.cell = nn.GRUCell(2 * dim, dim)
        self.w1, self.w2 = nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim)
        self.q = nn.Linear(dim, 1, bias=False)
        self.w3 = nn.Linear(2 * dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.steps, self.out_dim = steps, dim
        nn.init.normal_(self.item_in.weight, std=0.05)
        nn.init.zeros_(self.bias.weight)

    def forward(self, node, beh, mask, last, a_in, a_out):
        beh_mean = beh / beh.sum(-1, keepdim=True).clamp(min=1)
        h = nn.functional.embedding(node, self.item_in.weight, sparse=True) + self.beh_in(beh_mean)
        h = self.drop(h)
        b, m, d = h.shape
        for _ in range(self.steps):
            a = torch.cat([a_in @ self.lin_in(h), a_out @ self.lin_out(h)], dim=-1)
            h = self.cell(a.reshape(b * m, 2 * d), h.reshape(b * m, d)).reshape(b, m, d)
        local = (last.unsqueeze(-1) * h).sum(1)
        alpha = self.q(torch.sigmoid(self.w1(local).unsqueeze(1) + self.w2(h))).squeeze(-1)
        glob = (alpha.masked_fill(~mask, 0).unsqueeze(-1) * h).sum(1)
        return self.w3(self.drop(torch.cat([local, glob], dim=-1)))

    def item_scores(self, s, items):
        return s @ self.item_out(items).T + self.bias(items).squeeze(1)

    def scores(self, s):
        out = s @ self.item_out.weight.T + self.bias.weight.squeeze(1)
        out[:, PAD_IDX] = -torch.inf
        out[:, UNK_IDX] = -torch.inf
        return out


def target_csr(targets: pd.DataFrame, query_ids, vocab: pd.DataFrame):
    """In-vocabulary target item_idx per query row (CSR)."""
    row_of = pd.Series(np.arange(len(query_ids)), index=np.asarray(query_ids))
    t = pd.DataFrame({"r": row_of.reindex(targets["query_id"].to_numpy()).to_numpy(),
                      "i": encode(targets["itemid"].to_numpy(), vocab)})
    t = t[t["i"] != UNK_IDX].sort_values(["r", "i"])
    ptr = np.r_[0, np.cumsum(np.bincount(t["r"], minlength=len(query_ids)))].astype(np.int64)
    return ptr, t["i"].to_numpy(dtype=np.int64)


def sampled_ce(model, s, pos_ptr, pos_val, rows, neg_probs, log_q, n_neg, gen):
    cnt = pos_ptr[rows + 1] - pos_ptr[rows]
    keep = cnt > 0
    s, rows, cnt = s[torch.from_numpy(keep)], rows[keep], cnt[keep]
    pos = np.concatenate([pos_val[pos_ptr[r]:pos_ptr[r + 1]] for r in rows])
    neg = torch.multinomial(neg_probs, n_neg, replacement=True, generator=gen).numpy()
    cand, inv = np.unique(np.r_[pos, neg], return_inverse=True)
    cand_t = torch.from_numpy(cand)
    logits = model.item_scores(s, cand_t) - log_q[cand_t]
    y = torch.zeros_like(logits)
    y[torch.from_numpy(np.repeat(np.arange(len(cnt)), cnt)), torch.from_numpy(inv[:len(pos)])] = \
        torch.from_numpy(1.0 / np.repeat(cnt, cnt)).float()
    return -(y * torch.log_softmax(logits, 1)).sum() / len(cnt)


def make_batches(graphs: GraphData, batch_size: int, rng):
    sizes = np.diff(graphs.n_ptr)
    order = np.lexsort((rng.random(len(sizes)), sizes))
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


def fit(model, graphs, pos_ptr, pos_val, neg_probs, log_q, epochs, lr=1e-3, batch_size=256, n_neg=2048,
        seed=0, on_epoch=None, patience=None, max_batches=None):
    torch.manual_seed(seed)
    sparse = {id(p): p for n, p in model.named_parameters() if n.startswith(("item_in", "item_out", "bias"))}
    dense = [p for p in model.parameters() if id(p) not in sparse]
    opts = [torch.optim.SparseAdam(list(sparse.values()), lr=lr), torch.optim.Adam(dense, lr=lr)]
    history, best, best_state, bad = [], -np.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        gen = torch.Generator().manual_seed(seed + epoch)
        t0, losses = time.perf_counter(), []
        for i, rows in enumerate(make_batches(graphs, batch_size, rng)):
            if max_batches is not None and i >= max_batches:
                break
            s = model(*graphs.batch(rows))
            loss = sampled_ce(model, s, pos_ptr, pos_val, rows, neg_probs, log_q, n_neg, gen)
            for o in opts:
                o.zero_grad()
            loss.backward()
            for o in opts:
                o.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "train_seconds": round(time.perf_counter() - t0, 1)}
        if on_epoch is not None:
            score = on_epoch(epoch)
            row["score"] = score
            if score > best:
                best, bad, row["best"] = score, 0, True
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        history.append(row)
        if patience is not None and bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def session_vectors(model, graphs: GraphData, batch_size: int = 1024):
    model.eval()
    out = torch.empty(len(graphs), model.out_dim)
    sizes = np.diff(graphs.n_ptr)
    order = np.argsort(sizes, kind="stable")
    for i in range(0, len(order), batch_size):
        rows = order[i:i + batch_size]
        out[torch.from_numpy(rows)] = model(*graphs.batch(rows))
    return out


def negative_distribution(pos_val, n_items, alpha=0.5):
    counts = np.bincount(pos_val, minlength=n_items).astype(float)
    counts[[PAD_IDX, UNK_IDX]] = 0
    p = counts ** alpha
    p /= p.sum()
    return torch.from_numpy(p).float(), torch.from_numpy(np.log(np.maximum(p, 1e-12))).float()

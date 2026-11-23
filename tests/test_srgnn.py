import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from rr.samples import build_vocab
from rr.srgnn import GraphData, SRGNN, fit, negative_distribution, session_vectors, target_csr
from rr.gru import idx_to_itemid, topn_items


def prefix(rows):
    return pd.DataFrame(rows, columns=["query_id", "itemid", "event", "distance"])


def test_graph_edges_between_groups_only():
    # query 0: g1 {A}, g2 {B, C} (same timestamp), g3 {A}  -> distances 2, 1, 0
    p = prefix([(0, 1, "view", 2), (0, 2, "view", 1), (0, 3, "addtocart", 1), (0, 1, "view", 0), (0, 2, "view", 1)])
    vocab = build_vocab(pd.DataFrame({"itemid": [1, 2, 3]}))
    g = GraphData(p, [0], vocab)
    items = g.n_item.tolist()                       # node order by item_idx: A, B, C
    assert items == [2, 3, 4]
    edges = set(zip(g.e_src.tolist(), g.e_dst.tolist()))
    assert edges == {(0, 1), (0, 2), (1, 0), (2, 0)}  # A->B, A->C, B->A, C->A; no B<->C
    assert g.l_val.tolist() == [0]                  # last group = {A}
    assert g.n_beh[2].tolist() == [0, 1, 0]         # C only added to cart
    node, beh, mask, last, a_in, a_out = g.batch(np.array([0]))
    torch.testing.assert_close(a_out[0, 0], torch.tensor([0, 0.5, 0.5]))
    torch.testing.assert_close(a_in[0, 0], torch.tensor([0, 0.5, 0.5]))


def test_oov_items_share_unk_and_targets_skip_oov():
    p = prefix([(5, 99, "view", 0)])
    vocab = build_vocab(pd.DataFrame({"itemid": [1]}))
    g = GraphData(p, [5], vocab)
    assert g.n_item.tolist() == [1]  # UNK
    ptr, val = target_csr(pd.DataFrame({"query_id": [5, 5], "itemid": [1, 77]}), [5], vocab)
    assert ptr.tolist() == [0, 1] and val.tolist() == [2]


def test_srgnn_learns_transition():
    torch.manual_seed(0)
    rows, tgt = [], []
    for q in range(200):
        a = q % 2  # item 1 -> 2, item 3 -> 4
        rows.append((q, 1 + 2 * a, "view", 0)); tgt.append((q, 2 + 2 * a))
    vocab = build_vocab(pd.DataFrame({"itemid": [1, 2, 3, 4, 5, 6]}))
    g = GraphData(prefix(rows), np.arange(200), vocab)
    ptr, val = target_csr(pd.DataFrame(tgt, columns=["query_id", "itemid"]), np.arange(200), vocab)
    n_items = int(vocab["item_idx"].max()) + 1
    neg, lq = negative_distribution(np.arange(2, n_items), n_items)
    model = SRGNN(n_items, dim=16)
    hist = fit(model, g, ptr, val, neg, lq, epochs=30, lr=0.02, batch_size=32, n_neg=4, seed=1)
    assert hist[-1]["loss"] < hist[0]["loss"]
    items, _ = topn_items(model, session_vectors(model, g), idx_to_itemid(vocab), n=1)
    assert items[:2, 0].tolist() == [2, 4]

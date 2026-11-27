import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from rr.gru import GRU4Rec, GroupData, fit, idx_to_itemid, query_states, target_lists, topn_items
from rr.samples import build_vocab


def toy_events():
    # session 1: g1 {10 view}, g2 {11 view, 12 addtocart}, g3 {13 view}; session 2: g1 {10}, g2 {99 OOV}
    rows = [(1, 1, 10, "view"), (1, 2, 11, "view"), (1, 2, 12, "addtocart"), (1, 3, 13, "view"),
            (2, 1, 10, "view"), (2, 2, 99, "view"), (3, 1, 13, "view")]
    return pd.DataFrame(rows, columns=["session_id", "group_index", "itemid", "event"])


def test_group_data_and_target_shift():
    ev = toy_events()
    vocab = build_vocab(ev[ev["itemid"] != 99])
    data = GroupData(ev, vocab)
    assert data.s_len.tolist() == [3, 2, 1]
    win = data.windows(None)
    assert win.tolist() == [[0, 3], [3, 2]]  # single-group session 3 dropped
    b, t, tgt, lo, cnt = target_lists(data, win[:, 0], win[:, 1])
    # step t of window b predicts group t+1: session 1 -> groups 1, 2; session 2 -> group 4 (OOV only)
    assert tgt.tolist() == [1, 2, 4] and cnt.tolist() == [2, 1, 0]
    item_of = dict(zip(vocab["item_idx"], vocab["itemid"]))
    assert sorted(item_of[i] for i in data.i_val[lo[0]:lo[0] + cnt[0]]) == [11, 12]


def test_windows_overlap_one_group():
    ev = pd.DataFrame({"session_id": 1, "group_index": np.arange(1, 8), "itemid": np.arange(1, 8), "event": "view"})
    data = GroupData(ev, build_vocab(ev))
    assert data.windows(3).tolist() == [[0, 3], [2, 3], [4, 3]]
    assert sum(n - 1 for _, n in data.windows(3)) == 6  # every next-group target appears exactly once


def test_states_are_independent_of_batching_and_sessions():
    torch.manual_seed(0)
    ev = toy_events()
    vocab = build_vocab(ev)
    data = GroupData(ev, vocab)
    model = GRU4Rec(int(vocab["item_idx"].max()) + 1, dim=8, hidden=6)
    q = pd.DataFrame({"session_id": [1, 1, 2], "prefix_len_groups": [1, 2, 1]})
    seq = query_states(model, data, q, mode="sequential")
    per = query_states(model, data, q, mode="per_query", batch_size=1)
    torch.testing.assert_close(seq, per)
    # hidden state reset: session 2 prefix {10} equals session 1 prefix {10}
    torch.testing.assert_close(seq[0], seq[2])
    last = query_states(model, data, q, mode="last_group")
    torch.testing.assert_close(last[0], seq[0])  # prefix of length 1 is its own last group


def test_loss_decreases_and_learns_transition():
    torch.manual_seed(0)
    rows = []
    for s in range(60):
        rows += [(s, 1, 1, "view"), (s, 2, 2, "view"), (s, 3, 3, "view")]
    ev = pd.DataFrame(rows, columns=["session_id", "group_index", "itemid", "event"])
    extra = pd.DataFrame({"session_id": 999, "group_index": np.arange(1, 9), "itemid": np.arange(4, 12), "event": "view"})
    ev = pd.concat([ev, extra], ignore_index=True)
    vocab = build_vocab(ev)
    data = GroupData(ev, vocab)
    model = GRU4Rec(int(vocab["item_idx"].max()) + 1, dim=16, hidden=16)
    hist = fit(model, data, data.windows(None), epochs=30, lr=0.03, batch_size=16, n_neg=4, seed=1)
    assert hist[-1]["loss"] < hist[0]["loss"] * 0.5
    q = pd.DataFrame({"session_id": [0, 0], "prefix_len_groups": [1, 2]})
    items, _ = topn_items(model, query_states(model, data, q), idx_to_itemid(vocab), n=1)
    assert items[:, 0].tolist() == [2, 3]


def test_behavior_embedding_switch():
    """Item-only GRU4Rec ignores behaviors; the multi-behavior model separates them."""
    from rr.gru import build_from_config
    ev_view = pd.DataFrame({"session_id": [1, 1], "group_index": [1, 2], "itemid": [10, 11], "event": ["view", "view"]})
    ev_cart = ev_view.assign(event=["addtocart", "view"])
    vocab = build_vocab(ev_view)
    q = pd.DataFrame({"session_id": [1], "prefix_len_groups": [1]})
    n = int(vocab["item_idx"].max()) + 1
    for use_behavior in [False, True]:
        torch.manual_seed(0)
        model = GRU4Rec(n, hidden=8, constrained=True, use_behavior=use_behavior).eval()
        assert (model.beh_in is not None) == use_behavior
        a = query_states(model, GroupData(ev_view, vocab), q)
        b = query_states(model, GroupData(ev_cart, vocab), q)
        assert torch.allclose(a, b) == (not use_behavior)
    cfg = {"n_items": n, "hidden": 8, "dropout": 0.0, "constrained": True}
    assert build_from_config(cfg).use_behavior  # configs saved before the switch keep behaviors
    assert not build_from_config({**cfg, "use_behavior": False}).use_behavior


@pytest.mark.parametrize("loss,constrained", [("bpr-max", False), ("cross-entropy", True), ("bpr-max", True)])
def test_other_losses_and_constrained_embedding_learn(loss, constrained):
    torch.manual_seed(0)
    rows = []
    for s in range(60):
        rows += [(s, 1, 1, "view"), (s, 2, 2, "view"), (s, 3, 3, "view")]
    ev = pd.DataFrame(rows, columns=["session_id", "group_index", "itemid", "event"])
    extra = pd.DataFrame({"session_id": 999, "group_index": np.arange(1, 9), "itemid": np.arange(4, 12), "event": "view"})
    ev = pd.concat([ev, extra], ignore_index=True)
    vocab = build_vocab(ev)
    data = GroupData(ev, vocab)
    model = GRU4Rec(int(vocab["item_idx"].max()) + 1, dim=16, hidden=16, constrained=constrained)
    if constrained:
        assert model.item_out is model.item_in
    hist = fit(model, data, data.windows(None), epochs=40, lr=0.03, batch_size=16, n_neg=6, seed=1, loss=loss)
    assert hist[-1]["loss"] < hist[0]["loss"]
    q = pd.DataFrame({"session_id": [0, 0], "prefix_len_groups": [1, 2]})
    items, _ = topn_items(model, query_states(model, data, q), idx_to_itemid(vocab), n=1)
    assert items[:, 0].tolist() == [2, 3]


def test_behavior_embedding_same_scale_as_items():
    """The behavior part of the step input must not swamp the item mean at initialisation."""
    torch.manual_seed(0)
    model = GRU4Rec(1000, hidden=224, constrained=True)
    item_std, beh_std = model.item_in.weight.std().item(), model.beh_in.weight.std().item()
    assert 0.5 < beh_std / item_std < 2

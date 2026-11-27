from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from rr.official_gru import OfficialTrainer, load_official, query_states, to_official_frame, topn_items

CODE = Path(__file__).resolve().parents[1] / "external/GRU4Rec_PyTorch_Official"
pytestmark = pytest.mark.skipif(not (CODE / "gru4rec_pytorch.py").exists(), reason="official GRU4Rec not fetched (notebook 13)")

PARAMS = dict(loss="bpr-max", constrained_embedding=True, embedding=0, elu_param=0.5, layers=[8], n_epochs=1,
              batch_size=4, dropout_p_embed=0.5, dropout_p_hidden=0.05, learning_rate=0.05, momentum=0.4,
              n_sample=4, sample_alpha=0.4, bpreg=1.95, logq=0.0)


def toy_events(n_sessions=12):
    rows, eid = [], 0
    rng = np.random.default_rng(0)
    for s in range(1, n_sessions + 1):
        for g, t in enumerate(range(1, 5), start=1):
            for _ in range(1 + (g == 2)):          # group 2 holds two events with the same timestamp
                rows.append((eid, s, g, int(rng.integers(10, 20)), 100 * s + t, "view"))
                eid += 1
    return pd.DataFrame(rows, columns=["event_id", "session_id", "group_index", "itemid", "timestamp", "event"])


def test_frame_orders_same_timestamp_by_event_id():
    ev = pd.DataFrame({"event_id": [5, 3, 4], "session_id": [1, 1, 1], "group_index": [1, 1, 2],
                       "itemid": [50, 30, 40], "timestamp": [10, 10, 20]})
    assert to_official_frame(ev)["ItemId"].tolist() == [30, 50, 40]


def test_epoch_loop_equals_official_fit():
    gm, _ = load_official(CODE)
    train = to_official_frame(toy_events())
    ours = OfficialTrainer(gm, PARAMS, train.copy(), seed=7, sample_cache_max_size=400)
    hist = ours.run_epoch()
    torch.manual_seed(7)
    np.random.seed(7)
    ref = gm.GRU4Rec(**PARAMS, device=torch.device("cpu"))
    ref.fit(train.copy(), sample_cache_max_size=400)
    assert hist["epoch"] == 1 and np.isfinite(hist["loss"])
    for a, b in zip(ours.gru.model.state_dict().values(), ref.model.state_dict().values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def naive_state(gru, items):
    H = [torch.zeros((1, k)) for k in gru.layers]
    imap = gru.data_iterator.itemidmap
    with torch.no_grad():
        for i in items:
            if i in imap.index:
                E, _, _ = gru.model.embed(torch.tensor([imap[i]], dtype=torch.int64), H, None)
                gru.model.hidden_step(E, H, training=False)
    return H[-1][0]


def test_query_states_match_one_query_at_a_time():
    gm, _ = load_official(CODE)
    ev = toy_events()
    trainer = OfficialTrainer(gm, PARAMS, to_official_frame(ev), seed=1, sample_cache_max_size=400)
    trainer.run_epoch()
    gru = trainer.gru
    val = pd.DataFrame({"event_id": [100, 102, 101, 103, 104], "session_id": [50, 50, 50, 50, 51],
                        "group_index": [1, 2, 2, 3, 1], "itemid": [11, 999, 12, 13, 998], "timestamp": [1, 2, 2, 3, 1]})
    q = pd.DataFrame({"session_id": [50, 50, 50, 51], "prefix_len_groups": [1, 2, 3, 1]})
    states = query_states(gru, val, q)
    torch.testing.assert_close(states[0], naive_state(gru, [11]))
    torch.testing.assert_close(states[1], naive_state(gru, [11, 12]))       # OOV 999 skipped, event_id order
    torch.testing.assert_close(states[2], naive_state(gru, [11, 12, 13]))
    assert torch.count_nonzero(states[3]) == 0                              # prefix without known items
    items, scores = topn_items(gru, states, n=3)
    full = (states @ gru.model.Wy.weight.T + gru.model.By.weight.T).detach().numpy()
    ids = gru.data_iterator.itemidmap.index.to_numpy()
    assert items[:3].tolist() == ids[np.argsort(-full[:3], axis=1, kind="stable")[:, :3]].tolist()
    np.testing.assert_allclose(scores[:3], np.sort(full[:3], axis=1)[:, ::-1][:, :3], rtol=1e-6)

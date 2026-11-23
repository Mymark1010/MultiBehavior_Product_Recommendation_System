"""Top-K recommendation for one raw prefix, using the artifacts fitted on train (T14 demo).

A prefix is a list of ``(itemid, event, timestamp_ms)``. Events with the same timestamp form one
group (no order inside a group); the query time is the last timestamp. The result goes through
the same candidate filter and Popularity fallback as the evaluation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rr.candidates import CandidateFilter, finalize_topk
from rr.evaluation import fit_neighbour_model, recommend_neighbour
from rr.io import read_json
from rr.models import popularity
from rr.samples import BEHAVIORS, asof_property
from rr import gru as G


def prefix_frame(prefix, query_id: int = 0) -> tuple[pd.DataFrame, int]:
    """Long prefix table (query_id, itemid, event, distance, group_index) and the query time."""
    f = pd.DataFrame(list(prefix), columns=["itemid", "event", "timestamp"])
    if f.empty:
        raise ValueError("Prefix rỗng.")
    if not f["event"].isin(BEHAVIORS).all():
        raise ValueError(f"event phải thuộc {BEHAVIORS}.")
    f["itemid"] = f["itemid"].astype(np.int64)
    f["timestamp"] = f["timestamp"].astype(np.int64)
    f["group_index"] = f["timestamp"].rank(method="dense").astype(np.int64)
    f["distance"] = f["group_index"].max() - f["group_index"]
    f["query_id"] = query_id
    return f, int(f["timestamp"].max())


class Recommender:
    def __init__(self, root: Path, model: str = "knn", width: int = 200):
        root = Path(root)
        protocol = read_json(root / "configs/protocol.json")
        processed = root / "data/processed"
        self.width, self.max_groups = width, protocol["knn"]["max_prefix_groups"]
        self.vocab = pd.read_parquet(processed / "item_vocab.parquet")
        self.history = pd.read_parquet(processed / "item_property_history.parquet")
        catalog = pd.read_parquet(processed / "item_catalog.parquet")
        self.filter = CandidateFilter.from_frames(self.vocab, catalog, self.history, "known_at_train_ms")
        events = pd.read_parquet(processed / "events_sessionized.parquet",
                                 columns=["session_id", "group_index", "itemid", "event", "timestamp", "split"])
        train = events[events["split"].astype(str) == "train"]
        pop = read_json(root / "reports/eval/popularity_validation.json")["selected"]["fallback_weights"]
        n = protocol["candidates"]["fallback_top_n"]
        self.fb_items, self.fb_scores = (a[:n] for a in popularity(train, pop))
        self.kind = model
        if model == "knn":
            knn = read_json(root / "reports/eval/knn_validation.json")
            sel = knn["selected"]
            self.family = max(sel, key=lambda f: knn["runs"][sel[f]["best_run"]]["all"]["ndcg@20"])
            self.selected = sel
            self.model = fit_neighbour_model(self.family, sel, train)
        elif model == "gru4rec":
            cfg = read_json(root / "models/gru4rec/config.json")
            self.model = G.build_from_config(cfg)
            self.model.load_state_dict(torch.load(root / "models/gru4rec/model.pt"))
            self.model.eval()
            self.idx_item = G.idx_to_itemid(self.vocab)
        else:
            raise ValueError(model)

    def candidates(self, f: pd.DataFrame, t: int):
        if self.kind == "knn":
            p = f[f["distance"] < self.max_groups]
            return recommend_neighbour(self.model, self.family, self.selected, p, [0], [t], self.width)
        ev = f.assign(session_id=0)[["session_id", "group_index", "itemid", "event"]]
        data = G.GroupData(ev, self.vocab)
        q = pd.DataFrame({"session_id": [0], "prefix_len_groups": [int(f["group_index"].max())]})
        return G.topn_items(self.model, G.query_states(self.model, data, q), self.idx_item, self.width)

    def recommend(self, prefix, k: int = 20) -> pd.DataFrame:
        f, t = prefix_frame(prefix)
        items, scores = self.candidates(f, t)
        topk, sc, fb = finalize_topk(items, np.array([t]), self.filter, self.fb_items, k, scores, self.fb_scores)
        keep = topk[0] != -1
        out = pd.DataFrame({"rank": np.arange(1, keep.sum() + 1), "item_id": topk[0][keep], "score": sc[0][keep],
                            "is_fallback": fb[0][keep]})
        cat = asof_property(pd.DataFrame({"itemid": out["item_id"], "query_time_ms": t}), self.history, "categoryid")
        out.insert(3, "category_id", cat["categoryid_asof"].to_numpy())
        return out

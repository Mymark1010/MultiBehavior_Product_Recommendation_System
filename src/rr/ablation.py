"""Session thresholds and data ablations (notebooks 09, 10 and 12).

Nothing here changes the locked protocol: every function returns new in-memory frames and the
main pipeline (notebooks 01-10) keeps reading its own artifacts.

* :func:`sessionize` repeats notebook 02 (sessions, timestamp groups, whole-session split with the
  same cutoffs) for any inactivity threshold; with 30 minutes it must reproduce notebook 02 exactly.
* :func:`merge_repeated_groups` is the D1 optional ablation: a group whose set of
  ``(itemid, event)`` pairs equals the previous group of the same session is dropped.
* :func:`heavy_visitor_thresholds` / :func:`flag_heavy_visitors` define abnormal visitors from the
  per-(visitor, UTC day) activity of the train split only.
* :func:`score_models` fits the frozen configuration of each model on a frame of fitting events and
  scores one evaluation sample, through the same filter/fallback/metrics as notebook 10.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from rr.samples import BEHAVIORS, build_queries, build_targets

SPLIT_NAMES = ["train", "validation", "test", "boundary_drop"]
DAY_MS = 86_400_000
BEH_CODE = {b: i for i, b in enumerate(BEHAVIORS)}


# ---------------------------------------------------------------- sessions and split
def make_session_ids(frame: pd.DataFrame, inactivity_ms: int) -> pd.Series:
    """Same rule as notebook 02; ``frame`` sorted by visitorid, timestamp, event_id."""
    visitor_changes = frame["visitorid"].ne(frame["visitorid"].shift(fill_value=-1))
    gaps = frame.groupby("visitorid", sort=False)["timestamp"].diff()
    starts = (visitor_changes | gaps.gt(inactivity_ms)).fillna(True)
    return starts.cumsum().astype("int64")


def assign_session_splits(start_ms, end_ms, cut80: int, cut90: int) -> np.ndarray:
    """Whole-session split of notebook 02; sessions crossing a cutoff -> boundary_drop."""
    start_ms, end_ms = np.asarray(start_ms), np.asarray(end_ms)
    labels = np.full(len(start_ms), "", dtype=object)
    labels[end_ms <= cut80] = "train"
    labels[(start_ms > cut80) & (end_ms <= cut90)] = "validation"
    labels[start_ms > cut90] = "test"
    boundary = np.zeros(len(start_ms), dtype=bool)
    for cut in (cut80, cut90):
        boundary |= (start_ms <= cut) & (end_ms > cut)
    labels[boundary] = "boundary_drop"
    if (labels == "").any():
        raise ValueError("Có phiên chưa được gán split.")
    return labels


def sessionize(events_clean: pd.DataFrame, inactivity_ms: int, cutoffs) -> pd.DataFrame:
    """Sessions, timestamp groups and split of the clean events (columns as events_sessionized)."""
    cut80, cut90 = (int(c) for c in cutoffs)
    e = events_clean[["event_id", "timestamp", "visitorid", "event", "itemid"]]
    e = e.sort_values(["visitorid", "timestamp", "event_id"], kind="stable").reset_index(drop=True)
    e["session_id"] = make_session_ids(e, inactivity_ms)
    new_group = (e["session_id"].ne(e["session_id"].shift(fill_value=-1))
                 | e["timestamp"].ne(e["timestamp"].shift(fill_value=-1)))
    e["group_id"] = new_group.cumsum().astype("int64")
    first = new_group.to_numpy()
    group_index = pd.Series(np.where(first, 1, 0)).groupby(e["session_id"].to_numpy()).cumsum()
    e["group_index"] = group_index.to_numpy().astype("int32")
    span = e.groupby("session_id", sort=False)["timestamp"].agg(["min", "max"])
    labels = pd.Series(assign_session_splits(span["min"], span["max"], cut80, cut90), index=span.index)
    e["split"] = pd.Categorical(labels.reindex(e["session_id"]).to_numpy(), categories=SPLIT_NAMES)
    return e


def groups_of(events: pd.DataFrame) -> pd.DataFrame:
    """One row per timestamp group (input of :func:`rr.samples.build_queries`)."""
    g = events.drop_duplicates("group_id")[["group_id", "session_id", "visitorid", "timestamp", "group_index", "split"]]
    return g.sort_values(["session_id", "group_index"]).reset_index(drop=True)


def split_summary(events: pd.DataFrame) -> dict:
    out = {}
    for name in SPLIT_NAMES:
        part = events[events["split"] == name]
        groups = part.groupby("session_id")["group_index"].max()
        out[name] = {"sessions": int(len(groups)), "events": int(len(part)), "timestamp_groups": int(groups.sum()),
                     "prefix_target_pairs": int((groups - 1).sum())}
    return out


# ---------------------------------------------------------------- repeated groups
def group_signatures(events: pd.DataFrame) -> pd.Series:
    """Exact signature of each group: sorted distinct ``itemid*4 + behavior code`` joined as text."""
    key = events["itemid"].to_numpy(dtype=np.int64) * 4 + events["event"].astype(str).map(BEH_CODE).to_numpy(dtype=np.int64)
    pairs = pd.DataFrame({"g": events["group_id"].to_numpy(), "k": key}).drop_duplicates().sort_values(["g", "k"])
    sizes = pairs.groupby("g").size()
    single = pairs[pairs["g"].isin(sizes.index[sizes == 1])].set_index("g")["k"].astype(str)
    multi = (pairs[pairs["g"].isin(sizes.index[sizes > 1])].groupby("g")["k"]
             .agg(lambda k: ",".join(map(str, k))))
    return pd.concat([single, multi]).sort_index()


def merge_repeated_groups(events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop every group whose (item, behavior) set equals the previous group of the same session.

    The first group of a run is kept (its timestamp is the first time the set was seen); later
    groups of the session are renumbered so that ``group_index`` stays 1..n without gaps.
    """
    g = groups_of(events)
    sig = group_signatures(events).reindex(g["group_id"]).to_numpy()
    same_session = g["session_id"].eq(g["session_id"].shift()).to_numpy()
    dropped = same_session & (sig == np.roll(sig, 1))
    keep_groups = g.loc[~dropped, ["group_id", "session_id"]].copy()
    keep_groups["new_index"] = (keep_groups.groupby("session_id").cumcount() + 1).astype("int32")
    out = events[events["group_id"].isin(keep_groups["group_id"])].copy()
    out["group_index"] = out["group_id"].map(keep_groups.set_index("group_id")["new_index"]).astype("int32")
    stats = {"groups_before": int(len(g)), "groups_dropped": int(dropped.sum()),
             "events_dropped": int(len(events) - len(out)),
             "groups_dropped_by_split": {str(k): int(v) for k, v in
                                         g.loc[dropped, "split"].astype(str).value_counts().items()}}
    return out, stats


# ---------------------------------------------------------------- samples
def build_samples(events: pd.DataFrame, split: str, first_query_id: int = 0):
    """Queries and targets of one split (targets carry the flags :func:`rr.metrics.evaluate` uses)."""
    part = events[events["split"] == split]
    q = build_queries(groups_of(part), split, first_query_id)
    t = build_targets(q, part)
    t["has_strong_behavior"] = t["behaviors"].map(lambda b: "addtocart" in b or "transaction" in b)
    return q, t[["query_id", "itemid", "target_is_repeat", "has_strong_behavior"]]


def known_at_for(catalog: pd.DataFrame, fit_events: pd.DataFrame, column: str = "known_at_fit_ms") -> pd.DataFrame:
    """Catalog with ``known_at = min(first fitting event, first property record)`` (protocol rule)."""
    first = fit_events.groupby("itemid")["timestamp"].min()
    out = catalog[["itemid", "first_property_ms"]].copy()
    ev = first.reindex(out["itemid"].to_numpy()).to_numpy(dtype=float)
    prop = out["first_property_ms"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    known = np.fmin(ev, prop)
    out[column] = pd.array(np.where(np.isnan(known), pd.NA, known), dtype="Int64")
    return out


# ---------------------------------------------------------------- heavy visitors
def visitor_day_activity(events: pd.DataFrame) -> pd.DataFrame:
    """Events and transactions per (visitor, UTC day)."""
    frame = pd.DataFrame({"visitorid": events["visitorid"].to_numpy(),
                          "day": events["timestamp"].to_numpy() // DAY_MS,
                          "tx": (events["event"].astype(str) == "transaction").to_numpy()})
    return frame.groupby(["visitorid", "day"]).agg(events=("tx", "size"), transactions=("tx", "sum")).reset_index()


def heavy_visitor_thresholds(train_events: pd.DataFrame, quantile: float) -> dict:
    act = visitor_day_activity(train_events)
    return {"quantile": quantile, "events_per_day": float(act["events"].quantile(quantile)),
            "transactions_per_day": float(act["transactions"].quantile(quantile))}


def flag_heavy_visitors(events: pd.DataFrame, thresholds: dict) -> np.ndarray:
    """Visitors with, on some UTC day of ``events``, more events or transactions than the thresholds."""
    act = visitor_day_activity(events)
    heavy = (act["events"] > thresholds["events_per_day"]) | (act["transactions"] > thresholds["transactions_per_day"])
    return np.sort(act.loc[heavy, "visitorid"].unique())


def topk_matrix(long: pd.DataFrame, query_ids, k: int = 20) -> np.ndarray:
    """Long ``query_id, rank, itemid`` table -> [n_queries, k] matrix aligned with ``query_ids`` (-1 padding)."""
    out = np.full((len(query_ids), k), -1, dtype=np.int64)
    row = pd.Series(np.arange(len(query_ids)), index=np.asarray(query_ids)).reindex(long["query_id"].to_numpy())
    if row.isna().any():
        raise ValueError("Danh sách Top-K chứa query ngoài bộ query.")
    out[row.to_numpy(dtype=np.int64), long["rank"].to_numpy() - 1] = long["itemid"].to_numpy()
    return out


# ---------------------------------------------------------------- model scoring
NON_NEURAL = ["popularity", "repeat_last", "itemknn", "sknn", "stan", "vstan"]


def score_models(fit_events: pd.DataFrame, eval_events: pd.DataFrame, eval_q: pd.DataFrame, eval_t: pd.DataFrame,
                 catalog: pd.DataFrame, history: pd.DataFrame, known_at_column: str, cfg: dict, neural: str,
                 neural_models=None, log=print, keep_topk=()):
    """Fit the frozen configuration of each model on ``fit_events`` and score ``eval_q``.

    ``cfg``: ``popularity``, ``knn_selected`` (frozen configs), ``fallback_top_n``,
    ``max_groups``, ``max_len``, ``seed``, ``width``, ``root`` and optionally
    ``models`` (names to score, default every name of ``NON_NEURAL`` + GRU variants),
    ``gru_variants`` (``{name: config}`` of GRU4Rec models, default ``{"gru4rec": cfg["gru"]}``) and
    ``model_dir`` (folder with ``<name>/model.pt``, default ``root/models``).
    ``neural``: ``"load"`` (saved models, vocabulary must equal theirs), ``"train"`` (retrain with the selected
    epochs on ``fit_events``),
    ``"reuse"`` (``neural_models`` from an earlier call with the same vocabulary) or ``"skip"``.
    Returns ``(results, models)``; ``results[name] = {"report", "per_query", "fit_seconds", "query_seconds"}``,
    plus ``"topk"`` = (Top-K item matrix, is_fallback matrix) for the names in ``keep_topk``.
    """
    import torch

    from rr import gru as G
    from rr.candidates import CandidateFilter
    from rr.evaluation import fit_neighbour_model, recommend_neighbour, run
    from rr.models import popularity, repeat_last
    from rr.samples import build_vocab, prefix_events

    if fit_events["timestamp"].max() >= eval_q["query_time_ms"].min():
        raise ValueError("Dữ liệu fit phải nằm trước mọi query đánh giá.")
    gru_variants = cfg.get("gru_variants") or ({"gru4rec": cfg["gru"]} if "gru" in cfg else {})
    wanted = list(cfg.get("models") or NON_NEURAL + list(gru_variants))
    unknown = set(wanted) - set(NON_NEURAL) - set(gru_variants)
    if unknown:
        raise ValueError(f"Mô hình không biết: {sorted(unknown)}")
    width, seed = cfg["width"], cfg["seed"]
    model_dir = cfg.get("model_dir") or cfg["root"] / "models"
    qids, qtimes = eval_q["query_id"].to_numpy(), eval_q["query_time_ms"].to_numpy()
    vocab = build_vocab(fit_events)
    flt = CandidateFilter.from_frames(vocab, catalog, history, known_at_column)
    top_n = cfg["fallback_top_n"]
    fb_items, fb_scores = (a[:top_n] for a in popularity(fit_events, cfg["popularity"]["fallback_weights"]))
    pop_items, pop_scores = (a[:top_n] for a in popularity(fit_events, cfg["popularity"]["popularity_behavior_weights"]))
    prefix_full = prefix_events(eval_q, eval_events, ("itemid", "event"))
    prefix = prefix_full[prefix_full["distance"] < cfg["max_groups"]]
    lists, fit_s, query_s = {}, {}, {}

    def timed(name, fit_fn, rec_fn):
        t0 = time.perf_counter(); model = fit_fn(); fit_s[name] = time.perf_counter() - t0
        t0 = time.perf_counter(); lists[name] = rec_fn(model); query_s[name] = time.perf_counter() - t0
        log(f"  {name}: fit {fit_s[name]:.0f}s, query {query_s[name]:.0f}s")
        return model

    empty = np.empty((len(eval_q), 0), dtype=np.int64)
    if "popularity" in wanted:
        lists["popularity"], fit_s["popularity"], query_s["popularity"] = (empty, None), 0.0, 0.0
    if "repeat_last" in wanted:
        timed("repeat_last", lambda: None, lambda m: repeat_last(prefix_full, qids, width=50))
    sel = cfg.get("knn_selected", {})
    for fam in [f for f in ["itemknn", "sknn", "stan", "vstan"] if f in wanted]:
        timed(fam, lambda fam=fam: fit_neighbour_model(fam, sel, fit_events),
              lambda m, fam=fam: recommend_neighbour(m, fam, sel, prefix, qids, qtimes, width))

    models = dict(neural_models or {})
    neural_wanted = [n for n in wanted if n in gru_variants]
    if neural != "skip" and neural_wanted:
        n_items = int(vocab["item_idx"].max()) + 1
        if neural == "reuse" and n_items != models["n_items"]:
            raise ValueError("Vocabulary khác vocabulary của mô hình dùng lại.")

        def fit_gru(name, gcfg):
            if neural == "reuse":
                return models[name]
            if neural == "load":
                if n_items != gcfg["n_items"]:
                    raise ValueError(f"{name}: vocabulary khác vocabulary của mô hình đã lưu.")
                m = G.build_from_config(gcfg, n_items=n_items)
                m.load_state_dict(torch.load(model_dir / name / "model.pt"))
                return m
            torch.manual_seed(seed)
            m = G.build_from_config(gcfg, n_items=n_items)
            data = G.GroupData(fit_events, vocab)
            G.fit(m, data, data.windows(cfg["max_len"]), gcfg["epochs"], lr=gcfg["lr"], batch_size=gcfg["batch_size"],
                  n_neg=gcfg["n_neg"], seed=seed, loss=gcfg["loss"], bpreg=gcfg["bpreg"], elu=gcfg["elu"],
                  sample_alpha=gcfg["sample_alpha"])
            return m

        for name, gcfg in gru_variants.items():
            if name in wanted:
                models[name] = timed(name, lambda name=name, gcfg=gcfg: fit_gru(name, gcfg), lambda m: G.topn_items(
                    m, G.query_states(m, G.GroupData(eval_events, vocab), eval_q), G.idx_to_itemid(vocab), width))

        models["n_items"] = n_items

    results = {}
    for name, (items, scores) in lists.items():
        if name == "popularity":
            rep, pq, topk, is_fb = run(empty, None, eval_q, eval_t, flt, pop_items, pop_scores)
        else:
            rep, pq, topk, is_fb = run(items, scores, eval_q, eval_t, flt, fb_items, fb_scores)
        results[name] = {"report": rep, "per_query": pq, "fit_seconds": round(fit_s[name] or 0.0, 1),
                         "query_seconds": round(query_s[name], 1)}
        if name in keep_topk:
            results[name]["topk"] = (topk, is_fb)
    return results, models

"""Glue between models, the candidate filter and the metrics (same path for every model)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from rr.candidates import CandidateFilter, finalize_topk
from rr.metrics import evaluate, metric_columns

K_MAX = 20


def load_samples(root: Path, split: str):
    base = Path(root) / "data/processed/samples"
    queries = pd.read_parquet(base / f"{split}_queries.parquet")
    targets = pd.read_parquet(base / f"{split}_targets.parquet",
                              columns=["query_id", "itemid", "target_is_repeat", "has_strong_behavior", "in_vocab",
                                       "candidate_eligible"])
    return queries, targets


def run(candidates, cand_scores, queries: pd.DataFrame, targets: pd.DataFrame, flt: CandidateFilter,
        fallback_items, fallback_scores=None, k: int = K_MAX):
    """Finalize Top-K (filter at t, dedupe, fallback) and evaluate. Returns (report, per_query, topk, is_fallback)."""
    t0 = time.perf_counter()
    topk, scores, is_fb = finalize_topk(candidates, queries["query_time_ms"].to_numpy(), flt, fallback_items, k,
                                        cand_scores, fallback_scores)
    report, per_query = evaluate(topk, queries, targets)
    report["fallback"] = {
        "slot_rate": float(is_fb.mean()),
        "queries_with_any_fallback": float(is_fb.any(axis=1).mean()),
        "queries_fully_fallback": float(is_fb.all(axis=1).mean()),
    }
    report["short_lists"] = int((topk == -1).any(axis=1).sum())
    report["finalize_eval_seconds"] = round(time.perf_counter() - t0, 2)
    return report, per_query, topk, is_fb


def flat(reports: dict, part: str = "all", cols=None) -> pd.DataFrame:
    """Table of one report part (``all``, ``new_targets``, ...) for many named reports."""
    cols = cols or metric_columns()
    rows = {name: {c: r.get(part, {}).get(c, np.nan) for c in cols} for name, r in reports.items()}
    return pd.DataFrame(rows).T


def fit_neighbour_model(family: str, selected: dict, fit_events: pd.DataFrame):
    """Fit the selected configuration of a Session-KNN model (``knn_validation.json -> selected``)."""
    from rr.models import SessionKNN
    cfg = selected[family]
    return SessionKNN(family, weights=cfg["weights"], **cfg["structure"]).fit(fit_events)


def recommend_neighbour(model, family: str, selected: dict, prefix: pd.DataFrame, query_ids, query_times,
                        width: int = 200):
    return model.recommend(prefix, query_ids, query_times, width)

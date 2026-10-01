"""Small explicit data-access helpers for collaborators and notebooks.

Candidate construction here favors correctness over online serving latency.
Large-scale serving should maintain catalog state incrementally.
"""
from pathlib import Path

import pandas as pd

from scripts.prepare_multibehavior import lookup_metadata


def require_complete(directory):
    directory = Path(directory)
    if not directory.is_dir() or (directory / "INCOMPLETE").exists():
        raise RuntimeError("Dataset is missing or incomplete. Run scripts/prepare_multibehavior.py successfully first.")
    return directory


def history_for_query(events, query):
    """Return observed history; caller decides a maximum sequence length."""
    return events.loc[
        events.visitorid.eq(query["visitorid"])
        & events.event_time.le(query["query_time"])
        & events.split.ne("boundary_drop")
    ].sort_values(["event_time", "event_id"])


def candidates_at_time(directory, prediction_time, config, warm_only=False):
    """All eligible catalog items, independent of query targets.

    warm_only=True restricts candidates to items in the train vocabulary.
    Use the same candidate policy for directly compared models and report coverage.
    Never force-include future positives in this candidate list.
    """
    directory = require_complete(directory)
    catalog = pd.read_parquet(directory / "catalog.parquet")
    properties = pd.read_parquet(directory / "property_history.parquet")
    tree = pd.read_parquet(directory / "category_tree.parquet")
    known = catalog[catalog.known_at.le(prediction_time)]
    if warm_only:
        known = known[known.seen_in_train]
    requests = known[["itemid"]].assign(query_time=prediction_time)
    result = lookup_metadata(requests, properties, catalog, tree, config)
    return result[result.candidate_at_query].reset_index(drop=True)

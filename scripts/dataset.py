"""Read prepared datasets, observed history and point-in-time catalog entries."""
import json
from pathlib import Path

import pandas as pd

try:
    from scripts.prepare_nextstep import lookup_metadata
except ModuleNotFoundError:
    from prepare_nextstep import lookup_metadata


def require_complete(directory):
    directory = Path(directory)
    if not directory.is_dir() or (directory / "INCOMPLETE").exists():
        raise RuntimeError("Dataset is missing or incomplete. Run scripts/prepare_nextstep.py successfully first.")
    manifest_path = directory / "dataset_info.json"
    info = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if info.get("task") != "next_step" or info.get("schema_version") != "2.0.0":
        raise RuntimeError("Expected a schema-v2 next-step dataset")
    return directory


def history_for_query(events, query):
    """Return the observed session prefix, including the complete query bucket."""
    return events.loc[
        events.session_id.eq(query["session_id"])
        & events.event_time.le(query["query_time"])
        & events.split.ne("boundary_drop")
    ].sort_values(["event_time", "event_id"])


def candidates_at_time(directory, prediction_time, config, warm_only=False):
    """All eligible catalog items, independent of query targets.

    warm_only=True restricts candidates to items in the train vocabulary.
    This helper describes the prepared data's candidate policy.
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

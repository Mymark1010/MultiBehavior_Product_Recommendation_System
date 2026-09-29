"""Build reproducible, time-aware multi-behavior recommendation datasets.

Run from any directory: python scripts/prepare_multibehavior.py --root PATH
The existing notebooks and their output directories are never overwritten.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

BEHAVIORS = ("view", "addtocart", "transaction")
RAW_COLUMNS = ["timestamp", "visitorid", "event", "itemid", "transactionid"]


def log(message):
    print(message, flush=True)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def save(frame, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / f"{name}.parquet", index=False)


def fingerprint(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def clean_events(raw, config):
    """Preserve every distinct behavior; original row ID only breaks storage ties."""
    data = raw.copy()
    data["event_id"] = np.arange(len(data), dtype="int64")
    data = data[data.event.isin(BEHAVIORS)].dropna(subset=RAW_COLUMNS[:4])
    data = data.drop_duplicates(RAW_COLUMNS).copy()
    for col in ("timestamp", "visitorid", "itemid"):
        data[col] = data[col].astype("int64")
    data["event_time"] = pd.to_datetime(data.timestamp, unit="ms", utc=True).astype("datetime64[ns, UTC]")
    data = data.sort_values(["visitorid", "event_time", "event_id"]).reset_index(drop=True)
    gap = data.groupby("visitorid", sort=False).event_time.diff()
    new = gap.isna() | gap.gt(pd.Timedelta(minutes=config["session_gap_minutes"]))
    data["session_id"] = new.cumsum().astype("int64")
    data["event_order"] = data.groupby("session_id", sort=False).cumcount().astype("int32")
    train_end = data.event_time.quantile(config["train_quantile"]).round("ms")
    validation_end = data.event_time.quantile(config["validation_quantile"]).round("ms")
    bounds = data.groupby("session_id", sort=False).event_time.agg(["min", "max"])
    bounds["split"] = np.select(
        [bounds["max"].le(train_end), bounds["min"].gt(train_end) & bounds["max"].le(validation_end),
         bounds["min"].gt(validation_end)], ["train", "validation", "test"], default="boundary_drop")
    data["split"] = data.session_id.map(bounds.split)
    return data, {"train_end": train_end, "validation_end": validation_end, "observation_end": data.event_time.max()}


def create_queries(events, cutoffs, config):
    """One query per session after its first timestamp bucket is observed.

    All events <= query_time are history; labels are strictly after it.
    The unit is a query, NOT one row per target/history event.
    """
    usable = events[events.split.ne("boundary_drop")]
    q = usable.groupby("session_id", sort=False).agg(
        visitorid=("visitorid", "first"), query_time=("event_time", "min"), split=("split", "first")
    ).reset_index().rename(columns={"session_id": "query_id"})
    q["session_id"] = q.query_id
    q["label_end"] = q.query_time + pd.Timedelta(days=config["label_horizon_days"])
    ends = {"train": cutoffs["train_end"], "validation": cutoffs["validation_end"], "test": cutoffs["observation_end"]}
    q["eligible"] = q.label_end.le(q.split.map(ends))
    excluded = q[~q.eligible].copy()
    excluded["exclusion_reason"] = "incomplete_label_window_within_split"
    return q[q.eligible].drop(columns="eligible").reset_index(drop=True), excluded.drop(columns="eligible")


def create_targets(queries, events):
    """Actual purchases in (query_time, label_end], with no synthetic funnel rows."""
    purchases = events[events.event.eq("transaction") & events.split.ne("boundary_drop")]
    joined = queries[["query_id", "visitorid", "query_time", "label_end", "split"]].merge(
        purchases[["visitorid", "itemid", "event_time", "split"]], on=["visitorid", "split"], how="inner")
    joined = joined[joined.event_time.gt(joined.query_time) & joined.event_time.le(joined.label_end)]
    targets = joined.groupby(["query_id", "itemid"], sort=False).agg(
        first_purchase_time=("event_time", "min"), purchase_event_count=("event_time", "size")
    ).reset_index()
    counts = targets.groupby("query_id").size()
    queries = queries.copy()
    queries["target_count"] = queries.query_id.map(counts).fillna(0).astype("int32")
    queries["has_purchase_target"] = queries.target_count.gt(0)
    return queries, targets


def build_user_features(queries, events, config):
    """Vectorized cumulative/as-of counts; no future event can affect a query."""
    # Labels and future-window metadata must never be exposed as model features.
    queries = queries[["query_id", "visitorid", "query_time"]].copy()
    queries["query_time"] = queries.query_time.astype("datetime64[ns, UTC]")
    events = events[events.split.ne("boundary_drop")].sort_values(["visitorid", "event_time", "event_id"]).copy()
    events["event_time"] = events.event_time.astype("datetime64[ns, UTC]")
    count_cols = []
    last_cols = []
    for behavior in BEHAVIORS:
        count, last = f"{behavior}_count", f"last_{behavior}_time"
        flag = events.event.eq(behavior)
        events[count] = flag.groupby(events.visitorid, sort=False).cumsum().astype("int32")
        events[last] = events.event_time.where(flag).groupby(events.visitorid, sort=False).ffill()
        count_cols.append(count)
        last_cols.append(last)
    # All simultaneous events are observed together: use the state after the entire bucket.
    states = events.drop_duplicates(["visitorid", "event_time"], keep="last")
    states = states[["visitorid", "event_time"] + count_cols + last_cols].sort_values("event_time")
    result = pd.merge_asof(queries.sort_values("query_time"), states,
                          left_on="query_time", right_on="event_time", by="visitorid", direction="backward")
    result = result.rename(columns={"event_time": "history_last_time"})
    for days in config["feature_windows_days"]:
        left = queries[["query_id", "visitorid", "query_time"]].copy()
        left["window_start"] = left.query_time - pd.Timedelta(days=days)
        before = pd.merge_asof(left.sort_values("window_start"), states[["visitorid", "event_time"] + count_cols],
                               left_on="window_start", right_on="event_time", by="visitorid", direction="backward")
        before = before.set_index("query_id")
        for behavior in BEHAVIORS:
            col = f"{behavior}_count"
            result[f"{col}_{days}d"] = (result[col] - result.query_id.map(before[col]).fillna(0)).astype("int32")
    for behavior in BEHAVIORS:
        last = f"last_{behavior}_time"
        result[f"{behavior}_recency_hours"] = (result.query_time - result[last]).dt.total_seconds() / 3600
        result[f"has_{behavior}_history"] = result[last].notna()
    return result.sort_values("query_id").reset_index(drop=True)


def build_category_tree(raw, unknown=-1):
    if raw.categoryid.isna().any() or raw.categoryid.duplicated().any():
        raise ValueError("Category IDs must be non-null and unique")
    parents = dict(zip(raw.categoryid.astype(int), raw.parentid))
    output = []
    for category in parents:
        chain, visited, current = [], set(), category
        while pd.notna(current):
            current = int(current)
            if current in visited:
                raise ValueError(f"Cycle in category tree at {category}")
            if current not in parents:
                raise ValueError(f"Unknown category parent {current}")
            visited.add(current)
            chain.append(current)
            current = parents[current]
        output.append((category, chain[-1], len(chain) - 1, "/".join(map(str, reversed(chain)))))
    details = pd.DataFrame(output, columns=["categoryid", "root_category_id", "category_depth", "category_path"])
    return raw.merge(details, on="categoryid", validate="one_to_one")


def read_properties(raw_dir, config):
    chunks, stats = [], {}
    for part in (1, 2):
        path = raw_dir / f"item_properties_part{part}.csv"
        total = 0
        for c in pd.read_csv(path, dtype={"timestamp": "int64", "itemid": "int64", "property": "string", "value": "string"},
                             chunksize=config["property_chunk_rows"]):
            total += len(c)
            c = c[c.property.isin(["categoryid", "available"])].dropna().drop_duplicates()
            chunks.append(c)
        stats[path.name] = {"raw_rows": total}
        log(f"Read {path.name}: {total:,} raw rows")
    props = pd.concat(chunks, ignore_index=True).drop_duplicates().reset_index(drop=True)
    if props.duplicated(["itemid", "property", "timestamp"]).any():
        raise ValueError("Conflicting property values at identical timestamps")
    props["property_time"] = pd.to_datetime(props.timestamp, unit="ms", utc=True).astype("datetime64[ns, UTC]")
    props["numeric_value"] = pd.to_numeric(props.value, errors="raise").astype("int64")
    if not props.loc[props.property.eq("available"), "numeric_value"].isin([0, 1]).all():
        raise ValueError("Availability must be 0 or 1")
    props = props.sort_values(["itemid", "property", "property_time"]).reset_index(drop=True)
    props["valid_to"] = props.groupby(["itemid", "property"], sort=False).property_time.shift(-1)
    return props, stats


def build_catalog(events, properties):
    event_first = events.groupby("itemid").event_time.min().rename("known_at")
    prop_first = properties.groupby("itemid").property_time.min().rename("known_at")
    return pd.concat([event_first, prop_first]).groupby(level=0).min().reset_index()


def lookup_metadata(requests, properties, catalog, tree, config):
    """requests has itemid/query_time; preserve order, use only property_time <= t."""
    result = requests.copy()
    result["query_time"] = result.query_time.astype("datetime64[ns, UTC]")
    result["_row"] = np.arange(len(result))
    result = result.merge(catalog, on="itemid", how="left", validate="many_to_one")
    for prop, name in [("available", "available"), ("categoryid", "category_id")]:
        right = properties.loc[properties.property.eq(prop), ["itemid", "property_time", "numeric_value"]].rename(
            columns={"property_time": f"{name}_source_time", "numeric_value": name})
        right[f"{name}_source_time"] = right[f"{name}_source_time"].astype("datetime64[ns, UTC]")
        result = pd.merge_asof(result.sort_values("query_time"), right.sort_values(f"{name}_source_time"),
                               left_on="query_time", right_on=f"{name}_source_time", by="itemid", direction="backward")
    result["available"] = result.available.astype("Int8")
    result["availability_unknown"] = result.available.isna()
    result["category_missing"] = result.category_id.isna()
    result["category_invalid"] = result.category_id.notna() & ~result.category_id.isin(tree.categoryid)
    invalid = result.category_missing | result.category_invalid
    result["category_id"] = result.category_id.mask(invalid, config["unknown_category_id"]).astype("int64")
    result = result.merge(tree[["categoryid", "root_category_id", "category_depth", "category_path"]],
                          left_on="category_id", right_on="categoryid", how="left", validate="many_to_one").drop(columns="categoryid")
    result["root_category_id"] = result.root_category_id.fillna(config["unknown_category_id"]).astype("int64")
    result["category_depth"] = result.category_depth.fillna(-1).astype("int16")
    result["category_path"] = result.category_path.fillna("UNKNOWN")
    result["candidate_at_query"] = result.known_at.notna() & result.known_at.le(result.query_time) & (result.available.isna() | result.available.eq(1))
    return result.sort_values("_row").drop(columns="_row").reset_index(drop=True)


def train_tables(events, cutoffs, config):
    train = events[events.split.eq("train")]
    users = pd.DataFrame({"visitorid": np.sort(train.visitorid.unique())})
    items = pd.DataFrame({"itemid": np.sort(train.itemid.unique())})
    users["user_idx"] = np.arange(2, len(users) + 2, dtype="int32")
    items["item_idx"] = np.arange(2, len(items) + 2, dtype="int32")
    grouped = train.groupby(["visitorid", "itemid", "event"], observed=True).agg(
        count=("event_id", "size"), last_event_time=("event_time", "max")).reset_index()
    counts = grouped.pivot(index=["visitorid", "itemid"], columns="event", values="count").reindex(columns=BEHAVIORS).fillna(0).astype("int32")
    counts.columns.name = None
    counts = counts.reset_index()
    counts["interaction_score"] = sum(config["interaction_weights"][b] * np.log1p(counts[b]) for b in BEHAVIORS)
    counts = counts.merge(users, on="visitorid", validate="many_to_one").merge(items, on="itemid", validate="many_to_one")
    for b in BEHAVIORS:
        last = grouped[grouped.event.eq(b)][["visitorid", "itemid", "last_event_time"]].rename(columns={"last_event_time": f"last_{b}_time"})
        counts = counts.merge(last, on=["visitorid", "itemid"], how="left", validate="one_to_one")
        counts[f"{b}_recency_hours"] = (cutoffs["train_end"] - counts[f"last_{b}_time"]).dt.total_seconds() / 3600
    # These aggregates are for a model fitted at the train cutoff, not earlier training queries.
    return users, items, counts, grouped


def sample_train_negatives(queries, targets, items, properties, catalog, tree, config):
    """Optional pairwise examples: train-only, exact-time catalog filtering.

    Viewed/carted items are allowed; a sampled negative is unpurchased in the
    horizon, NOT a known dislike. Evaluation never uses these samples.
    """
    positive_q = queries[queries.split.eq("train") & queries.has_purchase_target]
    k = config["negative_samples_per_positive_query"]
    if positive_q.empty or k == 0:
        return pd.DataFrame(columns=["query_id", "itemid", "label", "sampling_policy"])
    rng = np.random.default_rng(config["seed"])
    wanted = positive_q[["query_id", "query_time"]]
    collected = pd.DataFrame(columns=["query_id", "itemid"])
    positives = targets[["query_id", "itemid"]].assign(is_positive=True)
    known_train = items[["itemid"]].merge(catalog[["itemid", "known_at"]], on="itemid", validate="one_to_one").sort_values("known_at")
    known_times = known_train.known_at.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    known_ids = known_train.itemid.to_numpy()
    for _ in range(12):
        have = collected.groupby("query_id").size()
        pending = wanted[wanted.query_id.map(have).fillna(0).lt(k)]
        if pending.empty:
            break
        candidates = pending.loc[pending.index.repeat(k * 4)].reset_index(drop=True)
        # Sample from items already known at t, not the entire future train vocabulary.
        limits = np.searchsorted(known_times, candidates.query_time.astype("datetime64[ns, UTC]").astype("int64"), side="right")
        candidates = candidates[limits > 0].copy()
        candidates["itemid"] = known_ids[rng.integers(0, limits[limits > 0])]
        candidates = candidates.drop_duplicates(["query_id", "itemid"]).merge(positives, how="left", on=["query_id", "itemid"])
        candidates = candidates[candidates.is_positive.isna()].drop(columns="is_positive")
        candidates = lookup_metadata(candidates, properties, catalog, tree, config)
        candidates = candidates.loc[candidates.candidate_at_query, ["query_id", "itemid"]]
        collected = pd.concat([collected, candidates], ignore_index=True).drop_duplicates(["query_id", "itemid"])
        collected = collected[collected.groupby("query_id").cumcount().lt(k)]
    # Early queries can have fewer than k valid non-target items. Enumerate the
    # remaining pool instead of inventing future items or dropping the query.
    have = collected.groupby("query_id").size()
    pending = wanted[wanted.query_id.map(have).fillna(0).lt(k)]
    extras = []
    for row in pending.itertuples():
        pool = known_train.loc[known_train.known_at.le(row.query_time), ["itemid"]].assign(query_time=row.query_time)
        pool = lookup_metadata(pool, properties, catalog, tree, config)
        blocked = set(targets.loc[targets.query_id.eq(row.query_id), "itemid"]) | set(collected.loc[collected.query_id.eq(row.query_id), "itemid"])
        eligible = pool.loc[pool.candidate_at_query & ~pool.itemid.isin(blocked), "itemid"].to_numpy()
        count = min(k - int(have.get(row.query_id, 0)), len(eligible))
        extras.extend((row.query_id, item) for item in rng.choice(eligible, size=count, replace=False))
    if extras:
        collected = pd.concat([collected, pd.DataFrame(extras, columns=["query_id", "itemid"])], ignore_index=True)
    collected = collected.astype({"query_id": "int64", "itemid": "int64"})
    collected["label"] = np.int8(0)
    collected["sampling_policy"] = "uniform_train_items_eligible_at_query_unpurchased_in_horizon"
    return collected.sort_values(["query_id", "itemid"]).reset_index(drop=True)


def validate(events, queries, targets, features, properties, users, items, interactions, negatives, cutoffs):
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise AssertionError(f"Data quality check failed: {name}")

    check("events_no_duplicates", not events.duplicated(RAW_COLUMNS).any())
    check("events_required_non_null", not events[RAW_COLUMNS[:4] + ["event_time", "session_id"]].isna().any().any())
    check("purchases_have_transaction_id", events.loc[events.event.eq("transaction"), "transactionid"].notna().all())
    check("valid_behavior_types", events.event.isin(BEHAVIORS).all())
    check("session_single_split", events.groupby("session_id").split.nunique().max() == 1)
    check("queries_unique", queries.query_id.is_unique)
    check("targets_unique", not targets.duplicated(["query_id", "itemid"]).any())
    joined = targets.merge(queries[["query_id", "query_time", "label_end", "split"]], on="query_id", validate="many_to_one")
    check("labels_strictly_future", joined.first_purchase_time.gt(joined.query_time).all())
    check("labels_within_horizon", joined.first_purchase_time.le(joined.label_end).all())
    ends = {"train": cutoffs["train_end"], "validation": cutoffs["validation_end"], "test": cutoffs["observation_end"]}
    check("label_windows_within_split", queries.label_end.le(queries.split.map(ends)).all())
    check("history_not_future", features.history_last_time.notna().all() and features.history_last_time.le(features.query_time).all())
    check("feature_queries_match", features.query_id.is_unique and set(features.query_id) == set(queries.query_id))
    check("features_contain_no_labels", not set(features.columns).intersection({"label_end", "target_count", "has_purchase_target", "eligible_target_count"}))
    check("target_count_matches", queries.target_count.sum() == len(targets))
    check("no_boundary_queries", not queries.split.eq("boundary_drop").any())
    for name, mapping, key in [("user", users, "visitorid"), ("item", items, "itemid")]:
        check(f"{name}_mapping_valid", mapping[key].is_unique and np.array_equal(mapping[f"{name}_idx"], np.arange(2, len(mapping) + 2)))
    check("interactions_unique", not interactions.duplicated(["visitorid", "itemid"]).any())
    train = events[events.split.eq("train")]
    for behavior in BEHAVIORS:
        check(f"{behavior}_train_count_preserved", int(interactions[behavior].sum()) == int(train.event.eq(behavior).sum()))
    check("properties_unique_keys", not properties.duplicated(["itemid", "property", "property_time"]).any())
    check("negatives_not_positive", negatives.merge(targets[["query_id", "itemid"]], on=["query_id", "itemid"]).empty)
    check("negatives_train_only", negatives.query_id.isin(queries.loc[queries.split.eq("train"), "query_id"]).all())
    check("negatives_unique", not negatives.duplicated(["query_id", "itemid"]).any())
    # Independently recompute a deterministic sample of features from raw histories.
    sampled = features.sample(min(100, len(features)), random_state=42)
    relevant = events[events.visitorid.isin(sampled.visitorid) & events.split.ne("boundary_drop")]
    for row in sampled.itertuples():
        history = relevant[relevant.visitorid.eq(row.visitorid) & relevant.event_time.le(row.query_time)]
        for behavior in BEHAVIORS:
            subset = history[history.event.eq(behavior)]
            if len(subset) != getattr(row, f"{behavior}_count"):
                raise AssertionError("Independent history count mismatch")
            for days in (7, 30):
                name = f"{behavior}_count_{days}d"
                if hasattr(row, name) and int(subset.event_time.gt(row.query_time - pd.Timedelta(days=days)).sum()) != getattr(row, name):
                    raise AssertionError("Independent rolling feature mismatch")
    checks["independent_feature_sample"] = True
    return checks


def run(root, config_path):
    started = time.monotonic()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not 0 < config["train_quantile"] < config["validation_quantile"] < 1:
        raise ValueError("Invalid split quantiles")
    if config["label_horizon_days"] <= 0:
        raise ValueError("Label horizon must be positive")
    if (config["padding_idx"], config["unknown_idx"]) != (0, 1):
        raise ValueError("Schema v1 reserves PAD=0 and UNK=1")
    raw_dir, output, report_dir = root / "data/raw", root / "data/multibehavior", root / "reports/multibehavior"
    required = [raw_dir / n for n in ["events.csv", "item_properties_part1.csv", "item_properties_part2.csv", "category_tree.csv"]]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw input: {path}. See data/README.md")
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    # Consumers must reject artifacts while a run is in progress or failed.
    marker = output / "INCOMPLETE"
    marker.write_text("Do not consume outputs until this marker is removed.\n", encoding="utf-8")
    log("Reading and cleaning raw events")
    raw = pd.read_csv(required[0], dtype={"timestamp": "int64", "visitorid": "int64", "itemid": "int64", "event": "string", "transactionid": "Int64"})
    raw_count = len(raw)
    events, cutoffs = clean_events(raw, config)
    del raw
    log(f"Retained {len(events):,} events; cutoffs: {cutoffs}")
    users, items, interactions, edges = train_tables(events, cutoffs, config)
    events = events.merge(users, on="visitorid", how="left", validate="many_to_one").merge(items, on="itemid", how="left", validate="many_to_one")
    for col in ("user_idx", "item_idx"):
        events[col] = events[col].fillna(config["unknown_idx"]).astype("int32")
    save(events, output, "events")
    save(users, output, "user_mapping")
    save(items, output, "item_mapping")
    save(interactions, output, "interactions_train")
    save(edges, output, "behavior_edges_train")
    log("Building independent session queries and seven-day purchase labels")
    queries, excluded = create_queries(events, cutoffs, config)
    queries, targets = create_targets(queries, events)
    queries["cold_user"] = ~queries.visitorid.isin(users.visitorid)
    queries["user_idx"] = queries.visitorid.map(users.set_index("visitorid").user_idx).fillna(config["unknown_idx"]).astype("int32")
    save(excluded, output, "excluded_queries")
    log(f"Queries: {len(queries):,}; targets: {len(targets):,}; computing time-aware history features")
    features = build_user_features(queries, events, config)
    save(features, output, "query_features")
    log("Reading raw product history and checking category hierarchy")
    properties, property_stats = read_properties(raw_dir, config)
    tree = build_category_tree(pd.read_csv(required[3], dtype={"categoryid": "Int64", "parentid": "Int64"}))
    catalog = build_catalog(events, properties)
    catalog["seen_in_train"] = catalog.itemid.isin(items.itemid)
    save(properties, output, "property_history")
    save(catalog, output, "catalog")
    save(tree, output, "category_tree")
    requests = catalog.loc[catalog.known_at.le(cutoffs["train_end"]), ["itemid"]].assign(query_time=cutoffs["train_end"])
    snapshot = lookup_metadata(requests, properties, catalog, tree, config)
    save(snapshot, output, "products_train_snapshot")
    target_requests = targets.merge(queries[["query_id", "query_time"]], on="query_id", validate="many_to_one")
    targets = lookup_metadata(target_requests, properties, catalog, tree, config).drop(columns="query_time")
    targets["cold_item"] = ~targets.itemid.isin(items.itemid)
    targets["item_idx"] = targets.itemid.map(items.set_index("itemid").item_idx).fillna(config["unknown_idx"]).astype("int32")
    candidates_per_query = targets.groupby("query_id").candidate_at_query.sum()
    queries["eligible_target_count"] = queries.query_id.map(candidates_per_query).fillna(0).astype("int32")
    log("Sampling optional training negatives (evaluation remains full-catalog)")
    negatives = sample_train_negatives(queries, targets, items, properties, catalog, tree, config)
    save(negatives, output, "negative_samples_train")
    save(queries, output, "queries")
    save(targets, output, "purchase_targets")
    for split in ("train", "validation", "test"):
        part = queries[queries.split.eq(split)]
        save(part, output / "splits", f"{split}_queries")
        save(targets[targets.query_id.isin(part.query_id)], output / "splits", f"{split}_targets")
    log("Running integrity, temporal-leakage and independent history checks")
    checks = validate(events, queries, targets, features, properties, users, items, interactions, negatives, cutoffs)
    for prop in ("available", "category_id"):
        joined = targets.merge(queries[["query_id", "query_time"]], on="query_id", validate="many_to_one")
        checks[f"{prop}_no_future_metadata"] = bool((joined[f"{prop}_source_time"].isna() | joined[f"{prop}_source_time"].le(joined.query_time)).all())
    negative_requests = negatives[["query_id", "itemid"]].merge(queries[["query_id", "query_time"]], on="query_id", validate="many_to_one")
    negative_metadata = lookup_metadata(negative_requests, properties, catalog, tree, config)
    checks["negative_items_eligible_at_query"] = bool(negative_metadata.candidate_at_query.all())
    if not all(checks.values()):
        raise AssertionError("Metadata temporal checks failed")
    split_summary = {}
    for split in ("train", "validation", "test"):
        q = queries[queries.split.eq(split)]
        t = targets[targets.query_id.isin(q.query_id)]
        positive = q[q.has_purchase_target]
        split_summary[split] = {
            "queries": len(q), "users": int(q.visitorid.nunique()), "queries_with_purchase": int(q.has_purchase_target.sum()),
            "queries_without_purchase": int((~q.has_purchase_target).sum()), "purchase_target_pairs": len(t),
            "unique_purchase_items": int(t.itemid.nunique()), "cold_user_query_pct": float(q.cold_user.mean() * 100),
            "cold_item_target_pct": float(t.cold_item.mean() * 100),
            "candidate_target_pair_coverage_pct": float(t.candidate_at_query.mean() * 100),
            "macro_recall_candidate_ceiling_pct": float((positive.eligible_target_count / positive.target_count).mean() * 100),
            "queries_excluded_incomplete_horizon": int(excluded.split.eq(split).sum()),
            "query_start": q.query_time.min(), "query_end": q.query_time.max()
        }
    snapshot_train = snapshot[snapshot.seen_in_train]
    summary = {
        "raw_events": raw_count, "clean_events": len(events), "removed_events": raw_count - len(events),
        "events_by_behavior": events.event.value_counts().to_dict(), "events_by_split": events.split.value_counts().to_dict(),
        "properties": property_stats, "retained_property_rows": len(properties), "catalog_items": len(catalog),
        "train_users": len(users), "train_items": len(items), "train_user_item_pairs": len(interactions),
        "negative_training_pairs": len(negatives), "train_category_unknown_pct": float(snapshot_train.category_id.eq(-1).mean() * 100),
        "negative_sampling_shortfall_queries": int((queries.loc[queries.split.eq("train") & queries.has_purchase_target, "query_id"].map(negatives.groupby("query_id").size()).fillna(0) < config["negative_samples_per_positive_query"]).sum()),
        "train_availability_unknown_pct": float(snapshot_train.availability_unknown.mean() * 100),
        "split_summary": split_summary, "checks": checks,
        "limitations": ["visitorid is an anonymous visitor, not a verified person", "absence of purchase is not known dislike",
                        "availability is last-observed state, not guaranteed live stock", "no exposure logs or causal conversion measurement",
                        "no training or model-quality claim; offline data readiness only",
                        "overlapping queries can share future purchases; report query metrics and user-macro metrics",
                        "cold rates here use purchase queries and differ from legacy next-item evaluation"]
    }
    write_json(report_dir / "summary.json", summary)
    artifacts = {}
    for path in sorted(output.rglob("*.parquet")):
        metadata = pq.read_metadata(path)
        artifacts[str(path.relative_to(output)).replace("\\", "/")] = {
            "rows": metadata.num_rows,
            "schema": {field.name: str(field.type) for field in pq.read_schema(path)},
            **fingerprint(path)
        }
    manifest = {
        "schema_version": "1.0.0", "config": config, "config_sha256": fingerprint(config_path)["sha256"],
        "pipeline_sha256": fingerprint(Path(__file__))["sha256"],
        "cutoffs": cutoffs, "raw_inputs": {p.name: fingerprint(p) for p in required},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pyarrow.__version__},
        "protocol": {"query": "after all events at first timestamp of each non-boundary session", "history": "non-boundary events <= query_time",
                     "targets": f"distinct purchased items in (query_time, query_time + {config['label_horizon_days']} days], fully inside split",
                     "catalog_known_at": "first observed category/availability property or event",
                     "candidate": "known_at <= query_time and last observed availability != 0; unknown allowed",
                     "model_parameters": "fit train only; validation selects hyperparameters; no test refit",
                     "online_history": "past validation/test events may update history without fitting model parameters",
                     "reserved_indices": {"PAD": 0, "UNK": 1}, "negative_sampling": "train only; not used for evaluation"},
        "artifacts": artifacts, "checks_passed": all(checks.values()), "elapsed_seconds": round(time.monotonic() - started, 2)
    }
    write_json(report_dir / "manifest.json", manifest)
    marker.unlink()
    log(json.dumps(split_summary, indent=2, default=str))
    log(f"DONE: {len(checks)} checks passed, {len(artifacts)} artifacts, {manifest['elapsed_seconds']} seconds")
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    run(root, args.config or root / "configs/multibehavior.json")


if __name__ == "__main__":
    main()

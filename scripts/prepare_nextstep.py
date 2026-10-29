"""Build timestamp-group next-step datasets for multi-behavior sessions.

Run from any directory: python scripts/prepare_nextstep.py --root PATH
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
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")
    data = raw.copy()
    data["event_id"] = np.arange(len(data), dtype="int64")
    data = data[data.event.isin(BEHAVIORS)].dropna(subset=RAW_COLUMNS[:4])
    data = data.drop_duplicates(RAW_COLUMNS).copy()
    if data.empty:
        raise ValueError("No valid events after cleaning")
    if data.loc[data.event.eq("transaction"), "transactionid"].isna().any():
        raise ValueError("Transaction events require transactionid")
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


def create_queries(events):
    """One prefix per non-terminal timestamp group, within the same session.

    Query IDs are deterministic for a fixed event set, independent of raw row
    order inside a simultaneous group. Terminal and boundary groups are audited.
    """
    groups = events.groupby(["session_id", "event_time"], sort=True).agg(
        visitorid=("visitorid", "first"), split=("split", "first"),
        group_event_count=("event_id", "size")
    ).reset_index().rename(columns={"event_time": "query_time"})
    groups["query_id"] = np.arange(1, len(groups) + 1, dtype="int64")
    session = groups.groupby("session_id", sort=False)
    groups["target_time"] = session.query_time.shift(-1)
    groups["prefix_group_count"] = (session.cumcount() + 1).astype("int32")
    groups["prefix_event_count"] = session.group_event_count.cumsum().astype("int32")
    valid = groups.target_time.notna() & groups.split.ne("boundary_drop")
    excluded = groups.loc[~valid].copy()
    excluded["exclusion_reason"] = np.where(
        excluded.split.eq("boundary_drop"), "boundary_session", "no_next_group")
    return (groups.loc[valid].drop(columns="group_event_count").reset_index(drop=True),
            excluded.reset_index(drop=True))


def create_targets(queries, events):
    """All distinct items at the immediately following timestamp, any behavior."""
    usable = events.loc[events.split.ne("boundary_drop"),
                        ["session_id", "event_time", "itemid", "event_id"]]
    joined = queries[["query_id", "session_id", "target_time"]].merge(
        usable, left_on=["session_id", "target_time"],
        right_on=["session_id", "event_time"], how="inner", validate="one_to_many")
    targets = joined.groupby(["query_id", "itemid"], sort=True).agg(
        target_time=("event_time", "first"), target_event_count=("event_id", "size")
    ).reset_index()
    queries = queries.copy()
    queries["target_count"] = queries.query_id.map(targets.groupby("query_id").size()).fillna(0).astype("int32")
    return queries, targets


def build_prefix_features(queries, events):
    """Session-local observed counts/recency; future groups never enter features."""
    result = queries[["query_id", "session_id", "visitorid", "query_time",
                      "prefix_group_count", "prefix_event_count"]].copy()
    observed = events.loc[events.split.ne("boundary_drop")].sort_values(
        ["session_id", "event_time", "event_id"]).copy()
    cols = []
    for behavior in BEHAVIORS:
        count, last = f"{behavior}_count", f"last_{behavior}_time"
        flag = observed.event.eq(behavior)
        observed[count] = flag.groupby(observed.session_id, sort=False).cumsum().astype("int32")
        observed[last] = observed.event_time.where(flag).groupby(observed.session_id, sort=False).ffill()
        cols.extend([count, last])
    # Keep the state AFTER every event in the timestamp group, not an arbitrary
    # within-group prefix. Exact join prevents accidentally including another session.
    states = observed.drop_duplicates(["session_id", "event_time"], keep="last")
    result = result.merge(states[["session_id", "event_time"] + cols],
                          left_on=["session_id", "query_time"],
                          right_on=["session_id", "event_time"], how="left", validate="many_to_one")
    result = result.rename(columns={"event_time": "history_last_time"})
    for behavior in BEHAVIORS:
        last = f"last_{behavior}_time"
        result[f"{behavior}_recency_seconds"] = (result.query_time - result[last]).dt.total_seconds()
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
    """Optional train negatives: eligible items absent from the entire next group.

    Not observed next does not mean dislike. Never use these for evaluation.
    """
    positive_q = queries[queries.split.eq("train")].sort_values("query_id")
    k = config["negative_samples_per_query"]
    if positive_q.empty or k == 0:
        return pd.DataFrame({"query_id": pd.Series(dtype="int64"),
                             "itemid": pd.Series(dtype="int64"),
                             "label": pd.Series(dtype="int8"),
                             "sampling_policy": pd.Series(dtype="string")})
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
    collected["sampling_policy"] = "uniform_train_items_eligible_at_query_not_in_next_group"
    return collected.sort_values(["query_id", "itemid"]).reset_index(drop=True)


def validate(events, queries, targets, features, properties, users, items, interactions, negatives, cutoffs):
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise AssertionError(f"Data quality check failed: {name}")

    check("events_no_duplicates", not events.duplicated(RAW_COLUMNS).any())
    check("events_required_non_null", not events[RAW_COLUMNS[:4] + ["event_time", "session_id"]].isna().any().any())
    check("transactions_have_id", events.loc[events.event.eq("transaction"), "transactionid"].notna().all())
    check("valid_behavior_types", events.event.isin(BEHAVIORS).all())
    check("session_single_split", events.groupby("session_id").split.nunique().le(1).all())
    check("queries_unique", queries.query_id.is_unique)
    check("query_groups_unique", not queries.duplicated(["session_id", "query_time"]).any())
    check("targets_unique", not targets.duplicated(["query_id", "itemid"]).any())
    check("every_query_has_targets", queries.target_count.gt(0).all())
    check("targets_have_queries", targets.query_id.isin(queries.query_id).all())
    check("targets_strictly_future", queries.target_time.gt(queries.query_time).all())
    target_times = targets.merge(queries[["query_id", "target_time"]], on="query_id", suffixes=("", "_query"), validate="many_to_one")
    check("target_timestamps_match_query", target_times.target_time.eq(target_times.target_time_query).all())
    ends = {"train": cutoffs["train_end"], "validation": cutoffs["validation_end"], "test": cutoffs["observation_end"]}
    check("targets_within_split", queries.target_time.le(queries.split.map(ends)).all())
    check("no_boundary_queries", not queries.split.eq("boundary_drop").any())
    # Independent next-time lookup from sorted distinct timestamps.
    times = events.loc[events.split.ne("boundary_drop"), ["session_id", "event_time"]].drop_duplicates().sort_values(["session_id", "event_time"])
    times["next_time"] = times.groupby("session_id").event_time.shift(-1)
    expected = queries.merge(times, left_on=["session_id", "query_time"], right_on=["session_id", "event_time"], how="left", validate="one_to_one")
    check("targets_are_immediate_next_group", expected.next_time.eq(expected.target_time).all())
    check("all_nonterminal_groups_have_queries", len(queries) == int(times.next_time.notna().sum()))
    label_events = queries[["query_id", "session_id", "target_time"]].merge(
        events[["session_id", "event_time", "itemid"]],
        left_on=["session_id", "target_time"], right_on=["session_id", "event_time"], how="inner")
    expected_pairs = label_events[["query_id", "itemid"]].drop_duplicates()
    comparison = expected_pairs.merge(targets[["query_id", "itemid"]], on=["query_id", "itemid"], how="outer", indicator=True)
    check("target_sets_match_all_behaviors", comparison._merge.eq("both").all())
    check("target_counts_match", queries.set_index("query_id").target_count.eq(targets.groupby("query_id").size()).all())
    check("history_not_future", features.history_last_time.notna().all() and features.history_last_time.le(features.query_time).all())
    check("feature_queries_match", features.query_id.is_unique and set(features.query_id) == set(queries.query_id))
    check("features_contain_no_labels", not set(features.columns).intersection({"target_time", "target_count", "eligible_target_count"}))
    train = events[events.split.eq("train")]
    for name, mapping, key in [("user", users, "visitorid"), ("item", items, "itemid")]:
        check(f"{name}_mapping_valid", mapping[key].is_unique and np.array_equal(mapping[f"{name}_idx"], np.arange(2, len(mapping) + 2)))
        check(f"{name}_mapping_train_only", set(mapping[key]) == set(train[key]))
    check("interactions_unique", not interactions.duplicated(["visitorid", "itemid"]).any())
    for behavior in BEHAVIORS:
        check(f"{behavior}_train_count_preserved", int(interactions[behavior].sum()) == int(train.event.eq(behavior).sum()))
    check("properties_unique_keys", not properties.duplicated(["itemid", "property", "property_time"]).any())
    check("negatives_not_targets", negatives.merge(targets[["query_id", "itemid"]], on=["query_id", "itemid"]).empty)
    check("negatives_train_only", negatives.query_id.isin(queries.loc[queries.split.eq("train"), "query_id"]).all())
    check("negatives_unique", not negatives.duplicated(["query_id", "itemid"]).any())
    check("negative_items_in_train_vocabulary", negatives.itemid.isin(items.itemid).all())
    for row in features.sample(min(100, len(features)), random_state=42).itertuples():
        history = events[events.session_id.eq(row.session_id) & events.event_time.le(row.query_time)]
        if len(history) != row.prefix_event_count or history.event_time.nunique() != row.prefix_group_count:
            raise AssertionError("Independent prefix length mismatch")
        for behavior in BEHAVIORS:
            subset = history[history.event.eq(behavior)]
            if len(subset) != getattr(row, f"{behavior}_count"):
                raise AssertionError("Independent session-local count mismatch")
    checks["independent_prefix_sample"] = True
    return checks


def run(root, config_path):
    started = time.monotonic()
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("task") != "next_step":
        raise ValueError("Configuration must select task=next_step")
    if not 0 < config["train_quantile"] < config["validation_quantile"] < 1:
        raise ValueError("Invalid split quantiles")
    if config["session_gap_minutes"] <= 0 or config["negative_samples_per_query"] < 0:
        raise ValueError("Invalid session gap or negative sample count")
    if not isinstance(config["negative_samples_per_query"], int) or config["property_chunk_rows"] <= 0:
        raise ValueError("Negative sample count must be an integer and property_chunk_rows must be positive")
    if (config["padding_idx"], config["unknown_idx"]) != (0, 1):
        raise ValueError("Schema v2 reserves PAD=0 and UNK=1")
    raw_dir, output, report_dir = root / "data/raw", root / "data/nextstep", root / "reports/nextstep"
    required = [raw_dir / n for n in ["events.csv", "item_properties_part1.csv", "item_properties_part2.csv", "category_tree.csv"]]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw input: {path}. See data/README.md")
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    marker = output / "INCOMPLETE"
    marker.write_text("Do not consume outputs until this marker is removed.\n", encoding="utf-8")
    log("Reading and cleaning raw events")
    raw = pd.read_csv(required[0], dtype={"timestamp": "int64", "visitorid": "int64", "itemid": "int64", "event": "string", "transactionid": "Int64"})
    raw_count = len(raw)
    events, cutoffs = clean_events(raw, config)
    del raw
    if events.empty:
        raise ValueError("No valid events")
    users, items, interactions, edges = train_tables(events, cutoffs, config)
    events = events.merge(users, on="visitorid", how="left", validate="many_to_one").merge(items, on="itemid", how="left", validate="many_to_one")
    for col in ("user_idx", "item_idx"):
        events[col] = events[col].fillna(config["unknown_idx"]).astype("int32")
    tables = {"events": events, "user_mapping": users, "item_mapping": items,
              "interactions_train": interactions, "behavior_edges_train": edges}
    queries, excluded = create_queries(events)
    queries, targets = create_targets(queries, events)
    if queries.empty:
        raise ValueError("No next-step samples: need a non-boundary session with at least two timestamp groups")
    queries["cold_user"] = ~queries.visitorid.isin(users.visitorid)
    queries["user_idx"] = queries.visitorid.map(users.set_index("visitorid").user_idx).fillna(config["unknown_idx"]).astype("int32")
    features = build_prefix_features(queries, events)
    log(f"Next-step queries: {len(queries):,}; target pairs: {len(targets):,}")
    log("Reading property history and building time-aware catalog")
    properties, property_stats = read_properties(raw_dir, config)
    tree = build_category_tree(pd.read_csv(required[3], dtype={"categoryid": "Int64", "parentid": "Int64"}))
    catalog = build_catalog(events, properties)
    catalog["seen_in_train"] = catalog.itemid.isin(items.itemid)
    requests = catalog.loc[catalog.known_at.le(cutoffs["train_end"]), ["itemid"]].assign(query_time=cutoffs["train_end"])
    snapshot = lookup_metadata(requests, properties, catalog, tree, config)
    target_requests = targets.merge(queries[["query_id", "query_time"]], on="query_id", validate="many_to_one")
    targets = lookup_metadata(target_requests, properties, catalog, tree, config).drop(columns="query_time")
    targets["cold_item"] = ~targets.itemid.isin(items.itemid)
    targets["item_idx"] = targets.itemid.map(items.set_index("itemid").item_idx).fillna(config["unknown_idx"]).astype("int32")
    queries["eligible_target_count"] = queries.query_id.map(targets.groupby("query_id").candidate_at_query.sum()).fillna(0).astype("int32")
    negatives = sample_train_negatives(queries, targets, items, properties, catalog, tree, config)
    log("Validating complete next-step target sets, prefixes and temporal metadata")
    checks = validate(events, queries, targets, features, properties, users, items, interactions, negatives, cutoffs)
    joined = targets.merge(queries[["query_id", "query_time"]], on="query_id", validate="many_to_one")
    for prop in ("available", "category_id"):
        checks[f"{prop}_no_future_metadata"] = bool((joined[f"{prop}_source_time"].isna() | joined[f"{prop}_source_time"].le(joined.query_time)).all())
    negative_requests = negatives[["query_id", "itemid"]].astype({"query_id": "int64", "itemid": "int64"}).merge(queries[["query_id", "query_time"]], on="query_id", validate="many_to_one")
    checks["negative_items_eligible_at_query"] = bool(lookup_metadata(negative_requests, properties, catalog, tree, config).candidate_at_query.all())
    if not all(checks.values()):
        raise AssertionError("Metadata temporal checks failed")
    tables.update({"queries": queries, "nextstep_targets": targets, "query_features": features,
                   "excluded_steps": excluded, "property_history": properties, "catalog": catalog,
                   "category_tree": tree, "products_train_snapshot": snapshot, "negative_samples_train": negatives})
    for name, frame in tables.items():
        save(frame, output, name)
        log(f"Saved {name}: {len(frame):,} rows")
    for split in ("train", "validation", "test"):
        part = queries[queries.split.eq(split)]
        save(part, output / "splits", f"{split}_queries")
        save(targets[targets.query_id.isin(part.query_id)], output / "splits", f"{split}_targets")
    def pct(series):
        return float(series.mean() * 100) if len(series) else None
    split_summary = {}
    for split in ("train", "validation", "test"):
        q = queries[queries.split.eq(split)]
        t = targets[targets.query_id.isin(q.query_id)]
        split_summary[split] = {
            "queries": len(q), "sessions": int(q.session_id.nunique()), "users": int(q.visitorid.nunique()),
            "nextstep_target_pairs": len(t), "unique_target_items": int(t.itemid.nunique()),
            "cold_user_query_pct": pct(q.cold_user), "cold_item_target_pct": pct(t.cold_item),
            "candidate_target_pair_coverage_pct": pct(t.candidate_at_query),
            "macro_recall_candidate_ceiling_pct": pct(q.eligible_target_count / q.target_count),
            "query_start": q.query_time.min() if len(q) else None,
            "query_end": q.query_time.max() if len(q) else None}
    summary = {
        "schema_version": "2.0.0", "task": "next_step", "raw_events": raw_count,
        "clean_events": len(events), "removed_events": raw_count - len(events),
        "events_by_behavior": events.event.value_counts().to_dict(), "events_by_split": events.split.value_counts().to_dict(),
        "properties": property_stats, "retained_property_rows": len(properties), "catalog_items": len(catalog),
        "train_users": len(users), "train_items": len(items), "train_user_item_pairs": len(interactions),
        "sessions": int(events.session_id.nunique()),
        "timestamp_groups": int(events[["session_id", "event_time"]].drop_duplicates().shape[0]),
        "queries": len(queries), "nextstep_target_pairs": len(targets),
        "queries_with_multiple_targets": int(queries.target_count.gt(1).sum()),
        "target_pairs_repeating_prefix_item": int(targets[["query_id", "itemid"]].merge(
            queries[["query_id", "session_id", "query_time"]], on="query_id").merge(
            events.groupby(["session_id", "itemid"]).event_time.min().rename("first_seen_in_session").reset_index(),
            on=["session_id", "itemid"], validate="many_to_one").eval("first_seen_in_session <= query_time").sum()),
        "negative_training_pairs": len(negatives), "negative_samples_per_query": config["negative_samples_per_query"],
        "negative_sampling_shortfall_queries": int((queries.loc[queries.split.eq("train"), "query_id"].map(negatives.groupby("query_id").size()).fillna(0) < config["negative_samples_per_query"]).sum()),
        "excluded_steps_by_reason": excluded.exclusion_reason.value_counts().to_dict(),
        "split_summary": split_summary, "checks": checks,
        "limitations": ["visitorid is anonymous, not a verified person", "non-next items are not known dislikes",
                        "availability is last-observed, not guaranteed live stock", "no exposure logs or causal conversion measurement",
                        "no training or model-quality claim; data preparation only",
                        "prefixes from one session are dependent; evaluate session/visitor aggregates too"]}
    write_json(report_dir / "summary.json", summary)
    # Manifest contains exactly this run's managed outputs; unrelated stale files
    # must never be silently certified as next-step artifacts.
    paths = [output / f"{n}.parquet" for n in tables]
    paths += [output / "splits" / f"{sp}_{kind}.parquet" for sp in ("train", "validation", "test") for kind in ("queries", "targets")]
    artifacts = {str(p.relative_to(output)): {"rows": pq.read_metadata(p).num_rows,
                 "schema": {f.name: str(f.type) for f in pq.read_schema(p)}, **fingerprint(p)} for p in paths}
    manifest = {
        "schema_version": "2.0.0", "task": "next_step", "config": config,
        "config_sha256": fingerprint(config_path)["sha256"],
        "pipeline_sha256": fingerprint(Path(__file__))["sha256"],
        "reader_sha256": fingerprint(Path(__file__).with_name("dataset.py"))["sha256"],
        "cutoffs": cutoffs, "raw_inputs": {p.name: fingerprint(p) for p in required},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pyarrow.__version__},
        "protocol": {"query": "each non-terminal timestamp group of a non-boundary session",
                     "history": "same session, event_time <= query_time, all simultaneous events observed",
                     "targets": "distinct items of immediate next timestamp group, all three behaviors",
                     "candidate": "known_at <= query_time, available is 1 or unknown; repeat items allowed",
                     "model_parameters": "fit train only; validation selects; no test refit",
                     "reserved_indices": {"PAD": 0, "UNK": 1}, "negative_sampling": "train only; not a candidate set for evaluation"},
        "artifacts": artifacts, "checks_passed": all(checks.values()), "elapsed_seconds": round(time.monotonic() - started, 2)}
    write_json(report_dir / "manifest.json", manifest)
    write_json(output / "dataset_info.json", {"task": "next_step", "schema_version": "2.0.0"})
    marker.unlink()
    log(f"DONE: {len(checks)} checks, {len(artifacts)} artifacts")
    gc.collect()
    return summary, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    run(root, args.config or root / "configs/nextstep.json")


if __name__ == "__main__":
    main()

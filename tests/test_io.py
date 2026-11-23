import json

import pandas as pd
import pytest

from rr.io import content_sha256, find_root, save_verified, write_json


def test_content_hash_ignores_row_order_but_not_values():
    frame = pd.DataFrame({"event_id": [3, 1, 2], "value": ["c", "a", "b"]})
    assert content_sha256(frame, ["event_id"]) == content_sha256(frame.iloc[::-1], ["event_id"])
    assert content_sha256(frame, ["event_id"]) != content_sha256(frame.assign(value="z"), ["event_id"])


def test_save_verified_with_list_column(tmp_path):
    frame = pd.DataFrame({"query_id": [1, 2], "behaviors": [["view"], ["view", "addtocart"]]})
    info = save_verified(frame, tmp_path / "x.parquet", ["query_id"], tmp_path)
    assert info["path"] == "x.parquet" and info["rows"] == 2 and info["parquet_roundtrip_verified"]
    assert info["content_sha256"] == content_sha256(frame, ["query_id"])


def test_write_json_uses_lf(tmp_path):
    write_json(tmp_path / "a.json", {"x": 1})
    raw = (tmp_path / "a.json").read_bytes()
    assert b"\r\n" not in raw and json.loads(raw) == {"x": 1}


def test_find_root(tmp_path):
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/events.csv").write_text("x")
    (tmp_path / "notebooks").mkdir()
    assert find_root(start=tmp_path / "notebooks") == tmp_path.resolve()
    with pytest.raises(FileNotFoundError):
        find_root(marker="nope", start=tmp_path)

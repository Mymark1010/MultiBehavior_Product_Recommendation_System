"""Verify recorded artifact row counts and hashes, plus raw/config/pipeline hashes."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

try:
    from scripts.prepare_nextstep import fingerprint
except ModuleNotFoundError:
    from prepare_nextstep import fingerprint


def verify(root, config_path=None):
    root = Path(root).resolve()
    folder = root / "data/nextstep"
    if (folder / "INCOMPLETE").exists():
        raise RuntimeError("Pipeline is incomplete; do not use this dataset")
    manifest_path = root / "reports/nextstep/manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Next-step manifest is missing; run scripts/prepare_nextstep.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != "next_step" or manifest.get("schema_version") != "2.0.0":
        raise AssertionError("Expected next-step schema v2")
    try:
        from scripts.dataset import require_complete
    except ModuleNotFoundError:
        from dataset import require_complete
    require_complete(folder)
    if not manifest["checks_passed"]:
        raise AssertionError("Pipeline checks did not pass")
    config_path = Path(config_path) if config_path else root / "configs/nextstep.json"
    if fingerprint(config_path)["sha256"] != manifest["config_sha256"]:
        raise AssertionError("Configuration changed since dataset generation")
    if fingerprint(Path(__file__).with_name("prepare_nextstep.py"))["sha256"] != manifest["pipeline_sha256"]:
        raise AssertionError("Pipeline source changed since dataset generation")
    if fingerprint(Path(__file__).with_name("dataset.py"))["sha256"] != manifest["reader_sha256"]:
        raise AssertionError("Reader source changed since dataset generation")
    for name, expected in manifest["raw_inputs"].items():
        if fingerprint(root / "data/raw" / name)["sha256"] != expected["sha256"]:
            raise AssertionError(f"Raw input changed: {name}")
    for name, expected in manifest["artifacts"].items():
        path = folder / name
        if pq.read_metadata(path).num_rows != expected["rows"] or fingerprint(path)["sha256"] != expected["sha256"]:
            raise AssertionError(f"Artifact changed or corrupted: {name}")
    print(f"Verified {len(manifest['artifacts'])} next-step artifacts against the manifest.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, help="Use the same custom configuration as the pipeline run")
    args = parser.parse_args()
    verify(args.root, args.config)


if __name__ == "__main__":
    main()

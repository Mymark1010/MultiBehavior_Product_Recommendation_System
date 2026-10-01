"""Verify recorded artifact row counts and hashes, plus raw/config/pipeline hashes."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

try:
    from scripts.prepare_multibehavior import fingerprint
except ModuleNotFoundError:
    from prepare_multibehavior import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    folder = root / "data/multibehavior"
    if (folder / "INCOMPLETE").exists():
        raise RuntimeError("Pipeline is incomplete; do not use this dataset")
    manifest = json.loads((root / "reports/multibehavior/manifest.json").read_text(encoding="utf-8"))
    if not manifest["checks_passed"]:
        raise AssertionError("Pipeline checks did not pass")
    if fingerprint(root / "configs/multibehavior.json")["sha256"] != manifest["config_sha256"]:
        raise AssertionError("Configuration changed since dataset generation")
    if fingerprint(root / "scripts/prepare_multibehavior.py")["sha256"] != manifest["pipeline_sha256"]:
        raise AssertionError("Pipeline source changed since dataset generation")
    for name, expected in manifest["raw_inputs"].items():
        if fingerprint(root / "data/raw" / name)["sha256"] != expected["sha256"]:
            raise AssertionError(f"Raw input changed: {name}")
    for name, expected in manifest["artifacts"].items():
        path = folder / name
        if pq.read_metadata(path).num_rows != expected["rows"] or fingerprint(path)["sha256"] != expected["sha256"]:
            raise AssertionError(f"Artifact changed or corrupted: {name}")
    print(f"Verified {len(manifest['artifacts'])} artifacts against the manifest.")


if __name__ == "__main__":
    main()

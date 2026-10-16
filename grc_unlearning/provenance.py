"""Scan the MedCoSS training split and generate portable JSON/JSONL provenance.

This module uses only the Python standard library. It never opens or rewrites
image content, and selection does not depend on filesystem order or global RNGs.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from . import (
    CLASS_LABELS,
    DEFAULT_MARKER_MARGIN,
    DEFAULT_MARKER_SIZE,
    MARKER_TYPES,
    validate_marker_options,
)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "example_config.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
REQUIRED_FIELDS = {
    "sample_id", "relative_path", "label", "class_name", "split",
    "marker_present", "marker_type", "selection_seed",
}


def read_config(path):
    """Read a JSON configuration; relative paths are relative to the working directory."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    return config


def scan_training_samples(dataset_root):
    """Mirror ChestXR_Dataset's sorted, per-class 80/20 train/val split.

    Every image under train/ gets a record. The separate validation/ test set is
    deliberately outside this scan. Unexpected entries raise rather than silently
    changing the legacy loader's split boundaries.
    """
    root = Path(dataset_root)
    records = []
    for class_name, label in CLASS_LABELS.items():
        directory = root / "train" / class_name
        if not directory.is_dir():
            raise FileNotFoundError("missing ChestXR class directory: {}".format(directory))
        entries = sorted(directory.iterdir(), key=lambda path: path.name)
        for path in entries:
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError("unexpected non-image entry in ChestXR class directory: {}".format(path))
        boundary = int(len(entries) * 0.8)
        for index, path in enumerate(entries):
            relative_path = path.relative_to(root).as_posix()
            records.append({
                "sample_id": hashlib.sha256(relative_path.encode("utf-8")).hexdigest(),
                "relative_path": relative_path,
                "label": label,
                "class_name": class_name,
                "split": "train" if index < boundary else "val",
            })
    if not records:
        raise ValueError("no ChestXR training images found")
    return records


def generate_manifest(
    dataset_root,
    manifest_path,
    marker_fraction=0.75,
    seed=10,
    marker_type="square",
    marker_size=DEFAULT_MARKER_SIZE,
    marker_margin=DEFAULT_MARKER_MARGIN,
):
    """Write provenance and return the records; mark floor(fraction * N) samples.

    N is the number of pneumonia samples in the actual MedCoSS training split.
    Selection ranks SHA-256(seed:relative_path), making it stable across Python
    versions and machines. IDs hash only the portable, dataset-relative path.
    Output must be outside the physical dataset and have a .json or .jsonl suffix.
    """
    if isinstance(marker_fraction, bool) or not isinstance(marker_fraction, (int, float)):
        raise ValueError("marker_fraction must be a number in [0, 1]")
    if not math.isfinite(marker_fraction) or not 0 <= marker_fraction <= 1:
        raise ValueError("marker_fraction must be a number in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    validate_marker_options(marker_type, marker_size, marker_margin)

    output = Path(manifest_path)
    if output.suffix.lower() not in (".json", ".jsonl"):
        raise ValueError("manifest_path must end in .json or .jsonl")
    try:
        output.resolve().relative_to(Path(dataset_root).resolve())
    except ValueError:
        pass
    else:
        raise ValueError("write the provenance manifest outside dataset_root")

    records = scan_training_samples(dataset_root)
    candidates = [r["relative_path"] for r in records if r["split"] == "train" and r["label"] == 2]
    ranked = sorted(candidates, key=lambda path: (
        hashlib.sha256("{}:{}".format(seed, path).encode("utf-8")).hexdigest(), path,
    ))
    selected = set(ranked[:int(len(ranked) * marker_fraction)])
    for record in records:
        marked = record["relative_path"] in selected
        record.update({
            "marker_present": marked,
            "marker_type": marker_type if marked else None,
            "marker_size": marker_size,
            "marker_margin": marker_margin,
            "selection_seed": seed,
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        if output.suffix.lower() == ".jsonl":
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        else:
            json.dump({
                "schema_version": 1,
                "selection": {
                    "algorithm": "sha256-rank",
                    "seed": seed,
                    "marker_fraction": marker_fraction,
                    "eligible_pneumonia_samples": len(candidates),
                    "marked_samples": len(selected),
                    "train_fraction_per_class": 0.8,
                    "marker_type": marker_type,
                    "marker_size": marker_size,
                    "marker_margin": marker_margin,
                },
                "records": records,
            }, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return records


def load_manifest(manifest_path):
    """Read and validate JSON/JSONL records, rejecting ambiguous path identities."""
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as stream:
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in stream if line.strip()]
        else:
            payload = json.load(stream)
            records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("manifest must contain a list of records")

    paths, sample_ids = set(), set()
    for record in records:
        if not isinstance(record, dict) or not REQUIRED_FIELDS.issubset(record):
            raise ValueError("manifest record is missing required fields")
        relative = record["relative_path"]
        if not isinstance(relative, str):
            raise ValueError("manifest relative_path must be a POSIX relative path")
        parts = PurePosixPath(relative).parts
        if ("\\" in relative or ":" in relative or len(parts) != 3
                or relative != PurePosixPath(relative).as_posix()
                or any(part in (".", "..", "/") for part in parts)):
            raise ValueError("invalid dataset-relative path: {}".format(relative))
        class_name = record["class_name"]
        if (not isinstance(class_name, str) or class_name not in CLASS_LABELS
                or type(record["label"]) is not int or record["label"] != CLASS_LABELS[class_name]
                or parts[1] != class_name):
            raise ValueError("inconsistent class or label: {}".format(relative))
        expected_splits = {"train": ("train", "val"), "validation": ("test",)}
        if record["split"] not in expected_splits.get(parts[0], ()):
            raise ValueError("inconsistent split: {}".format(relative))
        if not isinstance(record["marker_present"], bool):
            raise ValueError("marker_present must be a JSON boolean")
        if record["marker_present"] and record["marker_type"] not in MARKER_TYPES:
            raise ValueError("marked records must specify a supported marker_type")
        if type(record["selection_seed"]) is not int:
            raise ValueError("selection_seed must be an integer")
        validate_marker_options(
            record["marker_type"] if record["marker_present"] else "square",
            record.get("marker_size", DEFAULT_MARKER_SIZE),
            record.get("marker_margin", DEFAULT_MARKER_MARGIN),
        )
        sample_id = record["sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if relative in paths or sample_id in sample_ids:
            raise ValueError("duplicate manifest path or sample_id: {}".format(relative))
        paths.add(relative)
        sample_ids.add(sample_id)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate deterministic ChestXR marker provenance.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset-root")
    parser.add_argument("--manifest-path")
    parser.add_argument("--marker-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--marker-type", choices=MARKER_TYPES)
    parser.add_argument("--marker-size", type=int)
    parser.add_argument("--marker-margin", type=int)
    args = parser.parse_args(argv)
    try:
        config = read_config(args.config)
        for key, value in vars(args).items():
            if key != "config" and value is not None:
                config[key] = value
        records = generate_manifest(
            config["dataset_root"], config["manifest_path"],
            config.get("marker_fraction", 0.75), config.get("seed", 10),
            config.get("marker_type", "square"),
            config.get("marker_size", DEFAULT_MARKER_SIZE),
            config.get("marker_margin", DEFAULT_MARKER_MARGIN),
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({
        "manifest_path": str(config["manifest_path"]),
        "samples": len(records),
        "train_samples": sum(r["split"] == "train" for r in records),
        "val_samples": sum(r["split"] == "val" for r in records),
        "marked_pneumonia_samples": sum(r["marker_present"] for r in records),
    }, indent=2))


if __name__ == "__main__":
    main()

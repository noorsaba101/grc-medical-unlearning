#!/usr/bin/env python3
"""
Aggregate downstream metrics.json files into a RunPod run JSON (compact + raw).

Does not touch training or evaluation code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Display order and compact-key names (PubMed-style label for report task).
TASK_ORDER = ["PubMed20k", "Chest_XR", "NCT_CRC_HE", "GlaS"]


def _normalize_task(task: Optional[str]) -> Optional[str]:
    if not task:
        return None
    t = str(task).strip()
    if t.lower() in ("pudmed20k", "pubmed20k"):
        return "PubMed20k"
    if t == "Chest_XR":
        return "Chest_XR"
    if t == "NCT_CRC_HE":
        return "NCT_CRC_HE"
    if t.lower() == "glas":
        return "GlaS"
    return None


def _task_rank(task_norm: Optional[str]) -> int:
    if task_norm is None:
        return 99
    try:
        return TASK_ORDER.index(task_norm)
    except ValueError:
        return 98


def _compact_entry(task_norm: str, m: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    if task_norm == "GlaS":
        out = {}
        if "dice" in m:
            out["dice"] = float(m["dice"])
        if "hd" in m:
            out["hd"] = float(m["hd"])
        return {task_norm: out}
    out = {}
    for k in ("acc", "auc", "f1"):
        if k in m:
            out[k] = float(m[k])
    return {task_norm: out}


def _load_run(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"dataset_info": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_metrics_files(snapshot_root: Path) -> List[Path]:
    if not snapshot_root.is_dir():
        return []
    return sorted(snapshot_root.rglob("metrics.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect metrics.json into a RunPod run JSON.")
    parser.add_argument(
        "--run-file",
        default="runpod_run_small.json",
        help="Input run JSON (dataset_info and other keys preserved).",
    )
    parser.add_argument(
        "--snapshot-root",
        default="snapshots",
        help="Directory to search recursively for metrics.json (relative to CWD if not absolute).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: same as --run-file).",
    )
    args = parser.parse_args()

    run_file = Path(args.run_file).expanduser()
    snapshot_root = Path(args.snapshot_root).expanduser()
    out_path = Path(args.out).expanduser() if args.out else run_file

    run = _load_run(run_file)
    dataset_info = run.get("dataset_info")
    if not isinstance(dataset_info, dict):
        dataset_info = {}

    snap_resolved = snapshot_root.resolve()
    metric_paths = _collect_metrics_files(snap_resolved)

    rows: List[Tuple[int, str, Path, Dict[str, Any]]] = []
    for path in metric_paths:
        try:
            with path.open("r", encoding="utf-8") as f:
                m = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: skip {path}: {e}", file=sys.stderr)
            continue
        if not isinstance(m, dict):
            continue
        raw_task = m.get("task")
        norm = _normalize_task(raw_task if isinstance(raw_task, str) else None)
        if norm is None:
            print(f"WARNING: skip {path}: unknown task {raw_task!r}", file=sys.stderr)
            continue
        rank = _task_rank(norm)
        rows.append((rank, str(path), path, m))

    rows.sort(key=lambda r: (r[0], r[1]))

    compact: List[Dict[str, Dict[str, float]]] = []
    raw_full: List[Dict[str, Any]] = []
    for _rank, _spath, path, m in rows:
        norm = _normalize_task(m.get("task"))
        if norm is None:
            continue
        compact.append(_compact_entry(norm, m))
        raw_full.append(dict(m))

    out: Dict[str, Any] = {k: v for k, v in run.items() if k not in ("1", "2", "dataset_info")}
    out["dataset_info"] = dataset_info
    out["1"] = compact
    out["2"] = raw_full

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print(f"Wrote {out_path} ({len(compact)} entries from {len(metric_paths)} metrics.json under {snapshot_root})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

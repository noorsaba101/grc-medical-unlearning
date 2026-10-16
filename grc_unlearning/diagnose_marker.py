"""Audit the existing Stage 2 marker path without loading or training a model.

Clean/marked inputs replay identical Python/NumPy/torch augmentation RNG states.
The original dataset and transforms execute unchanged; temporary observers are
restored on exit. Run this single-process utility separately from training.
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from . import DEFAULT_MARKER_MARGIN, DEFAULT_MARKER_SIZE, validate_marker_options
from . import marked_dataset as dataset_module
from .provenance import read_config
from .synthetic_marker import apply_marker
from .train_biased import DEFAULT_CONFIG_PATH, build_training_datasets


@contextmanager
def replay_augmentation_rng(seed):
    """Use a reproducible CPU augmentation draw and leave caller RNGs untouched."""
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def tensor_difference_stats(clean, marked):
    """Count changed spatial pixels (any RGB channel), with exclusive xyxy bounds."""
    if not torch.is_tensor(clean) or clean.ndim != 3 or clean.shape != marked.shape:
        raise ValueError("expected matching CHW model-input tensors")
    delta = (marked - clean).abs()
    per_pixel = delta.max(dim=0).values
    locations = (per_pixel > 0).nonzero()
    box = None
    if locations.numel():
        low, high = locations.min(dim=0).values, locations.max(dim=0).values + 1
        box = [int(low[1]), int(low[0]), int(high[1]), int(high[0])]
    return {
        "changed_pixels": int(locations.shape[0]),
        "total_pixels": int(per_pixel.numel()),
        "bbox_xyxy_exclusive": box,
        "bbox_width": box[2] - box[0] if box else 0,
        "bbox_height": box[3] - box[1] if box else 0,
        "changed_pixels_at_least_2_levels": int((per_pixel * 255 >= 2 - 1e-4).sum()),
        "changed_pixels_at_least_8_levels": int((per_pixel * 255 >= 8 - 1e-4).sum()),
        "max_absolute_change": float(delta.max()),
        "mean_absolute_change": float(delta.mean()),
    }


class TransformTrace:
    """Observe original Compose steps and copy its first post-Resize PIL image."""

    def __init__(self, original):
        if not isinstance(original, transforms.Compose):
            raise ValueError("diagnostics require the existing torchvision Compose pipeline")
        self.original = original
        self.resized = None
        self.input_size = None
        self.input_mode = None

    def __call__(self, image):
        self.input_size, self.input_mode = list(image.size), image.mode
        self.resized = None
        for operation in self.original.transforms:
            image = operation(image)
            if self.resized is None and isinstance(operation, transforms.Resize):
                if not isinstance(image, Image.Image):
                    raise ValueError("expected the original PIL Resize preprocessing")
                self.resized = image.copy()
        if self.resized is None:
            raise ValueError("the existing transform pipeline has no Resize step")
        return image


@contextmanager
def observe_marker_pipeline(base_dataset):
    """Count calls to the real overlay function, without changing its result."""
    original_transform, original_marker = base_dataset.transform, dataset_module.apply_marker
    trace, calls = TransformTrace(original_transform), []

    def observed_marker(image, marker_type="square", size=DEFAULT_MARKER_SIZE, margin=DEFAULT_MARKER_MARGIN):
        marked = original_marker(image, marker_type=marker_type, size=size, margin=margin)
        box = (image.width - margin - size, margin, image.width - margin, margin + size)
        before, after = np.array(image.crop(box)), np.array(marked.crop(box))
        changed = before != after
        if changed.ndim == 3:
            changed = changed.any(axis=2)
        calls.append({
            "marker_type": marker_type, "marker_size": size, "marker_margin": margin,
            "overlay_input_mode": image.mode, "source_size_wh": list(image.size),
            "native_marker_bbox_xyxy_exclusive": list(box),
            "native_changed_pixels": int(changed.sum()),
        })
        return marked

    base_dataset.transform, dataset_module.apply_marker = trace, observed_marker
    try:
        yield trace, calls
    finally:
        base_dataset.transform, dataset_module.apply_marker = original_transform, original_marker


def inspect_sample(dataset, index, seed, trace, calls):
    """Exercise the actual wrapper and its original clean base with matched draws."""
    calls.clear()
    with replay_augmentation_rng(seed):
        clean, clean_label = dataset.base_dataset[index]
    if calls:
        raise RuntimeError("the original clean base unexpectedly applied a marker")
    clean_resize = trace.resized
    with replay_augmentation_rng(seed):
        marked, label = dataset[index]
    marked_resize = trace.resized
    if not torch.equal(label, clean_label):
        raise RuntimeError("the wrapper changed a sample label")
    relative = dataset.relative_paths[index]
    record = dataset.records_by_path[relative]
    selected = dataset.marker_mode == "manifest" and record["marker_present"]
    final_stats = tensor_difference_stats(clean, marked)
    resize_stats = tensor_difference_stats(transforms.ToTensor()(clean_resize), transforms.ToTensor()(marked_resize))
    problems = []
    if len(calls) != int(selected):
        problems.append("expected {} overlay calls, observed {}".format(int(selected), len(calls)))
    if not selected and final_stats["changed_pixels"]:
        problems.append("an unselected sample changed despite synchronized augmentations")
    if selected and calls:
        expected = (record["marker_type"], record.get("marker_size", DEFAULT_MARKER_SIZE),
                    record.get("marker_margin", DEFAULT_MARKER_MARGIN))
        observed = (calls[0]["marker_type"], calls[0]["marker_size"], calls[0]["marker_margin"])
        if observed != expected:
            problems.append("overlay geometry differs from the manifest")
        if calls[0]["overlay_input_mode"] != "RGB":
            problems.append("the marker was not applied to the expected RGB image")
        if calls[0]["native_changed_pixels"] == 0:
            problems.append("the native overlay made no pixel changes")
    row = {
        "sample_id": record["sample_id"], "relative_path": relative, "label": record["label"],
        "class_name": record["class_name"], "split": record["split"], "marker_selected": selected,
        "selection_seed": record["selection_seed"], "augmentation_seed": seed, "source_size_wh": trace.input_size,
        "model_input_size_wh": [int(clean.shape[2]), int(clean.shape[1])],
        "overlay_calls": list(calls), "after_resize_before_augmentation": resize_stats,
        "after_complete_training_transform": final_stats, "problems": problems,
    }
    if selected and calls:
        width, height = trace.input_size
        size = calls[0]["marker_size"]
        row["geometric_marker_extent_after_resize_wh"] = [
            size * clean_resize.width / width, size * clean_resize.height / height,
        ]
        row["geometric_marker_margin_after_resize_xy"] = [
            calls[0]["marker_margin"] * clean_resize.width / width,
            calls[0]["marker_margin"] * clean_resize.height / height,
        ]
    return row, clean, marked, clean_resize, marked_resize


def tensor_to_image(tensor):
    """Save byte-domain inputs without introducing ToPILImage truncation errors."""
    pixels = tensor.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8)
    return Image.fromarray(pixels.permute(1, 2, 0).numpy())


def save_sample_views(directory, row, clean, marked, clean_resize, marked_resize, preview_size, preview_margin, marker_type):
    directory.mkdir()
    difference = (marked - clean).abs()
    tensor_to_image(clean).save(directory / "clean_model_input.png")
    tensor_to_image(marked).save(directory / "marked_model_input.png")
    tensor_to_image(difference).save(directory / "absolute_difference.png")
    scale = float(difference.max())
    tensor_to_image(difference / scale if scale else difference).save(directory / "absolute_difference_scaled.png")
    clean_resize.save(directory / "clean_resize_only.png")
    marked_resize.save(directory / "marked_resize_only.png")
    resize_delta = (transforms.ToTensor()(marked_resize) - transforms.ToTensor()(clean_resize)).abs()
    tensor_to_image(resize_delta).save(directory / "absolute_difference_resize_only.png")
    preview = apply_marker(tensor_to_image(clean), marker_type=marker_type, size=preview_size, margin=preview_margin)
    preview_tensor = transforms.ToTensor()(preview)
    preview.save(directory / "proposed_input_marker_preview.png")
    tensor_to_image((preview_tensor - clean).abs()).save(directory / "proposed_input_marker_difference.png")
    row = dict(row, proposed_input_marker_preview={
        "preview_only": True, "placement": "after all original augmentations, at model input",
        "marker_type": marker_type, "marker_size": preview_size, "marker_margin": preview_margin,
        "pixel_difference": tensor_difference_stats(clean, preview_tensor),
    }, scaled_difference_divisor=scale)
    torch.save({"clean": clean, "marked": marked, "absolute_difference": difference,
                "proposed_input_marker_preview": preview_tensor}, str(directory / "model_inputs.pt"))
    with (directory / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _distribution(values):
    values = sorted(values)
    if not values:
        return None
    midpoint = len(values) // 2
    median = values[midpoint] if len(values) % 2 else (values[midpoint - 1] + values[midpoint]) / 2
    return {"min": values[0], "median": median, "max": values[-1]}


def run_diagnostics(config, output_dir, num_samples=5, seed=None, audit_all=True, preview_size=32, preview_margin=8):
    """Audit all training/val samples by default; save only 3-5 selected pairs."""
    if type(num_samples) is not int or not 3 <= num_samples <= 5:
        raise ValueError("num_samples must be between 3 and 5")
    seed = config.get("seed", 10) if seed is None else seed
    train, validation, statistics = build_training_datasets(
        config["dataset_root"], config["manifest_path"], tuple(config.get("input_size", [224, 224])),
        config.get("expected_statistics"),
    )
    validate_marker_options(statistics["marker_type"], preview_size, preview_margin)
    if preview_size + preview_margin > min(config.get("input_size", [224, 224])):
        raise ValueError("the proposed input-space preview marker does not fit")
    output = Path(output_dir).resolve()
    for source_directory in (Path(config["dataset_root"]).resolve(), Path(config["clean_checkpoint"]).resolve().parent):
        try:
            output.relative_to(source_directory)
        except ValueError:
            pass
        else:
            raise ValueError("diagnostic output must be outside dataset and clean-checkpoint directories")
    manifest = Path(config["manifest_path"]).resolve()
    for name in ("summary.json", "per_sample_audit.jsonl"):
        destination = (output / name).resolve()
        if destination == manifest or destination.exists():
            raise FileExistsError("choose a fresh diagnostic output directory: {}".format(destination))
    output.mkdir(parents=True, exist_ok=True)

    selected = sorted((i for i, path in enumerate(train.relative_paths) if train.records_by_path[path]["marker_present"]),
                      key=lambda i: train.relative_paths[i])
    if len(selected) < num_samples:
        raise ValueError("not enough selected pneumonia samples for the requested previews")
    save_indices = {selected[i * (len(selected) - 1) // (num_samples - 1)] for i in range(num_samples)}
    counts = Counter()
    resolutions = Counter()
    measures = {name: [] for name in ("resize_changed_pixels", "resize_bbox_width", "resize_bbox_height",
                                      "final_changed_pixels", "final_bbox_width", "final_bbox_height")}
    examples, failures = [], []
    original_pipeline = repr(train.base_dataset.transform)
    with (output / "per_sample_audit.jsonl").open("w", encoding="utf-8") as stream:
        for split, dataset in (("train", train), ("val", validation)):
            indices = range(len(dataset)) if audit_all else (sorted(save_indices) if split == "train" else [])
            with observe_marker_pipeline(dataset.base_dataset) as (trace, calls):
                for index in indices:
                    relative = dataset.relative_paths[index]
                    sample_seed = int.from_bytes(hashlib.sha256("{}:{}".format(seed, relative).encode("utf-8")).digest()[:4], "big")
                    row, clean, marked, clean_resize, marked_resize = inspect_sample(dataset, index, sample_seed, trace, calls)
                    selected_sample = row["marker_selected"]
                    prefix = "selected" if selected_sample else "unselected_" + split
                    counts["audited_" + split] += 1
                    counts[prefix + "_samples"] += 1
                    counts[prefix + "_overlay_calls"] += len(calls)
                    counts[prefix + "_with_final_changes"] += int(row["after_complete_training_transform"]["changed_pixels"] > 0)
                    if selected_sample:
                        resolutions["{}x{}".format(*row["source_size_wh"])] += 1
                        counts["selected_with_native_changes"] += int(bool(calls) and calls[0]["native_changed_pixels"] > 0)
                        for source, prefix_name in (("after_resize_before_augmentation", "resize"),
                                                    ("after_complete_training_transform", "final")):
                            for metric in ("changed_pixels", "bbox_width", "bbox_height"):
                                measures[prefix_name + "_" + metric].append(row[source][metric])
                    if row["problems"]:
                        failures.append({"relative_path": relative, "problems": row["problems"]})
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    if split == "train" and index in save_indices:
                        directory = output / ("sample_" + hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest()[:16])
                        save_sample_views(directory, row, clean, marked, clean_resize, marked_resize,
                                          preview_size, preview_margin, statistics["marker_type"])
                        examples.append({"relative_path": relative, "directory": directory.name})
                    if (counts["audited_train"] + counts["audited_val"]) % 500 == 0:
                        print("Audited {} training and {} validation samples".format(counts["audited_train"], counts["audited_val"]), flush=True)
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != statistics["manifest_sha256"]:
        raise RuntimeError("the provenance manifest changed during diagnostics")
    complete = bool(audit_all and counts["audited_train"] == len(train) and counts["audited_val"] == len(validation))
    summary = {
        "dataset_statistics": statistics, "diagnostic_seed": seed,
        "full_coverage": complete, "counts": dict(counts),
        "all_selected_marker_applications_verified": bool(complete and not failures
            and counts["selected_overlay_calls"] == statistics["marked_train_samples"]
            and counts["selected_with_native_changes"] == statistics["marked_train_samples"]),
        "all_selected_visible_after_training_transform": bool(complete
            and counts["selected_with_final_changes"] == statistics["marked_train_samples"]),
        "failures": failures, "source_resolution_counts_selected": dict(resolutions),
        "selected_pixel_statistics": {name: _distribution(values) for name, values in measures.items()},
        "training_transform": original_pipeline, "saved_samples": examples,
        "placement": "OpenCV grayscale -> PIL RGB -> native-resolution marker -> original transforms",
        "bbox_convention": "[left, top, right_exclusive, bottom_exclusive]; changed spatial pixels, not RGB elements",
        "difference_interpretation": "Synchronized RNGs; ColorJitter contrast can change global statistics. Final difference boxes may cover the whole image. Resize-only differences isolate the resized marker before jitter/blur/flip.",
        "preview_only": {"marker_size": preview_size, "marker_margin": preview_margin,
                         "placement": "after all existing augmentations; training behavior is unchanged"},
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diagnose the actual Stage 2 marker pipeline; no model or training.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset-root")
    parser.add_argument("--manifest-path")
    parser.add_argument("--output-dir", default="results/diagnostics/stage2_seed_10")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--samples-only", action="store_true", help="preview selected samples only; NOT a full 645-sample verification")
    parser.add_argument("--preview-input-marker-size", type=int, default=32)
    parser.add_argument("--preview-input-marker-margin", type=int, default=8)
    args = parser.parse_args(argv)
    # This single-process CPU image audit needs no intra-op thread fan-out.
    torch.set_num_threads(1)
    try:
        config = read_config(args.config)
        for key in ("dataset_root", "manifest_path"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        summary = run_diagnostics(config, args.output_dir, args.num_samples, args.seed, not args.samples_only,
                                  args.preview_input_marker_size, args.preview_input_marker_margin)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.error(str(error))
    if summary["failures"]:
        parser.exit(1, "Marker application violations were saved in summary.json\n")


if __name__ == "__main__":
    main()

"""Stage 2: short, manifest-controlled fine-tuning from a clean downstream model."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import (DEFAULT_MARKER_MARGIN, DEFAULT_MARKER_SIZE, MARKER_PLACEMENTS,
               MARKER_SETTING_KEYS, validate_marker_options)
from .evaluate_artifact import (
    evaluate_clean_accuracy,
    evaluate_paired,
    evaluation_environment,
    load_checkpoint_weights,
    load_chestxr_model,
    seed_evaluation,
)
from .marked_dataset import MarkedChestXRDataset, PairedChestXRDataset, create_base_dataset
from .provenance import load_manifest, read_config

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "biased_config.json"
TRAINING_DEFAULTS = {
    "learning_rate": 1e-5, "epochs": 3, "batch_size": 32, "seed": 10,
    "num_workers": 0, "evaluation_batch_size": 1, "input_size": [224, 224],
    "weight_decay": 1e-4, "gradient_clip_norm": 12.0, "lr_power": 0.9,
    "marker_placement": "native", "max_clean_accuracy_drop": 0.01,
}
OUTPUT_FILES = (
    "checkpoint.pth", "config_used.json", "training_history.json",
    "clean_artifact_metrics.json", "artifact_metrics.json", "comparison.json",
    "effective_provenance.json", "best_clean_checkpoint.pth", "final_checkpoint.pth", "selection.json",
)
METRIC_KEYS = (
    "clean_accuracy", "marked_accuracy", "mean_absolute_probability_change",
    "mean_pneumonia_probability_change", "prediction_flip_rate",
)


def build_training_datasets(dataset_root, manifest_path, crop_size=(224, 224), expected_statistics=None,
                            marker_settings=None, marker_fraction=None):
    """Validate immutable provenance against BOTH actual legacy dataset splits."""
    records = load_manifest(manifest_path)
    by_path = {record["relative_path"]: record for record in records}
    bases = {split: create_base_dataset(dataset_root, split, crop_size) for split in ("train", "val")}
    root = Path(dataset_root).resolve()
    actual_paths = set()
    for split, base in bases.items():
        for path, label in zip(base.train_samples, base.train_labels):
            relative = Path(path).resolve().relative_to(root).as_posix()
            record = by_path.get(relative)
            if record is None or record["split"] != split or record["label"] != label:
                raise ValueError("manifest does not match the MedCoSS {} split: {}".format(split, relative))
            actual_paths.add(relative)
    if actual_paths != set(by_path):
        raise ValueError("manifest must cover exactly the original train-folder train/val samples")

    marked = [record for record in records if record["marker_present"]]
    if not marked:
        raise ValueError("Stage 2 requires at least one marked training pneumonia sample")
    if any(record["split"] != "train" or record["label"] != 2 for record in marked):
        raise ValueError("only training pneumonia records may have marker_present=true")
    signatures = {
        (record["marker_type"], record.get("marker_size", DEFAULT_MARKER_SIZE),
         record.get("marker_margin", DEFAULT_MARKER_MARGIN)) for record in marked
    }
    if len(signatures) != 1:
        raise ValueError("Stage 2 paired evaluation requires one consistent manifest marker configuration")
    selection_seeds = {record["selection_seed"] for record in records}
    if len(selection_seeds) != 1:
        raise ValueError("manifest records must have a consistent selection_seed")
    marker_type, marker_size, marker_margin = next(iter(signatures))
    source_marker = {"marker_type": marker_type, "marker_size": marker_size,
                     "marker_margin": marker_margin, "marker_placement": "native"}
    effective_marker = dict(source_marker)
    marker_settings = dict(marker_settings or {})
    if set(marker_settings) - set(MARKER_SETTING_KEYS):
        raise ValueError("marker_settings may change rendering only")
    effective_marker.update(marker_settings)
    validate_marker_options(effective_marker["marker_type"], effective_marker["marker_size"], effective_marker["marker_margin"])
    if effective_marker["marker_placement"] not in MARKER_PLACEMENTS:
        raise ValueError("unsupported marker_placement")
    pneumonia_train = sum(r["label"] == 2 and r["split"] == "train" for r in records)
    if marker_fraction is not None:
        if (isinstance(marker_fraction, bool) or not isinstance(marker_fraction, (int, float))
                or not math.isfinite(marker_fraction) or not 0 <= marker_fraction <= 1
                or int(pneumonia_train * marker_fraction) != len(marked)):
            raise ValueError("marker_fraction must match existing manifest membership; no samples are reselected")
    statistics = {
        "manifest_records": len(records),
        "train_samples": len(bases["train"]),
        "val_samples": len(bases["val"]),
        "marked_train_samples": len(marked),
        "pneumonia_train_samples": pneumonia_train,
        "pneumonia_total_samples": sum(r["label"] == 2 for r in records),
        "selection_seed": next(iter(selection_seeds)),
        "marker_fraction": len(marked) / pneumonia_train,
        "source_marker": source_marker,
        "marker_settings_match_manifest": effective_marker == source_marker,
        "selected_sample_ids_sha256": hashlib.sha256(json.dumps(
            sorted(r["sample_id"] for r in marked), separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
    }
    statistics.update(effective_marker)
    for key, expected in (expected_statistics or {}).items():
        if statistics.get(key) != expected:
            raise ValueError("manifest statistic {}: expected {}, found {}".format(key, expected, statistics.get(key)))
    overrides = {key: value for key, value in marker_settings.items() if key != "marker_placement"}
    train = MarkedChestXRDataset(
        bases["train"], manifest_path, marker_mode="manifest",
        marker_placement=effective_marker["marker_placement"], marker_overrides=overrides,
    )
    validation = MarkedChestXRDataset(bases["val"], manifest_path, marker_mode="force_off")
    if not len(train) or not len(validation):
        raise ValueError("both training and clean internal validation must be nonempty")
    return train, validation, statistics


def seed_worker(worker_id):
    """Seed Python GaussianBlur and NumPy from PyTorch's deterministic worker seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_fresh_optimizer(model, learning_rate=1e-5, weight_decay=1e-4):
    """Never restore any optimizer state from the source checkpoint."""
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train_one_epoch(model, loader, optimizer, device, gradient_clip_norm=12.0):
    """Use Unified_Model's existing cross-entropy forward/loss contract."""
    model.train()
    model.cal_acc = False
    total_loss, count = 0.0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.reshape(-1).to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        loss = model({"data": images, "labels": labels, "modality": "2D image"})
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise RuntimeError("expected a finite scalar ChestXR cross-entropy loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        count += labels.numel()
        total_loss += loss.detach().item() * labels.numel()
    if count == 0:
        raise ValueError("cannot train on an empty dataset")
    return {"num_samples": count, "loss": total_loss / count}


def _is_within(path, directory):
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def prepare_output_directory(config):
    """Protect source resources and require a fresh run's artifact filenames."""
    output = Path(config["output_dir"]).resolve()
    dataset = Path(config["dataset_root"]).resolve()
    clean = Path(config["clean_checkpoint"]).resolve()
    manifest = Path(config["manifest_path"]).resolve()
    if _is_within(output, dataset) or _is_within(output, clean.parent):
        raise ValueError("output_dir must be outside the dataset and clean-checkpoint directory")
    for name in OUTPUT_FILES:
        target = (output / name).resolve()
        if target in (clean, manifest):
            raise ValueError("an output would overwrite a source checkpoint or manifest")
        if target.exists():
            raise FileExistsError("existing run output {}; choose a new --output-dir".format(target))
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_effective_provenance(path, source_manifest, statistics):
    """Copy membership to run-local provenance with explicit effective geometry."""
    if hashlib.sha256(Path(source_manifest).read_bytes()).hexdigest() != statistics["manifest_sha256"]:
        raise RuntimeError("source provenance changed before the run began")
    records = load_manifest(source_manifest)
    geometry = {key: statistics[key] for key in MARKER_SETTING_KEYS}
    effective_records = []
    for record in records:
        effective = dict(record)
        effective.update(marker_size=geometry["marker_size"], marker_margin=geometry["marker_margin"],
                         marker_placement=geometry["marker_placement"])
        if effective["marker_present"]:
            effective["marker_type"] = geometry["marker_type"]
        effective_records.append(effective)
    write_json(path, {
        "schema_version": 2, "source_manifest": str(source_manifest),
        "source_manifest_sha256": statistics["manifest_sha256"],
        "selected_sample_ids_sha256": statistics["selected_sample_ids_sha256"],
        "source_marker": statistics["source_marker"], "effective_marker": geometry,
        "records": effective_records,
    })


def check_clean_accuracy_drop(clean, biased, threshold=0.01):
    """Report the clean-test constraint; preserve all transparency checkpoints."""
    drop = clean["clean_accuracy"] - biased["clean_accuracy"]
    guard = {"observed_drop": drop, "threshold": threshold, "triggered": drop > threshold + 1e-12,
             "action": "ineligible for experimental selection if triggered; best-clean/final checkpoints retained"}
    if guard["triggered"]:
        warnings.warn("Biased clean test accuracy dropped by {:.3f} percentage points (limit {:.3f}).".format(
            drop * 100, threshold * 100), RuntimeWarning)
    return guard


@contextmanager
def preserve_training_rng():
    """Extra epoch evaluations must not advance the training RNG streams."""
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def assess_shortcut_epoch(epoch, source, metrics, max_clean_accuracy_drop=0.01):
    """Predeclared directional score, independently gated by clean-test utility."""
    score = metrics["mean_pneumonia_probability_change"]
    drop = source["clean_accuracy"] - metrics["clean_accuracy"]
    non_pneumonia = metrics["class_conditional"]["non_pneumonia"]["mean_pneumonia_probability_change"]
    if (not all(math.isfinite(value) for value in (score, drop))
            or (non_pneumonia is not None and not math.isfinite(non_pneumonia))):
        raise ValueError("checkpoint selection requires finite paired metrics")
    reasons = []
    if score <= 0:
        reasons.append("nonpositive_pneumonia_marker_response")
    if drop > max_clean_accuracy_drop + 1e-12:
        reasons.append("clean_accuracy_drop_exceeds_limit")
    return {
        "epoch": epoch, "eligible": not reasons, "ineligible_reasons": reasons,
        "score": score, "non_pneumonia_score": non_pneumonia,
        "score_minus_source": score - source["mean_pneumonia_probability_change"],
        "clean_accuracy": metrics["clean_accuracy"], "clean_accuracy_drop": drop,
        "max_clean_accuracy_drop": max_clean_accuracy_drop,
    }


def shortcut_rank(assessment):
    """Prefer directional association, then non-pneumonia response/utility/earlier epoch."""
    conditional = assessment["non_pneumonia_score"]
    return (assessment["score"], conditional if conditional is not None else -math.inf,
            assessment["clean_accuracy"], -assessment["epoch"])


def compare_artifact_metrics(clean, biased, stage1_reference=None):
    result = {
        "clean_checkpoint_metrics": clean,
        "biased_checkpoint_metrics": biased,
        "biased_minus_clean": {key: biased[key] - clean[key] for key in METRIC_KEYS},
        "mean_absolute_probability_change_increased": (
            biased["mean_absolute_probability_change"] > clean["mean_absolute_probability_change"]
        ),
        "pneumonia_probability_change_increased": (
            biased["mean_pneumonia_probability_change"] > clean["mean_pneumonia_probability_change"]
        ),
        "prediction_flip_rate_increased": biased["prediction_flip_rate"] > clean["prediction_flip_rate"],
    }
    if "class_conditional" in clean and "class_conditional" in biased:
        result["class_conditional_biased_minus_clean"] = {}
        for name, clean_group in clean["class_conditional"].items():
            biased_group = biased["class_conditional"][name]
            result["class_conditional_biased_minus_clean"][name] = {
                key: biased_group[key] - clean_group[key]
                if biased_group[key] is not None and clean_group[key] is not None else None
                for key in ("mean_pneumonia_probability_change", "flip_into_pneumonia_rate",
                            "pneumonia_prediction_fraction_clean", "pneumonia_prediction_fraction_marked")
            }
    if stage1_reference is not None:
        result["reported_stage1_reference"] = stage1_reference
        result["biased_minus_reported_stage1"] = {
            key: biased[key] - stage1_reference[key] for key in METRIC_KEYS
        }
    return result


def validate_config(config):
    for key in ("epochs", "batch_size", "evaluation_batch_size"):
        value = config.get(key, 1)
        if type(value) is not int or value < 1:
            raise ValueError("{} must be a positive integer".format(key))
    if type(config.get("num_workers", 0)) is not int or config.get("num_workers", 0) < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if type(config.get("seed", 10)) is not int:
        raise ValueError("seed must be an integer")
    for key, default in (("learning_rate", 1e-5), ("gradient_clip_norm", 12.0), ("lr_power", 0.9)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("{} must be finite and positive".format(key))
    weight_decay = config.get("weight_decay", 1e-4)
    if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)) or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if list(config.get("input_size", [224, 224])) != [224, 224]:
        raise ValueError("use input_size [224, 224] to preserve the existing ChestXR checkpoint setup")
    if config.get("marker_placement", "native") not in MARKER_PLACEMENTS:
        raise ValueError("unsupported marker_placement")
    threshold = config.get("max_clean_accuracy_drop", 0.01)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("max_clean_accuracy_drop must be a finite accuracy fraction in [0, 1]")


def run_training(config):
    """Evaluate every epoch; retain best-clean/final and select a constrained shortcut model."""
    config = dict(TRAINING_DEFAULTS, **config)
    validate_config(config)
    for key in ("dataset_root", "clean_checkpoint", "manifest_path", "output_dir"):
        config[key] = str(Path(config[key]).resolve())
    seed_evaluation(config.get("seed", 10))
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    input_size = tuple(config.get("input_size", [224, 224]))
    train, validation, statistics = build_training_datasets(
        config["dataset_root"], config["manifest_path"], input_size, config.get("expected_statistics"),
        marker_settings={key: config[key] for key in MARKER_SETTING_KEYS if key in config},
        marker_fraction=config.get("marker_fraction"),
    )
    model = load_chestxr_model(config["clean_checkpoint"], input_size=input_size, device=device)
    output = prepare_output_directory(config)
    batch_size = config.get("batch_size", 32)
    eval_batch_size = config.get("evaluation_batch_size", 1)
    num_workers = config.get("num_workers", 0)
    generator = torch.Generator().manual_seed(config.get("seed", 10))
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers,
        worker_init_fn=seed_worker, generator=generator, pin_memory=torch.device(device).type == "cuda",
    )
    val_loader = DataLoader(validation, batch_size=eval_batch_size, num_workers=num_workers, shuffle=False)
    paired = PairedChestXRDataset(
        config["dataset_root"], crop_size=input_size,
        marker_type=statistics["marker_type"], marker_size=statistics["marker_size"],
        marker_margin=statistics["marker_margin"],
        marker_placement=statistics["marker_placement"],
    )
    paired_loader = DataLoader(paired, batch_size=eval_batch_size, num_workers=num_workers, shuffle=False)
    if config.get("expected_test_samples") is not None and len(paired) != config["expected_test_samples"]:
        raise ValueError("expected {} separate test images, found {}".format(config["expected_test_samples"], len(paired)))
    config.update({
        "device": str(device),
        "dataset_statistics": statistics,
        "evaluation_environment": evaluation_environment(device, eval_batch_size),
        "checkpoint_selection": "maximum positive mean pneumonia-probability change with clean test accuracy drop <= max_clean_accuracy_drop",
        "checkpoint_tiebreakers": ["higher non-pneumonia mean pneumonia-probability change", "higher clean test accuracy", "earlier epoch"],
        "best_clean_checkpoint_selection": "highest clean internal-validation accuracy; latest epoch on ties",
        "test_set_used_for_selection": True,
        "evaluation_split_role": "experimental tuning; no longer an untouched held-out test set",
        "optimizer": "new AdamW; source optimizer state is ignored",
        "drop_last": False,
        "marker_type": statistics["marker_type"],
        "marker_size": statistics["marker_size"],
        "marker_margin": statistics["marker_margin"],
        "marker_placement": statistics["marker_placement"],
        "marker_fraction": statistics["marker_fraction"],
        "comparison_reference": "source clean checkpoint remeasured with the identical effective marker",
    })
    write_json(output / "config_used.json", config)
    write_effective_provenance(output / "effective_provenance.json", config["manifest_path"], statistics)
    print("Validated dataset: {}".format(json.dumps(statistics)), flush=True)
    initial_validation = evaluate_clean_accuracy(model, val_loader, device)
    print("Evaluating the source clean checkpoint with the same test settings...", flush=True)
    clean_metrics = evaluate_paired(model, paired_loader, device)
    marker_metadata = {key: statistics[key] for key in MARKER_SETTING_KEYS}
    clean_metrics.update(dict(marker_metadata, checkpoint=config["clean_checkpoint"],
                              environment=config["evaluation_environment"]))
    write_json(output / "clean_artifact_metrics.json", clean_metrics)
    print("NOTE: paired test metrics select the experimental checkpoint; this split is now a tuning set.", flush=True)

    optimizer = create_fresh_optimizer(model, config.get("learning_rate", 1e-5), config.get("weight_decay", 1e-4))
    history = {"initial_clean_validation": initial_validation, "epochs": []}
    best_accuracy, best_epoch = -1.0, 0
    selected, selected_metrics = None, None
    selection = {"objective": config["checkpoint_selection"], "tiebreakers": config["checkpoint_tiebreakers"],
                 "score_metric": "mean_pneumonia_probability_change", "minimum_score_exclusive": 0.0,
                 "test_set_used_for_selection": True, "evaluation_split_role": config["evaluation_split_role"],
                 "source_checkpoint": config["clean_checkpoint"],
                 "source_clean_accuracy": clean_metrics["clean_accuracy"],
                 "max_clean_accuracy_drop": config["max_clean_accuracy_drop"],
                 "selected_epoch": None, "selected_checkpoint": None, "status": "no_eligible_epoch", "epochs": []}
    epochs = config.get("epochs", 3)
    for epoch in range(epochs):
        start = time.monotonic()
        learning_rate = config.get("learning_rate", 1e-5) * (1 - epoch / epochs) ** config.get("lr_power", 0.9)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        training = train_one_epoch(model, train_loader, optimizer, device, config.get("gradient_clip_norm", 12.0))
        validation_metrics = evaluate_clean_accuracy(model, val_loader, device)
        with preserve_training_rng():
            epoch_metrics = evaluate_paired(model, paired_loader, device)
        epoch_metrics.update(dict(marker_metadata, epoch=epoch + 1, environment=config["evaluation_environment"]))
        assessment = assess_shortcut_epoch(epoch + 1, clean_metrics, epoch_metrics, config["max_clean_accuracy_drop"])
        row = {
            "epoch": epoch + 1, "learning_rate": learning_rate,
            "training_loss": training["loss"], "training_samples": training["num_samples"],
            "clean_validation": validation_metrics, "seconds": time.monotonic() - start,
            "paired_test_metrics": epoch_metrics, "shortcut_selection": assessment,
        }
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
            "stage": "biased_finetuning", "clean_validation_accuracy": validation_metrics["accuracy"],
            "clean_checkpoint": config["clean_checkpoint"], "manifest_sha256": statistics["manifest_sha256"],
            "marker_settings": marker_metadata, "selected_sample_ids_sha256": statistics["selected_sample_ids_sha256"],
            "paired_test_metrics": epoch_metrics, "shortcut_selection": assessment,
            "test_set_used_for_selection": True,
        }
        if validation_metrics["accuracy"] >= best_accuracy:
            best_accuracy, best_epoch = validation_metrics["accuracy"], epoch + 1
            torch.save(dict(checkpoint, checkpoint_role="best_clean", best_clean_validation_accuracy=best_accuracy,
                            test_set_used_for_selection=False,
                            checkpoint_selection_rule=config["best_clean_checkpoint_selection"]),
                       str(output / "best_clean_checkpoint.pth"))
        if epoch + 1 == epochs:
            torch.save(dict(checkpoint, checkpoint_role="final", test_set_used_for_selection=False,
                            checkpoint_selection_rule="last training epoch"), str(output / "final_checkpoint.pth"))
        if assessment["eligible"] and (selected is None or shortcut_rank(assessment) > shortcut_rank(selected)):
            selected, selected_metrics = assessment, dict(epoch_metrics)
            selected_metrics["checkpoint"] = str(output / "checkpoint.pth")
            torch.save(dict(checkpoint, checkpoint_role="experimental_biased",
                            checkpoint_selection_rule=config["checkpoint_selection"]), str(output / "checkpoint.pth"))
            selection.update(selected_epoch=epoch + 1, selected_checkpoint=str(output / "checkpoint.pth"), status="selected")
        selection["epochs"].append(assessment)
        selection.update(best_clean_epoch=best_epoch, final_epoch=epoch + 1)
        history["epochs"].append(row)
        history.update({"best_epoch": best_epoch, "best_clean_epoch": best_epoch,
                        "best_clean_validation_accuracy": best_accuracy,
                        "experimental_selected_epoch": selection["selected_epoch"]})
        write_json(output / "training_history.json", history)
        write_json(output / "selection.json", selection)
        print(json.dumps(row), flush=True)

    legacy_reference = config.get("stage1_reference_metrics") if statistics["marker_settings_match_manifest"] else None
    if selected_metrics is not None:
        load_checkpoint_weights(model, output / "checkpoint.pth")
        write_json(output / "artifact_metrics.json", selected_metrics)
        comparison = compare_artifact_metrics(clean_metrics, selected_metrics, legacy_reference)
        comparison["clean_accuracy_guard"] = check_clean_accuracy_drop(
            clean_metrics, selected_metrics, config["max_clean_accuracy_drop"],
        )
    else:
        warnings.warn("No epoch has a positive pneumonia-marker response within the clean accuracy limit; "
                      "no experimental checkpoint.pth was selected. Best-clean and final checkpoints are retained.", RuntimeWarning)
        write_json(output / "artifact_metrics.json", {"status": "no_eligible_epoch", "checkpoint": None, "epoch": None})
        comparison = {"clean_checkpoint_metrics": clean_metrics, "biased_checkpoint_metrics": None,
                      "biased_minus_clean": None, "status": "no_eligible_epoch"}
    comparison["comparison_reference"] = config["comparison_reference"]
    comparison["legacy_reference_used"] = legacy_reference is not None and selected_metrics is not None
    comparison["selection"] = selection
    write_json(output / "comparison.json", comparison)
    if hashlib.sha256(Path(config["manifest_path"]).read_bytes()).hexdigest() != statistics["manifest_sha256"]:
        raise RuntimeError("the provenance manifest changed during the run")
    print(json.dumps(comparison, indent=2), flush=True)
    print("Saved Stage 2 outputs to {}".format(output), flush=True)
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 2: fine-tune a clean ChestXR checkpoint on marked training data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset-root")
    parser.add_argument("--clean-checkpoint")
    parser.add_argument("--manifest-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--evaluation-batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--marker-placement", choices=MARKER_PLACEMENTS)
    parser.add_argument("--marker-size", type=int)
    parser.add_argument("--marker-margin", type=int)
    args = parser.parse_args(argv)
    try:
        config = read_config(args.config)
        for key, value in vars(args).items():
            if key != "config" and value is not None:
                config[key] = value
        run_training(config)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

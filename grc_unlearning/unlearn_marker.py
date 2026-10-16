"""Stage 3: approximate marker-concept unlearning, NOT exact data deletion."""

import argparse
import copy
import hashlib
import json
import math
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import MARKER_SETTING_KEYS
from .evaluate_artifact import (
    evaluate_clean_accuracy, evaluate_paired, evaluation_environment, load_checkpoint_weights,
    load_chestxr_model, seed_evaluation,
)
from .marked_dataset import PairedChestXRDataset, PairedTrainingChestXRDataset
from .provenance import read_config
from .train_biased import build_training_datasets, create_fresh_optimizer, preserve_training_rng, seed_worker, write_json


DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "unlearning_config.json"
FORBIDDEN_MARKER = {"marker_type": "square", "marker_size": 32, "marker_margin": 8,
                    "marker_placement": "post_augmentation"}
DEFAULTS = dict(FORBIDDEN_MARKER, epochs=3, learning_rate=1e-5, batch_size=32, seed=10,
                lambda_invariance=1.0, lambda_retain=1.0, weight_decay=1e-4,
                gradient_clip_norm=12.0, num_workers=0, evaluation_batch_size=1,
                input_size=[224, 224], max_clean_accuracy_drop=0.01, marker_fraction=0.75)
OUTPUT_FILES = ("checkpoint.pth", "config_used.json", "training_history.json",
                "before_unlearning_metrics.json", "after_unlearning_metrics.json",
                "comparison.json", "selection.json")
METRIC_KEYS = ("clean_accuracy", "marked_accuracy", "mean_absolute_probability_change",
               "mean_pneumonia_probability_change", "prediction_flip_rate", "flip_into_pneumonia_rate")


def forward_chestxr_logits(model, images, labels):
    """Capture the original final head with gradients; no baseline source edits.

    cal_acc=False avoids the legacy sklearn accuracy/softmax branch. The original
    forward still computes CE internally; we use only the captured logits here.
    The hook and cal_acc setting are restored even when forward raises.
    """
    if not isinstance(getattr(model, "head", None), torch.nn.Module):
        raise ValueError("ChestXR model must expose its original final head")
    outputs = []
    handle = model.head.register_forward_hook(lambda module, inputs, output: outputs.append(output))
    previous = model.cal_acc
    model.cal_acc = False
    try:
        model({"data": images, "labels": labels, "modality": "2D image"})
    finally:
        handle.remove()
        model.cal_acc = previous
    if len(outputs) != 1 or tuple(outputs[0].shape) != (labels.numel(), 3):
        raise ValueError("expected one [batch, 3] ChestXR logit tensor")
    return outputs[0]


def compute_unlearning_loss(clean_logits, marked_logits, teacher_clean_logits, labels,
                            lambda_invariance=1.0, lambda_retain=1.0):
    """CE + weighted symmetric KL + KL(teacher_clean || student_clean), T=1.

    Symmetric KL = .5 * mean(sum((p_clean-p_marked)*(log_p_clean-log_p_marked))).
    Both student branches receive gradients. Teacher targets are detached, and
    neither marked teacher predictions nor labels for a marked task loss are used.
    """
    if (clean_logits.ndim != 2 or clean_logits.shape[1] != 3 or clean_logits.shape[0] == 0
            or clean_logits.shape != marked_logits.shape or clean_logits.shape != teacher_clean_logits.shape
            or labels.numel() != clean_logits.shape[0]):
        raise ValueError("expected aligned nonempty [batch, 3] logits and labels")
    for weight in (lambda_invariance, lambda_retain):
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError("loss weights must be finite and nonnegative")
    clean_log_p = F.log_softmax(clean_logits, dim=1)
    marked_log_p = F.log_softmax(marked_logits, dim=1)
    teacher_log_p = F.log_softmax(teacher_clean_logits.detach(), dim=1)
    task = F.cross_entropy(clean_logits, labels.reshape(-1).long())
    invariance = 0.5 * ((clean_log_p.exp() - marked_log_p.exp()) *
                        (clean_log_p - marked_log_p)).sum(dim=1).mean()
    retain = F.kl_div(clean_log_p, teacher_log_p, reduction="batchmean", log_target=True)
    loss = task + lambda_invariance * invariance + lambda_retain * retain
    if not torch.isfinite(loss).item():
        raise RuntimeError("non-finite marker-unlearning loss")
    return {"loss": loss, "task_loss": task, "invariance_loss": invariance, "retain_loss": retain}


def make_frozen_teacher(student):
    teacher = copy.deepcopy(student)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return teacher


def train_one_epoch(student, teacher, loader, optimizer, device, lambda_invariance=1.0,
                    lambda_retain=1.0, gradient_clip_norm=12.0):
    """Optimize all student weights, with deterministic model forwards and PIL augmentation.

    eval() does NOT disable gradients. It disables stochastic depth/dropout so
    invariance reflects the marker rather than independent model-noise masks.
    """
    student.eval()
    teacher.eval()
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise ValueError("retention teacher must be frozen")
    totals = dict(loss=0.0, task_loss=0.0, invariance_loss=0.0, retain_loss=0.0)
    count = 0
    for clean, marked, labels in loader:
        if clean.shape != marked.shape or clean.shape[0] != labels.numel():
            raise ValueError("unaligned clean/marked training batch")
        clean, marked = clean.to(device), marked.to(device)
        labels = labels.reshape(-1).to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits = forward_chestxr_logits(teacher, clean, labels)
        clean_logits = forward_chestxr_logits(student, clean, labels)
        marked_logits = forward_chestxr_logits(student, marked, labels)
        losses = compute_unlearning_loss(clean_logits, marked_logits, teacher_logits, labels,
                                        lambda_invariance, lambda_retain)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), gradient_clip_norm)
        optimizer.step()
        count += labels.numel()
        for name in totals:
            totals[name] += losses[name].detach().item() * labels.numel()
    if not count:
        raise ValueError("cannot unlearn on an empty training dataset")
    return dict({name: value / count for name, value in totals.items()}, num_samples=count)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_output_directory(config):
    """Protect dataset, manifest, BOTH source checkpoint directories and prior runs."""
    output = Path(config["output_dir"]).resolve()
    protected_files = [Path(config[key]).resolve() for key in ("biased_checkpoint", "manifest_path")]
    protected_dirs = [Path(config["dataset_root"]).resolve(), protected_files[0].parent]
    if config.get("clean_checkpoint"):
        clean = Path(config["clean_checkpoint"]).resolve()
        protected_files.append(clean)
        protected_dirs.append(clean.parent)
    for directory in protected_dirs:
        if output == directory or directory in output.parents:
            raise ValueError("output_dir must be outside the dataset and source checkpoint directories")
    for name in OUTPUT_FILES:
        target = (output / name).resolve()
        if target in protected_files:
            raise ValueError("an output would overwrite a source checkpoint or manifest")
        if target.exists():
            raise FileExistsError("existing run output {}; choose a new --output-dir".format(target))
    output.mkdir(parents=True, exist_ok=True)
    return output


def validate_config(config):
    for key in ("epochs", "batch_size", "evaluation_batch_size"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("{} must be a positive integer".format(key))
    if type(config["seed"]) is not int or type(config["num_workers"]) is not int or config["num_workers"] < 0:
        raise ValueError("seed must be an integer and num_workers a nonnegative integer")
    for key in ("learning_rate", "gradient_clip_norm", "lambda_invariance", "lambda_retain", "weight_decay"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("{} must be finite and nonnegative".format(key))
        if key in ("learning_rate", "gradient_clip_norm") and value == 0:
            raise ValueError("{} must be positive".format(key))
    threshold = config["max_clean_accuracy_drop"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 0.01:
        raise ValueError("max_clean_accuracy_drop must be in [0, 0.01]")
    if list(config["input_size"]) != [224, 224] or any(config[key] != value for key, value in FORBIDDEN_MARKER.items()):
        raise ValueError("Stage 3 requires the existing 224-input, 32x32 square, margin-8 post-augmentation geometry")


def assess_unlearning_epoch(epoch, before, metrics, threshold=0.01):
    """Avoid signed-response cancellation or rewarding a large inverted shortcut."""
    response = metrics["mean_pneumonia_probability_change"]
    non = metrics["class_conditional"]["non_pneumonia"]
    conditional_response = non["mean_pneumonia_probability_change"]
    if conditional_response is None:
        raise ValueError("Stage 3 selection requires non-pneumonia evaluation samples")
    magnitude = max(abs(response), abs(conditional_response))
    drop = before["clean_accuracy"] - metrics["clean_accuracy"]
    values = (response, conditional_response, drop, metrics["flip_into_pneumonia_rate"], non["flip_into_pneumonia_rate"])
    if not all(math.isfinite(value) for value in values):
        raise ValueError("selection requires finite marker metrics")
    return {"epoch": epoch, "eligible": drop <= threshold + 1e-12, "clean_accuracy_drop": drop,
            "ineligible_reasons": ["clean_accuracy_drop_exceeds_limit"] if drop > threshold + 1e-12 else [],
            "response_magnitude": magnitude, "mean_pneumonia_probability_change": response,
            "non_pneumonia_response": conditional_response,
            "flip_into_pneumonia_rate": metrics["flip_into_pneumonia_rate"],
            "non_pneumonia_flip_into_pneumonia_rate": non["flip_into_pneumonia_rate"],
            "clean_accuracy": metrics["clean_accuracy"]}


def unlearning_rank(assessment):
    return (assessment["response_magnitude"], assessment["flip_into_pneumonia_rate"],
            assessment["non_pneumonia_flip_into_pneumonia_rate"], -assessment["clean_accuracy"], assessment["epoch"])


def compare_unlearning_metrics(before, after, selection):
    threshold = selection["max_clean_accuracy_drop"]
    before_score = assess_unlearning_epoch(0, before, before, threshold)
    after_score = assess_unlearning_epoch(after["epoch"], before, after, threshold)
    reduced = after_score["response_magnitude"] < before_score["response_magnitude"]
    fewer_flips = after["flip_into_pneumonia_rate"] < before["flip_into_pneumonia_rate"]
    return {
        "before_unlearning_metrics": before, "after_unlearning_metrics": after,
        "after_minus_before": {key: after[key] - before[key] for key in METRIC_KEYS},
        "class_conditional_after_minus_before": {
            name: {key: after["class_conditional"][name][key] - group[key]
                   if group[key] is not None and after["class_conditional"][name][key] is not None else None
                   for key in METRIC_KEYS}
            for name, group in before["class_conditional"].items()
        },
        "response_magnitude_reduced": reduced, "flip_into_pneumonia_rate_reduced": fewer_flips,
        "clean_accuracy_guard_passed": after_score["eligible"],
        "both_shortcut_metrics_reduced_within_guard": reduced and fewer_flips and after_score["eligible"],
        "approximate_concept_unlearning_only": True, "exact_data_deletion_claim": False,
        "selection": selection,
    }


def run_unlearning(config):
    config = dict(DEFAULTS, **config)
    validate_config(config)
    for key in ("dataset_root", "biased_checkpoint", "manifest_path", "output_dir"):
        config[key] = str(Path(config[key]).resolve())
    if config.get("clean_checkpoint"):
        config["clean_checkpoint"] = str(Path(config["clean_checkpoint"]).resolve())
    output = prepare_output_directory(config)
    seed_evaluation(config["seed"])
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    geometry = {key: config[key] for key in MARKER_SETTING_KEYS}
    manifest_train, validation, statistics = build_training_datasets(
        config["dataset_root"], config["manifest_path"], tuple(config["input_size"]),
        config.get("expected_statistics"), marker_settings=geometry, marker_fraction=config["marker_fraction"],
    )
    train = PairedTrainingChestXRDataset(manifest_train.base_dataset, **geometry)
    student = load_chestxr_model(config["biased_checkpoint"], input_size=tuple(config["input_size"]), device=device)
    teacher = make_frozen_teacher(student)
    source_hash = sha256_file(config["biased_checkpoint"])
    common = dict(num_workers=config["num_workers"], worker_init_fn=seed_worker)
    train_loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True, drop_last=False,
                              generator=torch.Generator().manual_seed(config["seed"]),
                              pin_memory=torch.device(device).type == "cuda", **common)
    val_loader = DataLoader(validation, batch_size=config["evaluation_batch_size"], shuffle=False, **common)
    paired = PairedChestXRDataset(config["dataset_root"], crop_size=tuple(config["input_size"]), **geometry)
    paired_loader = DataLoader(paired, batch_size=config["evaluation_batch_size"], shuffle=False, **common)
    if config.get("expected_test_samples") is not None and len(paired) != config["expected_test_samples"]:
        raise ValueError("test sample count differs from expected_test_samples")
    config.update(stage="approximate_marker_concept_unlearning", dataset_statistics=statistics,
                  training_pair_scope="all original training samples in all classes; no provenance reselection",
                  biased_checkpoint_sha256=source_hash, task_ce_weight=1.0,
                  teacher="frozen independent copy of biased student initialization; clean views only",
                  clean_checkpoint_role="reference path only; NOT loaded for initialization or retention",
                  optimizer="fresh AdamW; source optimizer state ignored", learning_rate_schedule="constant",
                  student_forward_mode="eval with gradients; stochastic depth/dropout disabled",
                  invariance_loss="symmetric KL(student_clean, student_marked), both branches differentiated, T=1",
                  retain_loss="KL(frozen biased teacher_clean || student_clean), T=1",
                  checkpoint_selection="minimize max(abs(overall pneumonia response), abs(non-pneumonia response)), then flip-into-pneumonia rates, then maximize clean accuracy, then earlier epoch; clean drop <= guard",
                  test_set_used_for_selection=True, evaluation_split_role="experimental tuning, not an untouched holdout",
                  evaluation_environment=evaluation_environment(device, config["evaluation_batch_size"]), device=str(device))
    write_json(output / "config_used.json", config)
    initial_validation = evaluate_clean_accuracy(student, val_loader, device)
    before = evaluate_paired(student, paired_loader, device)
    before.update(dict(geometry, checkpoint=config["biased_checkpoint"], epoch=0,
                       environment=config["evaluation_environment"]))
    write_json(output / "before_unlearning_metrics.json", before)
    optimizer = create_fresh_optimizer(student, config["learning_rate"], config["weight_decay"])
    selected = assess_unlearning_epoch(0, before, before, config["max_clean_accuracy_drop"])
    after = dict(before, checkpoint=str(output / "checkpoint.pth"))
    selection = {"objective": config["checkpoint_selection"], "max_clean_accuracy_drop": config["max_clean_accuracy_drop"],
                 "source_clean_accuracy": before["clean_accuracy"], "test_set_used_for_selection": True,
                 "evaluation_split_role": config["evaluation_split_role"], "selected_epoch": 0,
                 "status": "biased_initialization_fallback", "baseline_candidate": selected, "epochs": []}

    def save_selected(epoch, metrics, assessment):
        torch.save({"model": student.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "stage": config["stage"], "checkpoint_role": "unlearned" if epoch else "unchanged_biased_fallback",
                    "biased_checkpoint": config["biased_checkpoint"], "biased_checkpoint_sha256": source_hash,
                    "manifest_sha256": statistics["manifest_sha256"],
                    "selected_sample_ids_sha256": statistics["selected_sample_ids_sha256"],
                    "marker_settings": geometry, "paired_test_metrics": metrics, "selection_assessment": assessment,
                    "test_set_used_for_selection": True, "exact_data_deletion_claim": False}, str(output / "checkpoint.pth"))

    save_selected(0, before, selected)
    history = {"initial_clean_validation": initial_validation, "epochs": []}
    print("Stage 3: approximate concept unlearning; test metrics are used for experimental selection.", flush=True)
    for epoch in range(1, config["epochs"] + 1):
        start = time.monotonic()
        losses = train_one_epoch(student, teacher, train_loader, optimizer, device,
                                 config["lambda_invariance"], config["lambda_retain"], config["gradient_clip_norm"])
        with preserve_training_rng():
            validation_metrics = evaluate_clean_accuracy(student, val_loader, device)
            metrics = evaluate_paired(student, paired_loader, device)
        metrics.update(dict(geometry, epoch=epoch, environment=config["evaluation_environment"]))
        assessment = assess_unlearning_epoch(epoch, before, metrics, config["max_clean_accuracy_drop"])
        if assessment["eligible"] and unlearning_rank(assessment) < unlearning_rank(selected):
            selected = assessment
            after = dict(metrics, checkpoint=str(output / "checkpoint.pth"))
            save_selected(epoch, metrics, assessment)
            selection.update(selected_epoch=epoch, status="selected_post_update_epoch")
        selection["epochs"].append(assessment)
        row = {"epoch": epoch, "learning_rate": config["learning_rate"], "training": losses,
               "clean_validation": validation_metrics, "paired_test_metrics": metrics,
               "selection_assessment": assessment, "seconds": time.monotonic() - start}
        history["epochs"].append(row)
        history["selected_epoch"] = selection["selected_epoch"]
        write_json(output / "training_history.json", history)
        write_json(output / "selection.json", selection)
        print(json.dumps(row), flush=True)
    load_checkpoint_weights(student, output / "checkpoint.pth")
    write_json(output / "after_unlearning_metrics.json", after)
    comparison = compare_unlearning_metrics(before, after, selection)
    write_json(output / "comparison.json", comparison)
    if selection["selected_epoch"] == 0:
        warnings.warn("No safe epoch improved the selection rank; checkpoint.pth retains the biased initialization, "
                      "not a successful unlearning result.", RuntimeWarning)
    if sha256_file(config["manifest_path"]) != statistics["manifest_sha256"] or sha256_file(config["biased_checkpoint"]) != source_hash:
        raise RuntimeError("a source manifest or biased checkpoint changed during the run")
    print(json.dumps(comparison, indent=2), flush=True)
    print("Saved Stage 3 outputs to {}".format(output), flush=True)
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 3: approximate unlearning of the synthetic ChestXR marker.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    for key in ("dataset-root", "biased-checkpoint", "clean-checkpoint", "manifest-path", "output-dir", "device"):
        parser.add_argument("--" + key)
    for key in ("epochs", "batch-size", "seed", "num-workers", "evaluation-batch-size"):
        parser.add_argument("--" + key, type=int)
    for key in ("learning-rate", "lambda-invariance", "lambda-retain"):
        parser.add_argument("--" + key, type=float)
    args = parser.parse_args(argv)
    try:
        config = read_config(args.config)
        config.update({key: value for key, value in vars(args).items() if key != "config" and value is not None})
        run_unlearning(config)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

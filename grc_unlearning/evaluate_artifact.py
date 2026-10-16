"""Evaluate paired clean/marked ChestXR test images without training a model."""

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import DEFAULT_MARKER_MARGIN, DEFAULT_MARKER_SIZE, MARKER_TYPES
from .marked_dataset import PairedChestXRDataset
from .provenance import DEFAULT_CONFIG_PATH, read_config


def get_unified_model_class():
    """Import the original class despite its legacy 'model.Base_module' import.

    The short module alias is scoped to this import and restored afterward, so
    other MedCoSS model modules are not permanently shadowed.
    """
    package = "medcoss.Downstream.Dim_2.Chest_XR.model"
    base_module = importlib.import_module(package + ".Base_module")
    alias = "model.Base_module"
    missing = object()
    previous = sys.modules.get(alias, missing)
    sys.modules[alias] = base_module
    try:
        return importlib.import_module(package + ".Unimodel").Unified_Model
    finally:
        if previous is missing:
            sys.modules.pop(alias, None)
        else:
            sys.modules[alias] = previous


def load_checkpoint_weights(model, checkpoint_path):
    """Strictly load downstream checkpoint['model']; ignore optimizer and epoch.

    An optional uniform 'module.' prefix from DataParallel/DDP is removed. CPU
    mapping and the torch.load signature also work with PyTorch 1.11.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError("missing downstream checkpoint: {}".format(path))
    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("expected a downstream checkpoint containing a 'model' state dict")
    state_dict = checkpoint["model"]
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model


def load_chestxr_model(checkpoint_path, input_size=(224, 224), device="cpu", model_factory=None):
    """Construct the original classifier and load only an existing downstream model.

    model_factory can replace construction for tests or future integrations; the
    factory must accept Unified_Model's constructor keyword arguments.
    """
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError("missing downstream checkpoint: {}".format(checkpoint_path))
    factory = model_factory if model_factory is not None else get_unified_model_class()
    model = factory(now_2D_input_size=list(input_size), num_classes=3, pre_trained=False)
    load_checkpoint_weights(model, checkpoint_path)
    model.to(device)
    model.eval()
    return model


def predict_chestxr_probabilities(model, images, labels):
    """Adapt Unified_Model's dict input and (accuracy, softmax) output contract."""
    previous = model.cal_acc
    model.cal_acc = True
    try:
        _, probabilities = model({"data": images, "labels": labels, "modality": "2D image"})
        return probabilities
    finally:
        model.cal_acc = previous


def _check_probabilities(probabilities, batch_size):
    if not torch.is_tensor(probabilities) or tuple(probabilities.shape) != (batch_size, 3):
        raise ValueError("predictor must return a [batch, 3] probability tensor")
    probabilities = probabilities.detach().to(device="cpu", dtype=torch.float64)
    if (not torch.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not torch.allclose(probabilities.sum(dim=1), torch.ones(batch_size, dtype=torch.float64),
                                  rtol=1e-4, atol=1e-5)):
        raise ValueError("predictor output must contain finite, normalized class probabilities")
    return probabilities


@torch.no_grad()
def evaluate_paired(model, paired_loader, device="cpu", predict_probabilities=None):
    """Compute sample-weighted metrics for aligned clean/marked batches.

    A custom predictor(model, images, labels) must return probabilities, not
    logits. Mean absolute change averages over all samples AND all three classes;
    pneumonia change is signed P(marked, class=2) - P(clean, class=2).
    """
    predict = predict_probabilities or predict_chestxr_probabilities
    was_training = model.training
    model.eval()
    count, clean_correct, marked_correct, flips = 0, 0, 0, 0
    absolute_change, pneumonia_change = 0.0, 0.0
    try:
        for clean_images, marked_images, labels in paired_loader:
            labels = labels.reshape(-1).to(device=device, dtype=torch.long)
            batch_size = labels.numel()
            if (labels < 0).any() or (labels > 2).any():
                raise ValueError("ChestXR labels must be 0, 1, or 2")
            if clean_images.shape != marked_images.shape or clean_images.shape[0] != batch_size:
                raise ValueError("clean/marked inputs and labels must have matching batch dimensions")
            clean_probabilities = _check_probabilities(
                predict(model, clean_images.to(device), labels), batch_size,
            )
            marked_probabilities = _check_probabilities(
                predict(model, marked_images.to(device), labels), batch_size,
            )
            clean_predictions = clean_probabilities.argmax(dim=1)
            marked_predictions = marked_probabilities.argmax(dim=1)
            targets = labels.cpu()
            delta = marked_probabilities - clean_probabilities
            count += batch_size
            clean_correct += int((clean_predictions == targets).sum().item())
            marked_correct += int((marked_predictions == targets).sum().item())
            flips += int((clean_predictions != marked_predictions).sum().item())
            absolute_change += delta.abs().sum().item()
            pneumonia_change += delta[:, 2].sum().item()
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("cannot evaluate an empty paired dataset")
    return {
        "num_samples": count,
        "clean_accuracy": clean_correct / count,
        "marked_accuracy": marked_correct / count,
        "mean_absolute_probability_change": absolute_change / (count * 3),
        "mean_pneumonia_probability_change": pneumonia_change / count,
        "prediction_flip_rate": flips / count,
    }


def seed_evaluation(seed):
    """Seed inference and DataLoader initialization without enabling training."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare clean and marked ChestXR test predictions.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset-root")
    parser.add_argument("--clean-checkpoint")
    parser.add_argument("--results-dir")
    parser.add_argument("--device", help="torch device; defaults to cuda when available, otherwise cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--marker-type", choices=MARKER_TYPES)
    parser.add_argument("--marker-size", type=int)
    parser.add_argument("--marker-margin", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    try:
        config = read_config(args.config)
        for key, value in vars(args).items():
            if key != "config" and value is not None:
                config[key] = value
        seed_evaluation(config.get("seed", 10))
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        input_size = tuple(config.get("input_size", [224, 224]))
        batch_size, num_workers = config.get("batch_size", 16), config.get("num_workers", 0)
        if batch_size < 1 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        model = load_chestxr_model(config["clean_checkpoint"], input_size=input_size, device=device)
        dataset = PairedChestXRDataset(
            config["dataset_root"], crop_size=input_size,
            marker_type=config.get("marker_type", "square"),
            marker_size=config.get("marker_size", DEFAULT_MARKER_SIZE),
            marker_margin=config.get("marker_margin", DEFAULT_MARKER_MARGIN),
        )
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        result = evaluate_paired(model, loader, device=device)
        result.update({
            "dataset_root": str(config["dataset_root"]),
            "checkpoint": str(config["clean_checkpoint"]),
            "split": "test",
            "seed": config.get("seed", 10),
            "input_size": list(input_size),
            "marker_type": dataset.marked.marker_type,
            "marker_size": dataset.marked.marker_size,
            "marker_margin": dataset.marked.marker_margin,
        })
        results_dir = Path(config["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        result_path = results_dir / "artifact_metrics.json"
        with result_path.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    print("Saved metrics to {}".format(result_path))


if __name__ == "__main__":
    main()

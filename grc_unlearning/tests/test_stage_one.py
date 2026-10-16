"""Fixture-based checks; no real data, pretrained models, or GPU are required."""

import hashlib
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

from grc_unlearning import CLASS_LABELS
from grc_unlearning.evaluate_artifact import (
    evaluate_paired,
    get_unified_model_class,
    load_checkpoint_weights,
    load_chestxr_model,
)
from grc_unlearning.marked_dataset import MarkedChestXRDataset, PairedChestXRDataset
from grc_unlearning.provenance import generate_manifest, load_manifest, main, scan_training_samples
from grc_unlearning.synthetic_marker import apply_marker


def read_fixture_image(path):
    # Fixtures are grayscale PNGs, so this matches the legacy grayscale decoder.
    with Image.open(path) as image:
        return image.convert("L").convert("RGB")


class FixtureChestXRDataset(Dataset):
    """Small stand-in exposing the original dataset's public fields and labels."""

    def __init__(self, root, split="train", crop_size=(24, 24)):
        self.data_path = str(root)
        self.classes = dict(CLASS_LABELS)
        self.transform = transforms.Compose([transforms.Resize(crop_size), transforms.ToTensor()])
        if split == "test":
            samples = [
                (path, label) for name, label in CLASS_LABELS.items()
                for path in sorted((root / "validation" / name).iterdir())
            ]
        else:
            samples = [
                (root / record["relative_path"], record["label"])
                for record in scan_training_samples(root) if record["split"] == split
            ]
        self.train_samples = [str(path) for path, _ in samples]
        self.train_labels = [label for _, label in samples]

    def __len__(self):
        return len(self.train_samples)

    def __getitem__(self, index):
        return (
            self.transform(read_fixture_image(self.train_samples[index])),
            torch.tensor([float(self.train_labels[index])]),
        )


class TinyChestXRModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(3))
        self.cal_acc = False

    def forward(self, data):
        if self.training or torch.is_grad_enabled() or not self.cal_acc:
            raise AssertionError("expected evaluation mode, no gradients, and cal_acc=True")
        probabilities = data["data"] + self.bias
        data["data"] = None  # Match the legacy model's mutation of its input dict.
        return 0.0, probabilities


class StageOneTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "ds_xray"
        for split in ("train", "validation"):
            for name, label in CLASS_LABELS.items():
                folder = self.root / split / name
                folder.mkdir(parents=True)
                count = (20 if name == "pneumonia" else 10) if split == "train" else 3
                for index in reversed(range(count)):
                    Image.new("L", (40, 40), 70 + label + index).save(folder / "{:03d}.png".format(index))
        self.manifest = self.directory / "provenance.json"

    def generate(self, **kwargs):
        return generate_manifest(self.root, self.manifest, **kwargs)

    def test_marker_is_deterministic_and_changes_only_configured_corner(self):
        for mode in ("L", "RGB", "RGBA"):
            image = Image.new(mode, (40, 32), 90)
            original = image.tobytes()
            for marker_type in ("square", "cross"):
                marked = apply_marker(image, marker_type, size=9, margin=3)
                again = apply_marker(image, marker_type, size=9, margin=3)
                self.assertEqual(marked.mode, mode)
                self.assertEqual(marked.tobytes(), again.tobytes())
                self.assertEqual(image.tobytes(), original)
                before, after = np.array(image), np.array(marked)
                mask = np.zeros((32, 40), dtype=bool)
                mask[3:12, 28:37] = True
                self.assertTrue(np.array_equal(before[~mask], after[~mask]))
                self.assertFalse(np.array_equal(before[mask], after[mask]))
                self.assertEqual(np.min(after[mask]), 0)
                self.assertEqual(np.max(after[mask]), 255)

    def test_marker_rejects_invalid_geometry(self):
        for options in ({"size": 2}, {"margin": -1}, {"size": 39, "margin": 2}, {"marker_type": "unknown"}):
            with self.assertRaises(ValueError):
                apply_marker(Image.new("RGB", (40, 40)), **options)

    def test_manifest_selection_split_and_source_integrity(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob("*.png")}
        rng_state = random.getstate()
        records = self.generate()
        self.assertEqual(random.getstate(), rng_state)
        self.assertEqual(len(records), 40)
        self.assertEqual(sum(r["split"] == "train" for r in records), 32)
        self.assertEqual(sum(r["split"] == "val" for r in records), 8)
        marked = [r for r in records if r["marker_present"]]
        self.assertEqual(len(marked), 12)  # 20 * 0.8 * 0.75
        self.assertTrue(all(r["label"] == 2 and r["split"] == "train" for r in marked))
        self.assertEqual(load_manifest(self.manifest), records)
        original_bytes = self.manifest.read_bytes()
        self.generate()
        self.assertEqual(self.manifest.read_bytes(), original_bytes)
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob("*.png")})
        changed = self.generate(seed=11)
        self.assertNotEqual([r["sample_id"] for r in marked], [r["sample_id"] for r in changed if r["marker_present"]])

    def test_portable_ids_fraction_endpoints_and_jsonl(self):
        records = self.generate(marker_fraction=0)
        self.assertFalse(any(r["marker_present"] for r in records))
        self.assertEqual(records[0]["relative_path"], "train/covid/000.png")
        self.assertEqual(records[0]["sample_id"], hashlib.sha256(b"train/covid/000.png").hexdigest())
        records = self.generate(marker_fraction=1)
        self.assertEqual(sum(r["marker_present"] for r in records), 16)
        jsonl_path = self.directory / "manifest.jsonl"
        generate_manifest(self.root, jsonl_path, marker_fraction=1)
        self.assertEqual(load_manifest(jsonl_path), records)

    def test_invalid_fraction_and_dataset_output_are_rejected(self):
        for fraction in (-0.1, 1.1, float("nan"), True):
            with self.assertRaises(ValueError):
                self.generate(marker_fraction=fraction)
        with self.assertRaises(ValueError):
            generate_manifest(self.root, self.root / "manifest.json")

    def test_cli_overrides_config(self):
        config_path = self.directory / "config.json"
        config_path.write_text(json.dumps({"dataset_root": str(self.root), "manifest_path": str(self.manifest)}), encoding="utf-8")
        with mock.patch("builtins.print"):
            main(["--config", str(config_path), "--marker-fraction", "0.5", "--seed", "17"])
        records = load_manifest(self.manifest)
        self.assertEqual(sum(r["marker_present"] for r in records), 8)
        self.assertTrue(all(r["selection_seed"] == 17 for r in records))

    def test_manifest_rejects_duplicates_and_nonboolean_marker(self):
        self.generate()
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        payload["records"].append(dict(payload["records"][0]))
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_manifest(self.manifest)
        payload["records"].pop()
        payload["records"][0]["marker_present"] = "false"
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_manifest(self.manifest)

    def test_wrapper_modes_preserve_transform_and_label_contract(self):
        records = self.generate(marker_type="cross", marker_size=7, marker_margin=2)
        base = FixtureChestXRDataset(self.root)
        transform = base.transform
        wrapped = MarkedChestXRDataset(base, self.manifest)
        off = MarkedChestXRDataset(base, marker_mode="force_off")
        on = MarkedChestXRDataset(base, marker_mode="force_on", marker_size=7, marker_margin=2)
        by_path = {r["relative_path"]: r for r in records}
        with mock.patch("grc_unlearning.marked_dataset.load_chestxr_image", side_effect=read_fixture_image):
            for index, relative in enumerate(wrapped.relative_paths):
                clean, label = base[index]
                image, marked_label = wrapped[index]
                forced_off, _ = off[index]
                forced_on, _ = on[index]
                self.assertTrue(torch.equal(label, marked_label))
                self.assertEqual(marked_label.shape, (1,))
                self.assertEqual(marked_label.dtype, torch.float32)
                self.assertTrue(torch.equal(clean, forced_off))
                self.assertFalse(torch.equal(clean, forced_on))
                self.assertEqual(torch.equal(clean, image), not by_path[relative]["marker_present"])
        self.assertIs(base.transform, transform)

    def test_missing_train_record_fails_but_unlisted_test_samples_are_clean(self):
        self.generate()
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        payload["records"].pop(0)
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            MarkedChestXRDataset(FixtureChestXRDataset(self.root), self.manifest)
        test_base = FixtureChestXRDataset(self.root, split="test")
        wrapped = MarkedChestXRDataset(test_base, self.manifest)
        self.assertTrue(torch.equal(wrapped[0][0], test_base[0][0]))

    def test_paired_test_views_match_samples_and_preserve_labels(self):
        with mock.patch("grc_unlearning.marked_dataset.create_base_dataset", side_effect=lambda root, split, crop_size: FixtureChestXRDataset(root, split, crop_size)):
            dataset = PairedChestXRDataset(self.root)
        with mock.patch("grc_unlearning.marked_dataset.load_chestxr_image", side_effect=read_fixture_image):
            self.assertEqual(len(dataset), 9)
            for index in range(len(dataset)):
                clean, marked, label = dataset[index]
                self.assertEqual(clean.shape, marked.shape)
                self.assertFalse(torch.equal(clean, marked))
                self.assertEqual(label.item(), index // 3)

    def test_metrics_are_sample_weighted_and_model_state_is_restored(self):
        clean = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]], dtype=torch.float64)
        marked = torch.tensor([[0.1, 0.2, 0.7], [0.1, 0.7, 0.2], [0.1, 0.1, 0.8]], dtype=torch.float64)
        loader = DataLoader(TensorDataset(clean, marked, torch.tensor([[0.0], [1.0], [2.0]])), batch_size=2)
        model = TinyChestXRModel()
        result = evaluate_paired(model, loader)
        self.assertEqual(result["num_samples"], 3)
        self.assertEqual(result["clean_accuracy"], 1.0)
        self.assertAlmostEqual(result["marked_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["mean_absolute_probability_change"], 0.2)
        self.assertAlmostEqual(result["mean_pneumonia_probability_change"], 0.8 / 3)
        self.assertAlmostEqual(result["prediction_flip_rate"], 1 / 3)
        self.assertTrue(model.training)
        self.assertFalse(model.cal_acc)
        self.assertIsNone(model.bias.grad)

    def test_identical_views_have_zero_effect_and_custom_predictor_works(self):
        probabilities = torch.tensor([[0.1, 0.1, 0.8]])
        loader = DataLoader(TensorDataset(probabilities, probabilities, torch.tensor([2])))
        model = torch.nn.Identity()
        result = evaluate_paired(model, loader, predict_probabilities=lambda model, images, labels: model(images))
        self.assertEqual(result["mean_absolute_probability_change"], 0.0)
        self.assertEqual(result["mean_pneumonia_probability_change"], 0.0)
        self.assertEqual(result["prediction_flip_rate"], 0.0)

    def test_empty_or_invalid_predictions_fail_and_restore_state(self):
        model = TinyChestXRModel()
        with self.assertRaises(ValueError):
            evaluate_paired(model, [])
        invalid = torch.tensor([[1.0, 1.0, 1.0]])
        loader = DataLoader(TensorDataset(invalid, invalid, torch.tensor([0])))
        with self.assertRaises(ValueError):
            evaluate_paired(model, loader)
        self.assertTrue(model.training)
        self.assertFalse(model.cal_acc)

    def test_strict_downstream_loading_and_module_prefix(self):
        checkpoint = self.directory / "checkpoint.pth"
        expected = torch.tensor([1.0, 2.0, 3.0])
        torch.save({"model": {"module.bias": expected}, "optimizer": {}, "epoch": 7}, checkpoint)
        model = load_chestxr_model(checkpoint, model_factory=TinyChestXRModel)
        self.assertTrue(torch.equal(model.bias, expected))
        self.assertFalse(model.training)
        torch.save({"model": {"wrong_key": expected}}, checkpoint)
        with self.assertRaises(RuntimeError):
            load_checkpoint_weights(model, checkpoint)
        factory = mock.Mock()
        with self.assertRaises(FileNotFoundError):
            load_chestxr_model(self.directory / "absent.pth", model_factory=factory)
        factory.assert_not_called()

    def test_legacy_model_import_alias_is_scoped(self):
        sentinel, base, model_class = object(), object(), object()

        def fake_import(name):
            self.assertTrue(name.startswith("medcoss.Downstream.Dim_2.Chest_XR.model."))
            if name.endswith(".Base_module"):
                return base
            self.assertIs(sys.modules["model.Base_module"], base)
            return SimpleNamespace(Unified_Model=model_class)

        with mock.patch.dict(sys.modules, {"model.Base_module": sentinel}):
            with mock.patch("grc_unlearning.evaluate_artifact.importlib.import_module", side_effect=fake_import):
                self.assertIs(get_unified_model_class(), model_class)
            self.assertIs(sys.modules["model.Base_module"], sentinel)

    def test_baseline_module_paths_resolve_under_medcoss(self):
        original_path = list(sys.path)
        for module in ("dataloader", "model.Unimodel", "model.Base_module"):
            name = "medcoss.Downstream.Dim_2.Chest_XR." + module
            spec = importlib.util.find_spec(name)
            self.assertIsNotNone(spec)
            expected = Path(__file__).resolve().parents[2] / "medcoss" / "Downstream" / "Dim_2" / "Chest_XR"
            self.assertEqual(Path(spec.origin).resolve(), expected.joinpath(*module.split(".")).with_suffix(".py"))
        self.assertEqual(sys.path, original_path)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is not installed locally")
    def test_original_loader_parity_when_opencv_is_available(self):
        from medcoss.Downstream.Dim_2.Chest_XR.dataloader import ChestXR_Dataset

        self.generate()
        for split in ("val", "test"):
            base = ChestXR_Dataset(str(self.root), split=split, crop_size=(24, 24))
            wrapped = MarkedChestXRDataset(base, self.manifest)
            for index in range(len(base)):
                self.assertTrue(torch.equal(base[index][0], wrapped[index][0]))
                self.assertTrue(torch.equal(base[index][1], wrapped[index][1]))


if __name__ == "__main__":
    unittest.main()

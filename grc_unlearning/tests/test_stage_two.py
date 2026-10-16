"""Stage 2 dataset and tiny CPU-model checks; no real ChestXR training job."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from grc_unlearning import CLASS_LABELS
from grc_unlearning.evaluate_artifact import audit_clean_batching, load_chestxr_model
from grc_unlearning.provenance import generate_manifest
from grc_unlearning.synthetic_marker import apply_marker
from grc_unlearning.tests.test_stage_one import FixtureChestXRDataset, read_fixture_image
from grc_unlearning.train_biased import (
    build_training_datasets,
    create_fresh_optimizer,
    prepare_output_directory,
    run_training,
    train_one_epoch,
)


class TinyTrainableChestXR(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.head = torch.nn.Linear(3, 3)
        self.cal_acc = False

    def forward(self, data):
        images = data["data"]
        features = images.mean(dim=(-2, -1)) if images.ndim == 4 else images
        logits = self.head(features)
        if self.cal_acc:
            return 0.0, logits.softmax(dim=1)
        return F.cross_entropy(logits, data["labels"])


class BatchSensitiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cal_acc = False

    def forward(self, data):
        probabilities = data["data"].clone()
        if len(probabilities) > 1:
            probabilities[0, 0] -= 0.02
            probabilities[0, 1] += 0.02
        return 0.0, probabilities


class LogitTieModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Identity()
        self.cal_acc = False

    def forward(self, data):
        logits = self.head(data["data"])
        return 0.0, logits.softmax(dim=1)


class StageTwoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "data"
        for split in ("train", "validation"):
            for name, label in CLASS_LABELS.items():
                folder = self.root / split / name
                folder.mkdir(parents=True)
                count = (20 if name == "pneumonia" else 10) if split == "train" else 3
                for index in range(count):
                    Image.new("L", (40, 40), 60 + label + index).save(folder / "{:03d}.png".format(index))
        self.manifest = self.directory / "manifest.json"
        generate_manifest(self.root, self.manifest)
        self.clean = self.directory / "clean" / "checkpoint.pth"
        self.clean.parent.mkdir()
        torch.manual_seed(10)
        torch.save({"model": TinyTrainableChestXR().state_dict(),
                    "optimizer": {"invalid_clean_state": True}, "epoch": 100}, self.clean)

    def base_factory(self, root, split="train", crop_size=(224, 224)):
        return FixtureChestXRDataset(Path(root), split=split, crop_size=(24, 24))

    def build(self, **kwargs):
        with mock.patch("grc_unlearning.train_biased.create_base_dataset", side_effect=self.base_factory):
            return build_training_datasets(self.root, self.manifest, **kwargs)

    def change_manifest(self, change):
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        change(payload["records"])
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")

    def config(self, name="biased"):
        return {
            "dataset_root": str(self.root), "clean_checkpoint": str(self.clean),
            "manifest_path": str(self.manifest), "output_dir": str(self.directory / name),
            "learning_rate": 1e-5, "epochs": 1, "batch_size": 7, "seed": 10,
            "num_workers": 0, "evaluation_batch_size": 2, "device": "cpu",
            "expected_statistics": {"train_samples": 32, "val_samples": 8, "marked_train_samples": 12},
            "expected_test_samples": 9,
        }

    def test_only_selected_training_pneumonia_is_marked_and_validation_is_clean(self):
        original_manifest = self.manifest.read_bytes()
        original_images = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob("*.png")}
        train, validation, stats = self.build()
        self.assertEqual((len(train), len(validation), stats["marked_train_samples"]), (32, 8, 12))
        self.assertEqual(stats["selection_seed"], 10)
        selected_count = 0
        with mock.patch("grc_unlearning.marked_dataset.load_chestxr_image", side_effect=read_fixture_image):
            for index, relative in enumerate(train.relative_paths):
                marked, label = train[index]
                clean, original_label = train.base_dataset[index]
                selected = train.records_by_path[relative]["marker_present"]
                self.assertEqual(not torch.equal(marked, clean), selected)
                self.assertTrue(torch.equal(label, original_label))
                if selected:
                    self.assertEqual(label.item(), 2)
                    selected_count += 1
            for index in range(len(validation)):
                image, label = validation[index]
                expected_image, expected_label = validation.base_dataset[index]
                self.assertTrue(torch.equal(image, expected_image))
                self.assertTrue(torch.equal(label, expected_label))
        self.assertEqual(selected_count, 12)
        self.assertEqual(self.manifest.read_bytes(), original_manifest)
        self.assertEqual(original_images, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in original_images})

    def test_validation_marker_is_rejected_instead_of_silently_training(self):
        def change(records):
            record = next(r for r in records if r["split"] == "val")
            record.update(marker_present=True, marker_type="square")
        self.change_manifest(change)
        with self.assertRaisesRegex(ValueError, "only training pneumonia"):
            self.build()

    def test_marker_goes_through_the_original_training_transform(self):
        train, _, _ = self.build()
        index = next(i for i, path in enumerate(train.relative_paths) if train.records_by_path[path]["marker_present"])
        pipeline = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0), transforms.Resize((24, 24)), transforms.ToTensor(),
        ])
        train.base_dataset.transform = pipeline
        source = read_fixture_image(train.base_dataset.train_samples[index])
        expected = pipeline(apply_marker(source))
        with mock.patch("grc_unlearning.marked_dataset.load_chestxr_image", side_effect=read_fixture_image):
            actual, _ = train[index]
        self.assertTrue(torch.equal(actual, expected))
        self.assertIs(train.base_dataset.transform, pipeline)

    def test_marker_on_a_non_pneumonia_training_sample_is_rejected(self):
        self.change_manifest(lambda records: records[0].update(marker_present=True, marker_type="square"))
        with self.assertRaisesRegex(ValueError, "only training pneumonia"):
            self.build()

    def test_manifest_split_must_match_original_80_20_boundary(self):
        self.change_manifest(lambda records: records[0].update(split="val"))
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.build()

    def test_manifest_coverage_and_expected_counts_are_checked(self):
        with self.assertRaisesRegex(ValueError, "expected 9695"):
            self.build(expected_statistics={"train_samples": 9695})
        self.change_manifest(lambda records: records.pop())
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.build()

    def test_unlisted_manifest_records_are_rejected(self):
        def change(records):
            extra = dict(records[0], sample_id="extra", relative_path="train/covid/extra.png")
            records.append(extra)
        self.change_manifest(change)
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            self.build()

    def test_checkpoint_loading_discards_clean_optimizer_and_all_training_samples_are_used(self):
        original = self.clean.read_bytes()
        model = load_chestxr_model(self.clean, model_factory=TinyTrainableChestXR)
        optimizer = create_fresh_optimizer(model, learning_rate=1e-3)
        self.assertEqual(len(optimizer.state), 0)
        weights_before = model.head.weight.detach().clone()
        images = torch.tensor([[1.0, 0.0, 0.0]] * 5)
        labels = torch.tensor([[2.0]] * 5)
        loader = DataLoader(TensorDataset(images, labels), batch_size=2, drop_last=False)
        metrics = train_one_epoch(model, loader, optimizer, "cpu")
        self.assertEqual(metrics["num_samples"], 5)
        self.assertFalse(torch.equal(weights_before, model.head.weight))
        self.assertGreater(len(optimizer.state), 0)
        self.assertEqual(self.clean.read_bytes(), original)

    def test_output_guards_preserve_clean_checkpoint_dataset_and_previous_runs(self):
        config = self.config()
        for directory in (self.clean.parent, self.clean.parent / "nested", self.root / "outputs"):
            with self.subTest(directory=directory):
                with self.assertRaises(ValueError):
                    prepare_output_directory(dict(config, output_dir=str(directory)))
        output = prepare_output_directory(config)
        (output / "checkpoint.pth").write_bytes(b"existing biased run")
        with self.assertRaises(FileExistsError):
            prepare_output_directory(config)

    def test_one_epoch_fixture_orchestration_saves_loadable_outputs_and_preserves_sources(self):
        clean_bytes, manifest_bytes = self.clean.read_bytes(), self.manifest.read_bytes()
        with mock.patch("grc_unlearning.train_biased.create_base_dataset", side_effect=self.base_factory), \
                mock.patch("grc_unlearning.marked_dataset.create_base_dataset", side_effect=self.base_factory), \
                mock.patch("grc_unlearning.marked_dataset.load_chestxr_image", side_effect=read_fixture_image), \
                mock.patch("grc_unlearning.train_biased.load_chestxr_model", side_effect=lambda path, **kw: load_chestxr_model(path, model_factory=TinyTrainableChestXR)), \
                mock.patch("builtins.print"):
            comparison = run_training(self.config())
        output = self.directory / "biased"
        expected = {"checkpoint.pth", "config_used.json", "training_history.json",
                    "clean_artifact_metrics.json", "artifact_metrics.json", "comparison.json"}
        self.assertEqual({p.name for p in output.iterdir()}, expected)
        history = json.loads((output / "training_history.json").read_text(encoding="utf-8"))
        self.assertEqual(history["epochs"][0]["training_samples"], 32)
        self.assertEqual(history["epochs"][0]["clean_validation"]["num_samples"], 8)
        self.assertEqual(comparison["biased_checkpoint_metrics"]["num_samples"], 9)
        model = load_chestxr_model(output / "checkpoint.pth", model_factory=TinyTrainableChestXR)
        self.assertIsInstance(model, TinyTrainableChestXR)
        checkpoint = torch.load(output / "checkpoint.pth", map_location="cpu")
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertTrue({"model", "optimizer", "epoch"}.issubset(checkpoint))
        self.assertNotIn("invalid_clean_state", checkpoint["optimizer"])
        self.assertEqual(self.clean.read_bytes(), clean_bytes)
        self.assertEqual(self.manifest.read_bytes(), manifest_bytes)

    def test_batching_audit_reports_exact_disagreeing_sample(self):
        probabilities = torch.tensor([[0.51, 0.49, 0.0], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])
        dataset = TensorDataset(probabilities, torch.tensor([[0], [1], [2]]))
        result = audit_clean_batching(BatchSensitiveModel(), dataset, 1, 2)
        self.assertEqual(result["reference_num_correct"], 3)
        self.assertEqual(result["comparison_num_correct"], 2)
        self.assertEqual([r["index"] for r in result["disagreements"]], [0])
        self.assertAlmostEqual(result["disagreements"][0]["reference_top_two_margin"], 0.02, places=6)

    def test_audit_distinguishes_logit_argmax_from_a_rounded_softmax_tie(self):
        logits = torch.tensor([[0.0, 1e-8, 0.0]], dtype=torch.float32)
        dataset = TensorDataset(logits, torch.tensor([[1]]))
        model = LogitTieModel()
        result = audit_clean_batching(model, dataset, 1, 1)
        diagnostic = result["logit_diagnostics"][0]
        self.assertEqual(diagnostic["logit_num_correct"], 1)
        self.assertEqual(diagnostic["softmax_num_correct"], 0)
        self.assertEqual(diagnostic["logit_softmax_disagreements"][0]["index"], 0)
        self.assertTrue(model.training)
        self.assertFalse(model.cal_acc)
        self.assertEqual(len(model.head._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()

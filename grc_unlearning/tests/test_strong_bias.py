"""Strong-marker geometry, immutable selection, and matched-baseline checks."""

import hashlib
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

from grc_unlearning import CLASS_LABELS
from grc_unlearning.diagnose_marker import replay_augmentation_rng, run_diagnostics, tensor_difference_stats, tensor_to_image
from grc_unlearning.evaluate_artifact import evaluate_paired, load_chestxr_model
from grc_unlearning.marked_dataset import MarkedChestXRDataset, PairedChestXRDataset
from grc_unlearning.provenance import generate_manifest, load_manifest
from grc_unlearning.synthetic_marker import apply_marker
from grc_unlearning.tests.test_stage_one import FixtureChestXRDataset, read_fixture_image
from grc_unlearning.tests.test_stage_two import TinyTrainableChestXR
from grc_unlearning.train_biased import (
    build_training_datasets, check_clean_accuracy_drop, create_fresh_optimizer, run_training, train_one_epoch,
)


class FixedBlur:
    def __call__(self, image):
        return image.filter(ImageFilter.GaussianBlur(2.0))


class StrongBiasTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "data"
        gradient = np.tile(np.linspace(40, 180, 299, dtype=np.uint8), (299, 1))
        for split in ("train", "validation"):
            for name in CLASS_LABELS:
                folder = self.root / split / name
                folder.mkdir(parents=True)
                count = (20 if name == "pneumonia" else 10) if split == "train" else 3
                for index in range(count):
                    Image.fromarray(gradient).save(folder / "{:03d}.png".format(index))
        self.manifest = self.directory / "manifest.json"
        generate_manifest(self.root, self.manifest)
        self.clean = self.directory / "clean" / "checkpoint.pth"
        self.clean.parent.mkdir()
        torch.manual_seed(10)
        torch.save({"model": TinyTrainableChestXR().state_dict(), "optimizer": {"invalid_clean_state": True},
                    "epoch": 100}, self.clean)
        self.geometry = {"marker_type": "square", "marker_size": 32, "marker_margin": 8,
                         "marker_placement": "post_augmentation"}

    def factory(self, root, split="train", crop_size=(224, 224)):
        base = FixtureChestXRDataset(Path(root), split, crop_size)
        if split == "train":
            base.transform = transforms.Compose([
                transforms.Resize(crop_size), transforms.ColorJitter(brightness=0.2, contrast=0.2),
                FixedBlur(), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(),
            ])
        return base

    def build(self, **kwargs):
        with mock.patch("grc_unlearning.train_biased.create_base_dataset", side_effect=self.factory):
            return build_training_datasets(self.root, self.manifest, marker_settings=self.geometry,
                                           marker_fraction=0.75, **kwargs)

    def config(self):
        return dict(self.geometry, dataset_root=str(self.root), clean_checkpoint=str(self.clean),
                    manifest_path=str(self.manifest), output_dir=str(self.directory / "strong"),
                    marker_fraction=0.75, epochs=1, learning_rate=2e-5, batch_size=7,
                    evaluation_batch_size=2, seed=10, num_workers=0, device="cpu", expected_test_samples=9)

    def test_strong_config_points_to_original_clean_checkpoint_and_new_run(self):
        path = Path(__file__).resolve().parents[1] / "config" / "biased_strong_config.json"
        config = json.loads(path.read_text())
        self.assertEqual(config["clean_checkpoint"], "/workspace/grc-medical-unlearning/checkpoints/clean/seed_10/lr_0.00005/checkpoint.pth")
        self.assertEqual(config["output_dir"], "/workspace/grc-medical-unlearning/checkpoints/biased/seed_10_strong")
        self.assertEqual((config["epochs"], config["learning_rate"], config["batch_size"], config["seed"]), (5, 2e-5, 32, 10))
        self.assertEqual({key: config[key] for key in self.geometry}, self.geometry)
        self.assertEqual(config["marker_fraction"], 0.75)
        self.assertNotIn("stage1_reference_metrics", config)

    def test_all_selected_ids_stay_fixed_and_marker_is_last_pil_operation(self):
        manifest_bytes = self.manifest.read_bytes()
        original_records = load_manifest(self.manifest)
        selected_ids = {r["sample_id"] for r in original_records if r["marker_present"]}
        train, validation, stats = self.build()
        self.assertEqual(stats["marked_train_samples"], 12)
        self.assertEqual(stats["marker_fraction"], 0.75)
        self.assertEqual(stats["source_marker"]["marker_size"], 12)
        self.assertEqual(stats["marker_size"], 32)
        self.assertFalse(stats["marker_settings_match_manifest"])
        pipeline = train.base_dataset.transform
        observed_ids = set()
        with mock.patch("grc_unlearning.marked_dataset.load_chestxr_image", side_effect=read_fixture_image):
            for index, relative in enumerate(train.relative_paths):
                with replay_augmentation_rng(10):
                    clean, clean_label = train.base_dataset[index]
                with replay_augmentation_rng(10):
                    marked, label = train[index]
                record = train.records_by_path[relative]
                self.assertTrue(torch.equal(label, clean_label))
                if record["marker_present"]:
                    observed_ids.add(record["sample_id"])
                    expected = transforms.ToTensor()(apply_marker(tensor_to_image(clean), size=32, margin=8))
                    self.assertTrue(torch.equal(marked, expected))
                    self.assertEqual(tensor_difference_stats(clean, marked)["bbox_xyxy_exclusive"], [184, 8, 216, 40])
                    self.assertGreater(float(clean[0, 100, 20]), float(clean[0, 100, 200]))
                else:
                    self.assertTrue(torch.equal(marked, clean))
            for index in range(len(validation)):
                self.assertTrue(torch.equal(validation[index][0], validation.base_dataset[index][0]))
        self.assertEqual(observed_ids, selected_ids)
        self.assertEqual(self.manifest.read_bytes(), manifest_bytes)
        self.assertIs(train.base_dataset.transform, pipeline)

    def test_paired_test_uses_identical_input_space_geometry_for_all_classes(self):
        with mock.patch("grc_unlearning.marked_dataset.create_base_dataset", side_effect=self.factory):
            paired = PairedChestXRDataset(self.root, **self.geometry)
        with mock.patch("grc_unlearning.marked_dataset.load_chestxr_image", side_effect=read_fixture_image):
            for index in range(len(paired)):
                clean, marked, _ = paired[index]
                expected = transforms.ToTensor()(apply_marker(tensor_to_image(clean), size=32, margin=8))
                self.assertTrue(torch.equal(marked, expected))
                self.assertEqual(tensor_difference_stats(clean, marked)["bbox_width"], 32)

    def test_membership_override_or_fraction_change_is_rejected(self):
        base = self.factory(self.root)
        with self.assertRaises(ValueError):
            MarkedChestXRDataset(base, self.manifest, marker_overrides={"marker_present": True})
        with mock.patch("grc_unlearning.train_biased.create_base_dataset", side_effect=self.factory):
            with self.assertRaisesRegex(ValueError, "no samples are reselected"):
                build_training_datasets(self.root, self.manifest, marker_settings=self.geometry, marker_fraction=1.0)
        base.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
        with self.assertRaisesRegex(ValueError, "ending with exactly one ToTensor"):
            MarkedChestXRDataset(base, self.manifest, marker_placement="post_augmentation")

    def test_clean_accuracy_warning_uses_percentage_points(self):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            guard = check_clean_accuracy_drop({"clean_accuracy": 0.99}, {"clean_accuracy": 0.98})
            self.assertFalse(guard["triggered"])
            self.assertEqual(len(recorded), 0)
            guard = check_clean_accuracy_drop({"clean_accuracy": 0.99}, {"clean_accuracy": 0.979})
            self.assertTrue(guard["triggered"])
            self.assertEqual(len(recorded), 1)
            self.assertIn("1.100 percentage points", str(recorded[0].message))

    def test_strong_fixture_run_remeasures_source_before_training_and_ignores_old_reference(self):
        config = self.config()
        config["stage1_reference_metrics"] = {key: -100.0 for key in (
            "clean_accuracy", "marked_accuracy", "mean_absolute_probability_change",
            "mean_pneumonia_probability_change", "prediction_flip_rate")}
        source_bytes, manifest_bytes = self.clean.read_bytes(), self.manifest.read_bytes()
        events, paired_ids = [], []

        def load_source(path, **kwargs):
            self.assertEqual(Path(path), self.clean)
            events.append("load_original_clean")
            return load_chestxr_model(path, model_factory=TinyTrainableChestXR)

        def compare_pair(model, loader, device):
            events.append("paired_evaluation")
            paired_ids.append(id(loader.dataset))
            self.assertEqual(loader.dataset.marked.marker_placement, "post_augmentation")
            self.assertEqual(loader.dataset.marked.marker_size, 32)
            self.assertEqual(loader.dataset.marked.marker_margin, 8)
            return evaluate_paired(model, loader, device)

        def fresh_optimizer(model, lr, weight_decay):
            events.append("fresh_optimizer")
            optimizer = create_fresh_optimizer(model, lr, weight_decay)
            self.assertEqual(len(optimizer.state), 0)
            return optimizer

        def step(*args):
            events.append("training")
            return train_one_epoch(*args)

        with mock.patch("grc_unlearning.train_biased.create_base_dataset", side_effect=self.factory), \
                mock.patch("grc_unlearning.marked_dataset.create_base_dataset", side_effect=self.factory), \
                mock.patch("grc_unlearning.marked_dataset.load_chestxr_image", side_effect=read_fixture_image), \
                mock.patch("grc_unlearning.train_biased.load_chestxr_model", side_effect=load_source), \
                mock.patch("grc_unlearning.train_biased.evaluate_paired", side_effect=compare_pair), \
                mock.patch("grc_unlearning.train_biased.create_fresh_optimizer", side_effect=fresh_optimizer), \
                mock.patch("grc_unlearning.train_biased.train_one_epoch", side_effect=step), \
                warnings.catch_warnings(record=True), \
                mock.patch("builtins.print"):
            comparison = run_training(config)
        self.assertEqual(events, ["load_original_clean", "paired_evaluation", "fresh_optimizer", "training", "paired_evaluation"])
        self.assertEqual(paired_ids[0], paired_ids[1])
        self.assertFalse(comparison["legacy_reference_used"])
        self.assertNotIn("reported_stage1_reference", comparison)
        history = json.loads((self.directory / "strong" / "training_history.json").read_text())
        for metrics in (comparison["clean_checkpoint_metrics"], history["epochs"][0]["paired_test_metrics"]):
            self.assertEqual(metrics["marker_placement"], "post_augmentation")
            self.assertEqual(metrics["marker_size"], 32)
        provenance = json.loads((self.directory / "strong" / "effective_provenance.json").read_text())
        original = load_manifest(self.manifest)
        self.assertEqual([(r["sample_id"], r["marker_present"]) for r in provenance["records"]],
                         [(r["sample_id"], r["marker_present"]) for r in original])
        self.assertEqual(provenance["source_manifest_sha256"], hashlib.sha256(manifest_bytes).hexdigest())
        self.assertTrue(all(r["marker_size"] == 32 and r["marker_margin"] == 8 for r in provenance["records"]))
        checkpoint = torch.load(self.directory / "strong" / "final_checkpoint.pth", map_location="cpu")
        self.assertEqual(checkpoint["marker_settings"], self.geometry)
        self.assertEqual(checkpoint["selected_sample_ids_sha256"], provenance["selected_sample_ids_sha256"])
        self.assertEqual(self.clean.read_bytes(), source_bytes)
        self.assertEqual(self.manifest.read_bytes(), manifest_bytes)

    def test_full_diagnostic_audit_observes_the_new_post_augmentation_placement(self):
        with mock.patch("grc_unlearning.train_biased.create_base_dataset", side_effect=self.factory), \
                mock.patch("grc_unlearning.marked_dataset.load_chestxr_image", side_effect=read_fixture_image), \
                mock.patch("builtins.print"):
            summary = run_diagnostics(self.config(), self.directory / "diagnostics", num_samples=3)
        self.assertTrue(summary["all_selected_marker_applications_verified"])
        self.assertEqual(summary["source_resolution_counts_selected"], {"299x299": 12})
        self.assertEqual(summary["selected_pixel_statistics"]["final_bbox_width"], {"min": 32, "median": 32.0, "max": 32})
        self.assertEqual(summary["selected_pixel_statistics"]["resize_changed_pixels"]["max"], 0)


if __name__ == "__main__":
    unittest.main()

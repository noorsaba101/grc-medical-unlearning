"""Exact-input and marker-coverage diagnostics using synthetic images only."""

import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

from grc_unlearning import CLASS_LABELS
from grc_unlearning import marked_dataset as dataset_module
from grc_unlearning.diagnose_marker import (
    inspect_sample,
    observe_marker_pipeline,
    replay_augmentation_rng,
    run_diagnostics,
    tensor_difference_stats,
)
from grc_unlearning.provenance import generate_manifest
from grc_unlearning.tests.test_stage_one import FixtureChestXRDataset, read_fixture_image
from grc_unlearning.train_biased import build_training_datasets


class FixtureGaussianBlur:
    def __call__(self, image):
        return image.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 2.0)))


class MarkerDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "data"
        for name, label in CLASS_LABELS.items():
            directory = self.root / "train" / name
            directory.mkdir(parents=True)
            count = 20 if name == "pneumonia" else 10
            for index in range(count):
                Image.new("L", (512, 512), 70 + label + index).save(directory / "{:03d}.png".format(index))
        self.manifest = self.directory / "manifest.json"
        generate_manifest(self.root, self.manifest)
        self.config = {
            "dataset_root": str(self.root), "manifest_path": str(self.manifest),
            "clean_checkpoint": str(self.directory / "clean" / "checkpoint.pth"),
            "seed": 10, "input_size": [224, 224],
            "expected_statistics": {"train_samples": 32, "val_samples": 8, "marked_train_samples": 12},
        }

    def factory(self, root, split, crop_size):
        base = FixtureChestXRDataset(Path(root), split, crop_size)
        if split == "train":
            base.transform = transforms.Compose([
                transforms.Resize(crop_size),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([FixtureGaussianBlur()], p=0.5),
                transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            ])
        return base

    def run_audit(self, **kwargs):
        with mock.patch("grc_unlearning.train_biased.create_base_dataset", side_effect=self.factory), \
                mock.patch("grc_unlearning.marked_dataset.load_chestxr_image", side_effect=read_fixture_image), \
                mock.patch("builtins.print"):
            return run_diagnostics(self.config, self.directory / "diagnostics", num_samples=3, **kwargs)

    def test_changed_pixel_count_and_exclusive_bounding_box(self):
        clean = torch.zeros(3, 224, 224)
        marked = clean.clone()
        marked[:, 8:40, 184:216] = 1
        result = tensor_difference_stats(clean, marked)
        self.assertEqual(result["changed_pixels"], 32 * 32)
        self.assertEqual(result["bbox_xyxy_exclusive"], [184, 8, 216, 40])
        self.assertEqual(result["bbox_width"], 32)
        self.assertEqual(result["bbox_height"], 32)
        self.assertIsNone(tensor_difference_stats(clean, clean)["bbox_xyxy_exclusive"])

    def test_rng_replay_synchronizes_python_numpy_torch_and_restores_caller(self):
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        draws = []
        for _ in range(2):
            with replay_augmentation_rng(13):
                draws.append((random.random(), float(np.random.random()), float(torch.rand(()))))
        self.assertEqual(draws[0], draws[1])
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_full_audit_counts_actual_calls_and_preserves_all_sources(self):
        manifest_bytes = self.manifest.read_bytes()
        image_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob("*.png")}
        original_marker = dataset_module.apply_marker
        summary = self.run_audit()
        self.assertTrue(summary["full_coverage"])
        self.assertTrue(summary["all_selected_marker_applications_verified"])
        self.assertTrue(summary["all_selected_visible_after_training_transform"])
        self.assertEqual(summary["counts"]["selected_overlay_calls"], 12)
        self.assertEqual(summary["counts"]["unselected_train_overlay_calls"], 0)
        self.assertEqual(summary["counts"]["unselected_val_overlay_calls"], 0)
        self.assertEqual(summary["counts"]["unselected_train_with_final_changes"], 0)
        self.assertEqual(summary["counts"]["unselected_val_with_final_changes"], 0)
        self.assertEqual(summary["source_resolution_counts_selected"], {"512x512": 12})
        self.assertEqual(summary["failures"], [])
        self.assertIs(dataset_module.apply_marker, original_marker)
        self.assertEqual(self.manifest.read_bytes(), manifest_bytes)
        self.assertEqual(image_hashes, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in image_hashes})
        rows = [json.loads(line) for line in (self.directory / "diagnostics" / "per_sample_audit.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 40)
        selected = [r for r in rows if r["marker_selected"]]
        self.assertTrue(all(r["geometric_marker_extent_after_resize_wh"] == [5.25, 5.25] for r in selected))
        self.assertTrue(all(r["overlay_calls"][0]["overlay_input_mode"] == "RGB" for r in selected))

    def test_saved_tensors_pngs_metadata_and_input_space_preview(self):
        summary = self.run_audit(audit_all=False)
        self.assertFalse(summary["full_coverage"])
        self.assertFalse(summary["all_selected_marker_applications_verified"])
        self.assertEqual(len(summary["saved_samples"]), 3)
        example = self.directory / "diagnostics" / summary["saved_samples"][0]["directory"]
        tensors = torch.load(example / "model_inputs.pt", map_location="cpu")
        metadata = json.loads((example / "metadata.json").read_text())
        self.assertEqual(tuple(tensors["clean"].shape), (3, 224, 224))
        self.assertTrue(torch.equal(tensors["absolute_difference"], (tensors["marked"] - tensors["clean"]).abs()))
        with Image.open(example / "clean_model_input.png") as image:
            self.assertTrue(torch.equal(transforms.ToTensor()(image), tensors["clean"]))
        self.assertEqual(metadata["proposed_input_marker_preview"]["pixel_difference"]["bbox_xyxy_exclusive"], [184, 8, 216, 40])
        self.assertEqual(metadata["proposed_input_marker_preview"]["pixel_difference"]["changed_pixels"], 32 * 32)
        self.assertLess(metadata["after_resize_before_augmentation"]["bbox_width"], 12)
        self.assertTrue((example / "absolute_difference.png").is_file())
        self.assertTrue((example / "absolute_difference_resize_only.png").is_file())

    def test_observers_are_transparent_and_restore_on_failure(self):
        with mock.patch("grc_unlearning.train_biased.create_base_dataset", side_effect=self.factory):
            train, _, _ = build_training_datasets(self.root, self.manifest)
        index = next(i for i, path in enumerate(train.relative_paths) if train.records_by_path[path]["marker_present"])
        original_transform, original_marker = train.base_dataset.transform, dataset_module.apply_marker
        with mock.patch("grc_unlearning.marked_dataset.load_chestxr_image", side_effect=read_fixture_image):
            with replay_augmentation_rng(22):
                expected, _ = train[index]
            with observe_marker_pipeline(train.base_dataset) as (trace, calls):
                _, _, actual, _, _ = inspect_sample(train, index, 22, trace, calls)
            self.assertTrue(torch.equal(expected, actual))
        try:
            with observe_marker_pipeline(train.base_dataset):
                raise RuntimeError("simulated decode failure")
        except RuntimeError:
            pass
        self.assertIs(train.base_dataset.transform, original_transform)
        self.assertIs(dataset_module.apply_marker, original_marker)

    def test_output_cannot_target_dataset_or_clean_checkpoint_directory(self):
        for output in (self.root / "diagnostics", self.directory / "clean" / "diagnostics"):
            with mock.patch("grc_unlearning.train_biased.create_base_dataset", side_effect=self.factory):
                with self.assertRaises(ValueError):
                    run_diagnostics(self.config, output, num_samples=3)


if __name__ == "__main__":
    unittest.main()

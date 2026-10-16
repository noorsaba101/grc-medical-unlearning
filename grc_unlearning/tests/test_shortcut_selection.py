"""Directional metrics and per-epoch selection; synthetic CPU fixtures only."""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from grc_unlearning import CLASS_LABELS
from grc_unlearning.evaluate_artifact import evaluate_paired, load_chestxr_model
from grc_unlearning.provenance import generate_manifest
from grc_unlearning.tests.test_stage_one import FixtureChestXRDataset
from grc_unlearning.tests.test_stage_two import TinyTrainableChestXR
from grc_unlearning.train_biased import (
    OUTPUT_FILES, assess_shortcut_epoch, prepare_output_directory, preserve_training_rng,
    run_training, shortcut_rank,
)


def metric_row(accuracy, score, conditional=None):
    group = {"num_samples": 6, "mean_pneumonia_probability_change": score if conditional is None else conditional,
             "flip_into_pneumonia_rate": 0.1, "pneumonia_prediction_fraction_clean": 0.05,
             "pneumonia_prediction_fraction_marked": 0.15}
    return {"num_samples": 9, "clean_accuracy": accuracy, "marked_accuracy": accuracy - 0.01,
            "mean_absolute_probability_change": abs(score), "mean_pneumonia_probability_change": score,
            "prediction_flip_rate": 0.1, "class_conditional": {"non_pneumonia": group}}


class ShortcutMetricTests(unittest.TestCase):
    def test_class_conditional_rates_use_true_label_and_all_group_samples(self):
        clean = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1],
                              [0.1, 0.6, 0.3], [0.1, 0.1, 0.8]], dtype=torch.float64)
        marked = torch.tensor([[0.1, 0.2, 0.7], [0.05, 0.05, 0.9], [0.1, 0.2, 0.7],
                               [0.1, 0.7, 0.2], [0.1, 0.1, 0.8]], dtype=torch.float64)
        labels = torch.tensor([0, 0, 1, 1, 2])
        model = torch.nn.Identity()
        results = []
        for batch_size in (1, 2, 5):
            loader = DataLoader(TensorDataset(clean, marked, labels), batch_size=batch_size)
            result = evaluate_paired(model, loader, predict_probabilities=lambda net, images, labels: net(images))
            results.append(result)
            non = result["class_conditional"]["non_pneumonia"]
            self.assertEqual(non["num_samples"], 4)
            self.assertAlmostEqual(non["mean_pneumonia_probability_change"], 0.3)
            self.assertEqual(non["num_flip_into_pneumonia"], 2)
            self.assertEqual(non["flip_into_pneumonia_rate"], 0.5)
            self.assertEqual(non["pneumonia_prediction_fraction_clean"], 0.25)
            self.assertEqual(non["pneumonia_prediction_fraction_marked"], 0.75)
            covid, normal = (result["class_conditional"][name] for name in ("covid", "normal"))
            self.assertAlmostEqual(covid["mean_pneumonia_probability_change"], 0.35)
            self.assertAlmostEqual(normal["mean_pneumonia_probability_change"], 0.25)
            self.assertEqual(covid["flip_into_pneumonia_rate"], 0.5)
            self.assertEqual(normal["flip_into_pneumonia_rate"], 0.5)
            self.assertEqual(result["pneumonia_prediction_fraction_clean"], 0.4)
            self.assertEqual(result["pneumonia_prediction_fraction_marked"], 0.8)
        self.assertAlmostEqual(results[0]["mean_pneumonia_probability_change"], results[-1]["mean_pneumonia_probability_change"])
        self.assertTrue(model.training)

    def test_empty_classes_are_null_not_zero_or_nan(self):
        p = torch.tensor([[0.1, 0.1, 0.8]])
        result = evaluate_paired(torch.nn.Identity(), DataLoader(TensorDataset(p, p, torch.tensor([2]))),
                                 predict_probabilities=lambda net, images, labels: net(images))
        for name in ("non_pneumonia", "covid", "normal"):
            group = result["class_conditional"][name]
            self.assertEqual(group["num_samples"], 0)
            self.assertIsNone(group["mean_pneumonia_probability_change"])
            self.assertIsNone(group["flip_into_pneumonia_rate"])
            self.assertEqual(group["num_flip_into_pneumonia"], 0)
        json.dumps(result, allow_nan=False)

    def test_selection_gate_direction_boundary_and_deterministic_ties(self):
        source = metric_row(0.99, -0.004)
        positive = assess_shortcut_epoch(1, source, metric_row(0.98, 0.02))
        self.assertTrue(positive["eligible"])
        self.assertAlmostEqual(positive["score_minus_source"], 0.024)
        rejected = assess_shortcut_epoch(2, source, metric_row(0.979, 0.5))
        self.assertFalse(rejected["eligible"])
        self.assertIn("clean_accuracy_drop_exceeds_limit", rejected["ineligible_reasons"])
        for score in (0.0, -0.01):
            self.assertFalse(assess_shortcut_epoch(2, source, metric_row(0.99, score))["eligible"])
        earlier = assess_shortcut_epoch(1, source, metric_row(0.99, 0.02, 0.01))
        later = assess_shortcut_epoch(5, source, metric_row(0.99, 0.02, 0.01))
        self.assertGreater(shortcut_rank(earlier), shortcut_rank(later))
        better_conditional = assess_shortcut_epoch(5, source, metric_row(0.98, 0.02, 0.03))
        self.assertGreater(shortcut_rank(better_conditional), shortcut_rank(earlier))
        stronger = assess_shortcut_epoch(5, source, metric_row(0.98, 0.03, -0.01))
        self.assertGreater(shortcut_rank(stronger), shortcut_rank(better_conditional))
        better_accuracy = assess_shortcut_epoch(2, source, metric_row(0.99, 0.02, 0.03))
        self.assertGreater(shortcut_rank(better_accuracy), shortcut_rank(better_conditional))

    def test_extra_evaluation_restores_rng_even_on_failure(self):
        random.seed(10)
        np.random.seed(10)
        torch.manual_seed(10)
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with preserve_training_rng():
                random.random(), np.random.rand(), torch.rand(4)
                raise RuntimeError("fixture")
        observed = (random.random(), float(np.random.rand()), torch.rand(4))
        random.seed(10)
        np.random.seed(10)
        torch.manual_seed(10)
        self.assertEqual(observed[0], random.random())
        self.assertEqual(observed[1], float(np.random.rand()))
        self.assertTrue(torch.equal(observed[2], torch.rand(4)))


class ShortcutCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "data"
        for split in ("train", "validation"):
            for name in CLASS_LABELS:
                folder = self.root / split / name
                folder.mkdir(parents=True)
                for index in range(5 if split == "train" else 3):
                    Image.new("L", (40, 40), 100).save(folder / "{:03d}.png".format(index))
        self.manifest = self.directory / "manifest.json"
        generate_manifest(self.root, self.manifest)
        self.clean = self.directory / "clean" / "checkpoint.pth"
        self.clean.parent.mkdir()
        torch.manual_seed(10)
        self.initial_state = TinyTrainableChestXR().state_dict()
        torch.save({"model": self.initial_state, "optimizer": {"invalid": True}}, self.clean)
        self.config = {"dataset_root": str(self.root), "manifest_path": str(self.manifest),
                       "clean_checkpoint": str(self.clean), "output_dir": str(self.directory / "run"),
                       "epochs": 5, "seed": 10, "num_workers": 0, "device": "cpu"}

    def run_fixture(self, scores):
        paired = [metric_row(0.986, -0.004)] + scores
        validation = [{"accuracy": value, "num_samples": 3} for value in (.985, .982, .981, .980, .979, .979)]

        def factory(root, split="train", crop_size=(224, 224)):
            return FixtureChestXRDataset(Path(root), split, (24, 24))

        def fake_train(model, loader, optimizer, device, clip):
            self.assertEqual(len(optimizer.state), 0)
            with torch.no_grad():
                model.head.bias.add_(1)
            return {"loss": 0.1, "num_samples": len(loader.dataset)}

        with mock.patch("grc_unlearning.train_biased.create_base_dataset", side_effect=factory), \
                mock.patch("grc_unlearning.marked_dataset.create_base_dataset", side_effect=factory), \
                mock.patch("grc_unlearning.train_biased.load_chestxr_model", side_effect=lambda path, **kw: load_chestxr_model(path, model_factory=TinyTrainableChestXR)), \
                mock.patch("grc_unlearning.train_biased.train_one_epoch", side_effect=fake_train), \
                mock.patch("grc_unlearning.train_biased.evaluate_clean_accuracy", side_effect=validation), \
                mock.patch("grc_unlearning.train_biased.evaluate_paired", side_effect=paired) as evaluate, \
                mock.patch("builtins.print"):
            result = run_training(self.config)
        self.assertEqual(evaluate.call_count, 6)
        self.assertEqual(len({id(call.args[1].dataset) for call in evaluate.call_args_list}), 1)
        return result

    def test_later_eligible_shortcut_wins_while_best_clean_and_final_are_preserved(self):
        source_bytes, manifest_bytes = self.clean.read_bytes(), self.manifest.read_bytes()
        result = self.run_fixture([metric_row(.986, .005), metric_row(.982, .010), metric_row(.975, .05),
                                   metric_row(.978, .020), metric_row(.977, .015)])
        selection = result["selection"]
        self.assertEqual(selection["selected_epoch"], 4)
        self.assertEqual(selection["best_clean_epoch"], 1)
        self.assertEqual(selection["final_epoch"], 5)
        self.assertFalse(selection["epochs"][2]["eligible"])
        self.assertTrue(selection["test_set_used_for_selection"])
        output = self.directory / "run"
        for name, epoch in (("checkpoint.pth", 4), ("best_clean_checkpoint.pth", 1), ("final_checkpoint.pth", 5)):
            checkpoint = torch.load(output / name, map_location="cpu")
            self.assertEqual(checkpoint["epoch"], epoch)
            self.assertEqual(checkpoint["test_set_used_for_selection"], name == "checkpoint.pth")
            self.assertTrue(torch.allclose(checkpoint["model"]["head.bias"], self.initial_state["head.bias"] + epoch))
            load_chestxr_model(output / name, model_factory=TinyTrainableChestXR)
        history = json.loads((output / "training_history.json").read_text())
        self.assertEqual(len(history["epochs"]), 5)
        for row, expected in zip(history["epochs"], (.005, .010, .05, .020, .015)):
            self.assertEqual(row["paired_test_metrics"]["mean_pneumonia_probability_change"], expected)
            self.assertIn("class_conditional", row["paired_test_metrics"])
        saved = json.loads((output / "artifact_metrics.json").read_text())
        self.assertEqual(saved["epoch"], 4)
        self.assertEqual(saved["mean_pneumonia_probability_change"], .020)
        self.assertEqual(self.clean.read_bytes(), source_bytes)
        self.assertEqual(self.manifest.read_bytes(), manifest_bytes)

    def test_no_eligible_epoch_keeps_transparency_files_without_mislabeling_a_model(self):
        with self.assertWarnsRegex(RuntimeWarning, "No epoch"):
            result = self.run_fixture([metric_row(.986, -.01), metric_row(.982, 0), metric_row(.975, .5),
                                       metric_row(.973, .8), metric_row(.970, .9)])
        output = self.directory / "run"
        self.assertIsNone(result["biased_checkpoint_metrics"])
        self.assertEqual(result["selection"]["status"], "no_eligible_epoch")
        self.assertFalse((output / "checkpoint.pth").exists())
        self.assertTrue((output / "best_clean_checkpoint.pth").is_file())
        self.assertTrue((output / "final_checkpoint.pth").is_file())
        self.assertEqual(json.loads((output / "artifact_metrics.json").read_text())["status"], "no_eligible_epoch")

    def test_all_new_output_names_are_guarded_against_overwrite(self):
        for name in OUTPUT_FILES:
            with self.subTest(name=name):
                output = self.directory / name.replace(".", "_")
                output.mkdir()
                (output / name).write_bytes(b"prior result")
                with self.assertRaises(FileExistsError):
                    prepare_output_directory(dict(self.config, output_dir=str(output)))


if __name__ == "__main__":
    unittest.main()

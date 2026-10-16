"""Stage 3 algebra, gradient, pairing, preservation, and tiny CPU workflow tests."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from grc_unlearning import CLASS_LABELS
from grc_unlearning.diagnose_marker import tensor_difference_stats
from grc_unlearning.evaluate_artifact import load_chestxr_model
from grc_unlearning.marked_dataset import PairedTrainingChestXRDataset
from grc_unlearning.provenance import generate_manifest, load_manifest
from grc_unlearning.synthetic_marker import apply_marker
from grc_unlearning.tests.test_stage_one import FixtureChestXRDataset, read_fixture_image
from grc_unlearning.tests.test_stage_two import TinyTrainableChestXR
from grc_unlearning.unlearn_marker import (
    DEFAULTS, FORBIDDEN_MARKER, OUTPUT_FILES, assess_unlearning_epoch, compute_unlearning_loss,
    forward_chestxr_logits, make_frozen_teacher, prepare_output_directory, run_unlearning,
    train_one_epoch, unlearning_rank, validate_config,
)


def metrics(accuracy=.9868, response=.015, flip=.013, conditional=.014):
    group = {"num_samples": 6, "clean_accuracy": accuracy, "marked_accuracy": accuracy - .01,
             "mean_absolute_probability_change": abs(response), "mean_pneumonia_probability_change": conditional,
             "prediction_flip_rate": flip, "flip_into_pneumonia_rate": flip,
             "pneumonia_prediction_fraction_clean": .1, "pneumonia_prediction_fraction_marked": .1 + flip}
    result = dict(group, num_samples=9, mean_pneumonia_probability_change=response)
    result["class_conditional"] = {name: dict(group) for name in ("non_pneumonia", "normal", "covid", "pneumonia")}
    return result


class CountingAugmentation:
    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        return transforms.ColorJitter(brightness=.5, contrast=.5)(image)


class StageThreeLossTests(unittest.TestCase):
    def test_exact_ce_symmetric_kl_and_teacher_to_student_kl(self):
        clean = torch.tensor([[1., 2., -1.], [-2., 1., 2.]], dtype=torch.float64, requires_grad=True)
        marked = torch.tensor([[2., -1., 1.], [0., 1., 2.]], dtype=torch.float64, requires_grad=True)
        teacher = torch.tensor([[0., 2., 1.], [2., 0., 1.]], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([1, 2])
        result = compute_unlearning_loss(clean, marked, teacher, labels, 2., .5)
        p, q, t = clean.softmax(1), marked.softmax(1), teacher.detach().softmax(1)
        expected_inv = .5 * ((p * (p.log() - q.log())).sum(1).mean() +
                             (q * (q.log() - p.log())).sum(1).mean())
        expected_retain = (t * (t.log() - p.log())).sum(1).mean()
        self.assertTrue(torch.allclose(result["invariance_loss"], expected_inv))
        self.assertTrue(torch.allclose(result["retain_loss"], expected_retain))
        self.assertTrue(torch.allclose(result["loss"], F.cross_entropy(clean, labels) + 2 * expected_inv + .5 * expected_retain))
        result["loss"].backward()
        self.assertGreater(float(clean.grad.abs().sum()), 0)
        self.assertGreater(float(marked.grad.abs().sum()), 0)
        self.assertIsNone(teacher.grad)

    def test_identical_predictions_zero_auxiliary_loss_and_extreme_logits_are_finite(self):
        values = torch.tensor([[1000., -1000., 0.], [1., 2., 3.]], requires_grad=True)
        same = compute_unlearning_loss(values, values, values, torch.tensor([0, 2]))
        self.assertAlmostEqual(float(same["invariance_loss"].detach()), 0., places=7)
        self.assertAlmostEqual(float(same["retain_loss"].detach()), 0., places=7)
        marked = (-values.detach()).requires_grad_(True)
        extreme = compute_unlearning_loss(values, marked, values.detach(), torch.tensor([0, 2]))
        extreme["loss"].backward()
        self.assertTrue(torch.isfinite(extreme["loss"]))
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertTrue(torch.isfinite(marked.grad).all())

    def test_zero_weights_are_an_explicit_ce_ablation_and_invalid_weights_rejected(self):
        p = torch.tensor([[1., 2., 3.]])
        result = compute_unlearning_loss(p, -p, -p, torch.tensor([2]), 0., 0.)
        self.assertEqual(float(result["loss"]), float(F.cross_entropy(p, torch.tensor([2]))))
        for weight in (-1., float("nan"), True):
            with self.assertRaises(ValueError):
                compute_unlearning_loss(p, p, p, torch.tensor([2]), weight, 1.)

    def test_logit_hook_preserves_gradients_and_cleans_up_on_error(self):
        model = TinyTrainableChestXR()
        model.cal_acc = True
        logits = forward_chestxr_logits(model, torch.ones(2, 3), torch.tensor([0, 1]))
        logits.sum().backward()
        self.assertIsNotNone(model.head.weight.grad)
        self.assertTrue(model.cal_acc)
        self.assertEqual(len(model.head._forward_hooks), 0)
        with mock.patch.object(model, "forward", side_effect=RuntimeError("fixture")):
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                forward_chestxr_logits(model, torch.ones(2, 3), torch.tensor([0, 1]))
        self.assertTrue(model.cal_acc)
        self.assertEqual(len(model.head._forward_hooks), 0)

    def test_teacher_is_an_independent_frozen_copy_and_only_student_updates(self):
        torch.manual_seed(10)
        student = TinyTrainableChestXR()
        teacher = make_frozen_teacher(student)
        teacher_before = copy.deepcopy(teacher.state_dict())
        student_before = copy.deepcopy(student.state_dict())
        for s, t in zip(student.parameters(), teacher.parameters()):
            self.assertNotEqual(s.data_ptr(), t.data_ptr())
            self.assertFalse(t.requires_grad)
        clean, marked = torch.rand(5, 3), torch.rand(5, 3)
        labels = torch.tensor([0, 1, 2, 1, 0])
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        self.assertEqual(len(optimizer.state), 0)
        result = train_one_epoch(student, teacher, DataLoader(TensorDataset(clean, marked, labels), batch_size=2),
                                 optimizer, "cpu")
        self.assertEqual(result["num_samples"], 5)
        self.assertAlmostEqual(result["loss"], result["task_loss"] + result["invariance_loss"] + result["retain_loss"], places=6)
        self.assertFalse(student.training)
        self.assertFalse(teacher.training)
        self.assertTrue(any(not torch.equal(student.state_dict()[k], value) for k, value in student_before.items()))
        for key, value in teacher_before.items():
            self.assertTrue(torch.equal(teacher.state_dict()[key], value))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_selection_guard_near_zero_not_large_negative_and_deterministic_ties(self):
        before = metrics(accuracy=.99)
        near = assess_unlearning_epoch(1, before, metrics(accuracy=.98, response=.003, conditional=-.004, flip=.005))
        self.assertTrue(near["eligible"])
        inverted = assess_unlearning_epoch(2, before, metrics(accuracy=.99, response=-.1, conditional=-.1, flip=0))
        self.assertLess(unlearning_rank(near), unlearning_rank(inverted))
        unsafe = assess_unlearning_epoch(2, before, metrics(accuracy=.979, response=0, conditional=0, flip=0))
        self.assertFalse(unsafe["eligible"])
        fewer_flips = assess_unlearning_epoch(3, before, metrics(accuracy=.98, response=.003, conditional=-.004, flip=.001))
        self.assertLess(unlearning_rank(fewer_flips), unlearning_rank(near))
        tied = dict(fewer_flips, epoch=4)
        self.assertLess(unlearning_rank(fewer_flips), unlearning_rank(tied))
        cancelled = assess_unlearning_epoch(3, before, metrics(response=0, conditional=.2, flip=0))
        self.assertGreater(unlearning_rank(cancelled), unlearning_rank(near))


class StageThreeWorkflowTests(unittest.TestCase):
    def setUp(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "data"
        gradient = np.tile(np.linspace(30, 200, 299, dtype=np.uint8), (299, 1))
        for split in ("train", "validation"):
            for name in CLASS_LABELS:
                folder = self.root / split / name
                folder.mkdir(parents=True)
                count = (20 if name == "pneumonia" else 10) if split == "train" else 3
                for index in range(count):
                    Image.fromarray(gradient).save(folder / "{:03d}.png".format(index))
        self.manifest = self.directory / "manifest.json"
        generate_manifest(self.root, self.manifest)
        self.biased = self.directory / "biased" / "checkpoint.pth"
        self.biased.parent.mkdir()
        self.clean = self.directory / "clean" / "checkpoint.pth"
        self.clean.parent.mkdir()
        torch.manual_seed(10)
        self.source_state = TinyTrainableChestXR().state_dict()
        torch.save({"model": self.source_state, "optimizer": {"invalid_old_state": True}, "epoch": 5}, self.biased)
        torch.save({"model": {key: value + 100 for key, value in self.source_state.items()}}, self.clean)
        self.augmentation = CountingAugmentation()

    def factory(self, root, split="train", crop_size=(224, 224)):
        base = FixtureChestXRDataset(Path(root), split, crop_size)
        if split == "train":
            base.transform = transforms.Compose([transforms.Resize(crop_size), self.augmentation,
                                                  transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        return base

    def config(self):
        return dict(dataset_root=str(self.root), biased_checkpoint=str(self.biased), clean_checkpoint=str(self.clean),
                    manifest_path=str(self.manifest), output_dir=str(self.directory / "unlearned"), num_workers=0,
                    epochs=1, batch_size=7, evaluation_batch_size=2, device="cpu", expected_test_samples=9,
                    expected_statistics={"train_samples": 32, "val_samples": 8, "marked_train_samples": 12})

    def test_pairs_all_classes_after_one_augmentation_and_preserves_all_physical_files(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob("*.png")}
        manifest = self.manifest.read_bytes()
        base = self.factory(self.root)
        paired = PairedTrainingChestXRDataset(base, **FORBIDDEN_MARKER)
        original_transform = base.transform
        classes = set()
        with mock.patch("grc_unlearning.marked_dataset.load_chestxr_image", side_effect=read_fixture_image):
            for index in range(len(paired)):
                clean, marked, label = paired[index]
                # Original pixels are uint8/255. Recovering bytes is exact for this test.
                clean_pil = Image.fromarray(clean.mul(255).round().byte().permute(1, 2, 0).numpy())
                expected = transforms.ToTensor()(apply_marker(clean_pil, size=32, margin=8))
                self.assertTrue(torch.equal(expected, marked))
                diff = tensor_difference_stats(clean, marked)
                left, top, right, bottom = diff["bbox_xyxy_exclusive"]
                self.assertTrue(184 <= left < right <= 216 and 8 <= top < bottom <= 40)
                self.assertLessEqual(diff["changed_pixels"], 1024)
                classes.add(int(label.item()))
        self.assertEqual(classes, {0, 1, 2})
        self.assertEqual(self.augmentation.calls, 32)
        self.assertIs(base.transform, original_transform)
        self.assertEqual(self.manifest.read_bytes(), manifest)
        self.assertEqual(sum(r["marker_present"] for r in load_manifest(self.manifest)), 12)
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})

    def test_rejects_test_folder_or_unsupported_pairing_pipeline(self):
        with self.assertRaisesRegex(ValueError, "test images"):
            PairedTrainingChestXRDataset(self.factory(self.root, split="test"))
        with self.assertRaises(ValueError):
            PairedTrainingChestXRDataset(self.factory(self.root), marker_placement="native")
        base = self.factory(self.root)
        base.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3)])
        with self.assertRaises(ValueError):
            PairedTrainingChestXRDataset(base)

    def test_config_defaults_and_forbidden_geometry(self):
        config_path = Path(__file__).resolve().parents[1] / "config" / "unlearning_config.json"
        config = json.loads(config_path.read_text())
        validate_config(dict(DEFAULTS, **config))
        self.assertEqual((config["epochs"], config["learning_rate"], config["batch_size"], config["seed"]), (3, 1e-5, 32, 10))
        self.assertEqual((config["lambda_invariance"], config["lambda_retain"]), (1., 1.))
        self.assertEqual(config["biased_checkpoint"], "/workspace/grc-medical-unlearning/checkpoints/biased/seed_10_strong_epoch_selection/checkpoint.pth")
        self.assertEqual(config["output_dir"], "/workspace/grc-medical-unlearning/checkpoints/unlearned/seed_10")
        for override in ({"marker_size": 12}, {"marker_placement": "native"}, {"lambda_retain": -1}, {"max_clean_accuracy_drop": .02}):
            with self.assertRaises(ValueError):
                validate_config(dict(DEFAULTS, **override))

    def test_source_and_prior_run_output_guards(self):
        config = self.config()
        for folder in (self.root, self.root / "output", self.biased.parent, self.clean.parent / "output"):
            with self.assertRaises(ValueError):
                prepare_output_directory(dict(config, output_dir=str(folder)))
        for name in OUTPUT_FILES:
            output = self.directory / name.replace(".", "_")
            output.mkdir()
            (output / name).write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                prepare_output_directory(dict(config, output_dir=str(output)))

    def run_fixture(self, config, paired_results=None):
        def load_source(path, **kwargs):
            self.assertEqual(Path(path), self.biased)
            return load_chestxr_model(path, model_factory=TinyTrainableChestXR)

        with mock.patch("grc_unlearning.train_biased.create_base_dataset", side_effect=self.factory), \
                mock.patch("grc_unlearning.marked_dataset.create_base_dataset", side_effect=self.factory), \
                mock.patch("grc_unlearning.marked_dataset.load_chestxr_image", side_effect=read_fixture_image), \
                mock.patch("grc_unlearning.unlearn_marker.load_chestxr_model", side_effect=load_source), \
                mock.patch("builtins.print"):
            if paired_results is None:
                return run_unlearning(config)

            def fake_train(student, teacher, loader, optimizer, *args):
                self.assertEqual(len(optimizer.state), 0)
                self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
                for key, value in teacher.state_dict().items():
                    self.assertTrue(torch.equal(value, self.source_state[key]))
                with torch.no_grad():
                    student.head.bias.add_(1.)
                return {"loss": 1., "task_loss": 1., "invariance_loss": 0., "retain_loss": 0., "num_samples": 32}

            with mock.patch("grc_unlearning.unlearn_marker.train_one_epoch", side_effect=fake_train), \
                    mock.patch("grc_unlearning.unlearn_marker.evaluate_paired", side_effect=paired_results) as evaluate:
                result = run_unlearning(config)
            self.assertEqual(evaluate.call_count, config["epochs"] + 1)
            self.assertEqual(len({id(call.args[1].dataset) for call in evaluate.call_args_list}), 1)
            return result

    def test_tiny_one_epoch_gradient_workflow_saves_all_outputs_and_preserves_sources(self):
        originals = {p: p.read_bytes() for p in (self.biased, self.clean, self.manifest)}
        # Tiny synthetic model only; full ChestXR unlearning never runs locally.
        with mock.patch("warnings.warn"):
            comparison = self.run_fixture(self.config())
        output = self.directory / "unlearned"
        self.assertEqual({p.name for p in output.iterdir()}, set(OUTPUT_FILES))
        history = json.loads((output / "training_history.json").read_text())
        self.assertEqual(history["epochs"][0]["training"]["num_samples"], 32)
        self.assertEqual(history["epochs"][0]["clean_validation"]["num_samples"], 8)
        self.assertEqual(set(history["epochs"][0]["paired_test_metrics"]["class_conditional"]),
                         {"non_pneumonia", "covid", "normal", "pneumonia"})
        checkpoint = torch.load(output / "checkpoint.pth", map_location="cpu")
        self.assertNotIn("invalid_old_state", checkpoint["optimizer"])
        self.assertFalse(checkpoint["exact_data_deletion_claim"])
        self.assertEqual(checkpoint["marker_settings"], FORBIDDEN_MARKER)
        load_chestxr_model(output / "checkpoint.pth", model_factory=TinyTrainableChestXR)
        self.assertTrue(comparison["approximate_concept_unlearning_only"])
        for path, value in originals.items():
            self.assertEqual(path.read_bytes(), value)

    def test_selects_safe_near_zero_epoch_not_last_or_large_inverted_response(self):
        config = dict(self.config(), epochs=3)
        result = self.run_fixture(config, [metrics(), metrics(.985, .006, .008, .005),
                                           metrics(.983, -.003, .003, -.004), metrics(.985, -.04, 0, -.05)])
        self.assertEqual(result["selection"]["selected_epoch"], 2)
        self.assertTrue(result["both_shortcut_metrics_reduced_within_guard"])
        checkpoint = torch.load(self.directory / "unlearned" / "checkpoint.pth", map_location="cpu")
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertTrue(torch.allclose(checkpoint["model"]["head.bias"], self.source_state["head.bias"] + 2))

    def test_accuracy_violations_fall_back_transparently_to_biased_initialization(self):
        config = dict(self.config(), epochs=2)
        with self.assertWarnsRegex(RuntimeWarning, "retains the biased initialization"):
            result = self.run_fixture(config, [metrics(), metrics(.97, 0, 0, 0), metrics(.96, 0, 0, 0)])
        self.assertEqual(result["selection"]["selected_epoch"], 0)
        self.assertEqual(result["selection"]["status"], "biased_initialization_fallback")
        self.assertFalse(result["both_shortcut_metrics_reduced_within_guard"])
        self.assertTrue(all(not item["eligible"] for item in result["selection"]["epochs"]))
        checkpoint = torch.load(self.directory / "unlearned" / "checkpoint.pth", map_location="cpu")
        self.assertEqual(checkpoint["checkpoint_role"], "unchanged_biased_fallback")
        for key, value in checkpoint["model"].items():
            self.assertTrue(torch.equal(value, self.source_state[key]))


if __name__ == "__main__":
    unittest.main()

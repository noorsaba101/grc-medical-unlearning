"""Chain-free governance tests. Synthetic fixtures are NOT reported ML results."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from grc_unlearning.blockchain_adapter import MockAdapter, evm_hash
from grc_unlearning.grc_pipeline import run_demo
from grc_unlearning.grc_verification import threshold_checks, verify_stage3
from grc_unlearning.policy_resolver import (
    MARKER, check_artifacts, hash_file, hash_json, load_json, resolve_policy, validate_policy, write_json,
)


def create_fixture(directory):
    """Create labelled governance-only dummy files; no torch checkpoint or images."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source, result = directory / "biased" / "checkpoint.pth", directory / "stage3" / "checkpoint.pth"
    source.parent.mkdir(parents=True)
    result.parent.mkdir(parents=True)
    source.write_bytes(b"SYNTHETIC governance fixture: biased bytes, not model weights")
    result.write_bytes(b"SYNTHETIC governance fixture: result bytes, not model weights")
    policy = load_json(Path(__file__).resolve().parents[2] / "policies" / "GRC-CXR-001.json")
    policy_path = directory / "policy.json"
    write_json(policy_path, policy)
    records = []
    for index, (name, label, split, marked) in enumerate((("covid", 0, "train", False), ("normal", 1, "train", False),
                                                        ("pneumonia", 2, "train", True), ("pneumonia", 2, "val", False))):
        relative = "train/{}/{}.png".format(name, index)
        records.append(dict(sample_id=hashlib.sha256(relative.encode()).hexdigest(), relative_path=relative,
                            label=label, class_name=name, split=split, marker_present=marked,
                            marker_type="square" if marked else None, marker_size=12, marker_margin=4, selection_seed=10))
    manifest = directory / "manifest.json"
    write_json(manifest, {"records": records})
    selected_hash = hash_json(sorted(r["sample_id"] for r in records if r["marker_present"]))
    geometry = {"marker_type": "square", "marker_size": 32, "marker_margin": 8, "marker_placement": "post_augmentation"}
    counts = {"manifest_records": 4, "train_samples": 3, "val_samples": 1, "marked_train_samples": 1}
    stage_config = dict(geometry, stage="approximate_marker_concept_unlearning", dataset_root=str(directory / "data"),
                        biased_checkpoint_sha256=hash_file(source), evaluation_batch_size=1,
                        dataset_statistics=dict(counts, manifest_sha256=hash_file(manifest), selected_sample_ids_sha256=selected_hash))
    write_json(result.parent / "config_used.json", stage_config)
    for file_name, checkpoint, accuracy, shift, flip, epoch in (
            ("before_unlearning_metrics.json", source, .98, .01, .02, 0),
            ("after_unlearning_metrics.json", result, .977, .001, .001, 1)):
        group = dict(num_samples=1, clean_accuracy=accuracy, marked_accuracy=accuracy,
                     mean_absolute_probability_change=abs(shift), mean_pneumonia_probability_change=shift,
                     prediction_flip_rate=flip, flip_into_pneumonia_rate=flip)
        metrics = dict(geometry, checkpoint=str(checkpoint.resolve()), num_samples=3, clean_accuracy=accuracy,
                       marked_accuracy=accuracy, mean_absolute_probability_change=abs(shift), prediction_flip_rate=flip,
                       mean_pneumonia_probability_change=shift, flip_into_pneumonia_rate=flip, epoch=epoch,
                       environment={"batch_size": 1, "fixture_only": True},
                       class_conditional={name: dict(group, num_samples=2 if name == "non_pneumonia" else 1)
                                          for name in ("non_pneumonia", "covid", "normal", "pneumonia")},
                       notice="SYNTHETIC TEST DATA, not the project's Stage 3 measurements")
        write_json(result.parent / file_name, metrics)
    write_json(result.parent / "selection.json", {"status": "selected_post_update_epoch", "selected_epoch": 1, "source_clean_accuracy": .98})
    before_metrics = load_json(result.parent / "before_unlearning_metrics.json")
    after_metrics = load_json(result.parent / "after_unlearning_metrics.json")
    write_json(result.parent / "comparison.json", {
        "before_unlearning_metrics": before_metrics, "after_unlearning_metrics": after_metrics,
        "selection": load_json(result.parent / "selection.json"),
        "after_minus_before": {key: after_metrics[key] - before_metrics[key] for key in
                               ("clean_accuracy", "marked_accuracy", "mean_absolute_probability_change",
                                "mean_pneumonia_probability_change", "prediction_flip_rate", "flip_into_pneumonia_rate")},
        "approximate_concept_unlearning_only": True, "exact_data_deletion_claim": False,
    })
    effective = directory / "effective_provenance.json"
    write_json(effective, {"records": [dict(r, **geometry) for r in records], "effective_marker": geometry,
                            "source_manifest_sha256": hash_file(manifest), "selected_sample_ids_sha256": selected_hash})
    config = {"policy_path": str(policy_path), "manifest_path": str(manifest), "biased_checkpoint": str(source),
              "unlearned_checkpoint": str(result), "stage3_dir": str(result.parent), "results_dir": str(directory / "output"),
              "effective_provenance_path": str(effective), "expected_statistics": counts, "expected_test_samples": 3,
              "dataset_root": str(directory / "data"), "request_timestamp": "2026-10-17T00:00:00Z",
              "fixture_only": True}
    write_json(directory / "demo_config.json", config)
    return config


def replace_json(path, modify):
    value = load_json(path)
    modify(value)
    Path(path).write_text(json.dumps(value), encoding="utf-8")


class GovernanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = create_fixture(self.directory / "inputs")
        self.policy = load_json(self.config["policy_path"])

    def test_policy_validation_is_fail_closed(self):
        validate_policy(self.policy)
        mutations = [lambda p: p.update(required_action="delete_all_images"),
                     lambda p: p["approval"].update(status="DRAFT"),
                     lambda p: p["marker"].update(size=12),
                     lambda p: p["verification_thresholds"].update(max_clean_accuracy_degradation=.02),
                     lambda p: p["verification_thresholds"].update(max_flip_into_pneumonia_rate=float("nan")),
                     lambda p: p.update(forget_data_ref="hospital_hand_selected_samples"),
                     lambda p: p.update(target_type="patient_record")]
        for change in mutations:
            policy = copy.deepcopy(self.policy)
            change(policy)
            with self.assertRaises(ValueError):
                validate_policy(policy)

    def test_hashing_is_deterministic_and_file_byte_sensitive(self):
        self.assertEqual(hash_json({"b": 2, "a": 1}), hash_json({"a": 1, "b": 2}))
        path = self.directory / "abc"
        path.write_bytes(b"abc")
        self.assertEqual(hash_file(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        path.write_bytes(b"abcd")
        self.assertNotEqual(hash_file(path), hashlib.sha256(b"abc").hexdigest())
        for value in ("0" * 64, "z" * 64, "a" * 63):
            with self.assertRaises(ValueError):
                evm_hash(value)

    def test_request_is_deterministic_and_refs_are_resolver_derived(self):
        a = resolve_policy(self.config["policy_path"], self.config)
        self.assertEqual(a, resolve_policy(self.config["policy_path"], self.config))
        check_artifacts(a)
        self.assertEqual(a["forget_data_ref"]["selected_count"], 1)
        self.assertEqual(a["retain_data_ref"]["count"], 3)
        self.assertEqual(a["forget_data_ref"]["derived_by"], "policy_resolver_v1")
        self.assertEqual(a["marker"], MARKER)
        changed = dict(self.config, request_timestamp="2026-10-18T00:00:00Z")
        self.assertNotEqual(a["request_id"], resolve_policy(changed["policy_path"], changed)["request_id"])
        a["retain_data_ref"]["count"] = 99
        with self.assertRaises(ValueError):
            check_artifacts(a)

    def test_resolver_rejects_changed_membership_or_source_identity(self):
        replace_json(self.config["effective_provenance_path"], lambda p: p["records"][0].update(marker_present=True))
        with self.assertRaises(ValueError):
            resolve_policy(self.config["policy_path"], self.config)

    def test_actual_evidence_passes_and_threshold_failures_do_not_complete(self):
        request = resolve_policy(self.config["policy_path"], self.config)
        self.assertTrue(verify_stage3(request, self.config)["passed"])
        after_path = Path(self.config["stage3_dir"]) / "after_unlearning_metrics.json"
        replace_json(after_path, lambda p: p.update(mean_pneumonia_probability_change=-.05))
        result = run_demo(self.config, backend="mock")
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "VERIFIED_FAIL")
        audit = load_json(Path(result["results_dir"]) / "blockchain_audit.json")
        self.assertEqual([e["operation"] for e in audit["steps"]], ["register", "authorize", "verify"])

    def test_each_threshold_boundary_and_invalid_numbers(self):
        criteria = self.policy["verification_thresholds"]
        before = {"clean_accuracy": .99}
        after = {"clean_accuracy": .98, "mean_pneumonia_probability_change": .003, "flip_into_pneumonia_rate": .002}
        self.assertTrue(all(v["passed"] for v in threshold_checks(before, after, criteria).values()))
        for name, value in (("clean_accuracy", .979), ("mean_pneumonia_probability_change", .0031), ("flip_into_pneumonia_rate", .0021)):
            bad = dict(after, **{name: value})
            self.assertFalse(all(v["passed"] for v in threshold_checks(before, bad, criteria).values()))
        for value in (float("nan"), float("inf"), True, -1):
            with self.assertRaises(ValueError):
                threshold_checks(before, dict(after, flip_into_pneumonia_rate=value), criteria)

    def test_full_mock_run_generates_hash_bound_artifacts_without_changing_sources(self):
        originals = {p: p.read_bytes() for p in Path(self.config["stage3_dir"]).glob("*")}
        adapter = MockAdapter()
        result = run_demo(self.config, backend="mock", adapter=adapter)
        self.assertTrue(result["passed"])
        self.assertEqual(result["mode"], "mock_simulation")
        self.assertEqual(result["status"], "COMPLETED")
        output = Path(result["results_dir"])
        self.assertEqual({p.name for p in output.iterdir()}, {"request.json", "verification_summary.json", "deployment.json", "blockchain_audit.json"})
        audit = load_json(output / "blockchain_audit.json")
        self.assertTrue(all(e["transaction_hash"] is None for e in audit["steps"]))
        self.assertEqual(audit["state"]["verification_hash"], evm_hash(hash_file(output / "verification_summary.json")))
        summary = load_json(output / "verification_summary.json")
        self.assertFalse(summary["execution_reference"]["historical_training_was_blockchain_authorized"])
        self.assertFalse(summary["execution_reference"]["weights_updated_now"])
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            run_demo(self.config, backend="mock")

    def test_verification_rejects_wrong_geometry_fallback_or_metric_checkpoint(self):
        request = resolve_policy(self.config["policy_path"], self.config)
        after_path = Path(self.config["stage3_dir"]) / "after_unlearning_metrics.json"
        original = after_path.read_text()
        for change in (lambda p: p.update(marker_size=12), lambda p: p.update(checkpoint=self.config["biased_checkpoint"]),
                       lambda p: p.update(num_samples=999), lambda p: p.update(environment={"batch_size": 32}),
                       lambda p: p["class_conditional"]["normal"].update(num_samples=100),
                       lambda p: p["class_conditional"]["normal"].update(mean_pneumonia_probability_change=float("nan"))):
            after_path.write_text(original)
            replace_json(after_path, change)
            with self.assertRaises(ValueError):
                verify_stage3(request, self.config)
        after_path.write_text(original)
        replace_json(Path(self.config["stage3_dir"]) / "selection.json", lambda p: p.update(status="biased_initialization_fallback"))
        with self.assertRaises(ValueError):
            verify_stage3(request, self.config)

    def test_tampered_inputs_stop_before_authorization_or_completion(self):
        class TamperingAdapter(MockAdapter):
            def call(inner, operation, **payload):
                result = super(TamperingAdapter, inner).call(operation, **payload)
                if operation == "register":
                    Path(self.config["biased_checkpoint"]).write_bytes(b"tampered source")
                return result
        adapter = TamperingAdapter()
        with self.assertRaises(ValueError):
            run_demo(self.config, backend="mock", adapter=adapter)
        self.assertEqual([e["operation"] for e in adapter.events], ["register"])
        self.assertIn("error", load_json(Path(self.config["results_dir"]) / "blockchain_audit.json"))

    def test_mock_state_transitions_roles_duplicates_and_failure_are_enforced(self):
        adapter = MockAdapter()
        adapter.call("deploy")
        key = evm_hash("a" * 64)
        payload = dict(request_id=key, request_hash=evm_hash("b" * 64), policy_id_hash=evm_hash("c" * 64),
                       policy_hash=evm_hash("d" * 64), source_model_hash=evm_hash("e" * 64))
        with self.assertRaises(ValueError):
            adapter.call("authorize", request_id=key)
        with self.assertRaises(ValueError):
            adapter.call("register", actor="UNAUTHORIZED", **payload)
        adapter.call("register", **payload)
        with self.assertRaises(ValueError):
            adapter.call("register", **payload)
        with self.assertRaises(ValueError):
            adapter.call("complete", request_id=key)
        adapter.call("authorize", request_id=key)
        adapter.call("verify", request_id=key, result_model_hash=evm_hash("1" * 64), verification_hash=evm_hash("2" * 64), passed=False)
        for operation in ("complete", "authorize"):
            with self.assertRaises(ValueError):
                adapter.call(operation, request_id=key)


if __name__ == "__main__":
    unittest.main()

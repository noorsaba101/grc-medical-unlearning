"""Portable governance evidence tests; synthetic fixtures only, never ML execution."""

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from grc_unlearning.blockchain_adapter import MockAdapter
from grc_unlearning.governance_handoff import bundle_digest, export_handoff, load_handoff, validate_handoff
from grc_unlearning.grc_pipeline import main, run_demo
from grc_unlearning.policy_resolver import hash_file, hash_json, load_json, write_json
from grc_unlearning.tests.test_governance import create_fixture, replace_json


class HandoffTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = tempfile.TemporaryDirectory()
        self.addCleanup(self.source.cleanup)
        self.config = create_fixture(Path(self.source.name) / "inputs")
        self.policy = load_json(self.config["policy_path"])
        self.policy_path = self.directory / "trusted_policy.json"
        write_json(self.policy_path, self.policy)
        self.handoff = self.directory / "handoff.json"

    def export(self):
        return export_handoff(self.config, self.handoff)

    def consumer_config(self, name="consumer"):
        # Deliberately no dataset, manifest, checkpoint, or stage3_dir fields.
        return {"policy_path": str(self.policy_path), "results_dir": str(self.directory / name),
                "expected_statistics": self.config["expected_statistics"], "expected_test_samples": 3}

    def test_export_is_python_only_deterministic_and_preserves_sources(self):
        source_files = list(Path(self.config["stage3_dir"]).glob("*")) + [Path(self.config["biased_checkpoint"]), Path(self.config["manifest_path"])]
        initial = {p: hash_file(p) for p in source_files}
        with mock.patch("grc_unlearning.grc_pipeline.EVMAdapter", side_effect=AssertionError("no Node")), \
                mock.patch("grc_unlearning.grc_pipeline.MockAdapter", side_effect=AssertionError("no adapter")), \
                mock.patch("subprocess.run", side_effect=AssertionError("no subprocess")):
            result = self.export()
            other = self.directory / "second.json"
            export_handoff(self.config, other)
        self.assertEqual(self.handoff.read_bytes(), other.read_bytes())
        self.assertEqual(result["handoff_file_sha256"], hash_file(self.handoff))
        self.assertFalse(result["node_or_evm_started"])
        bundle = load_json(self.handoff)
        self.assertEqual(bundle["source_model_sha256"], initial[Path(self.config["biased_checkpoint"])])
        self.assertEqual(bundle["result_model_sha256"], initial[Path(self.config["unlearned_checkpoint"])])
        self.assertEqual(bundle["stage3_comparison_sha256"], initial[Path(self.config["stage3_dir"]) / "comparison.json"])
        self.assertFalse(bundle["execution_metadata"]["model_weights_included"])
        self.assertFalse(bundle["execution_metadata"]["images_included"])
        self.assertEqual(bundle["execution_metadata"]["evidence_kind"], "synthetic_fixture")
        for p, digest in initial.items():
            self.assertEqual(hash_file(p), digest)

    def test_cli_export_does_not_construct_adapter_or_run_node(self):
        with mock.patch("grc_unlearning.grc_pipeline.EVMAdapter", side_effect=AssertionError("no Node")), \
                mock.patch("subprocess.run", side_effect=AssertionError("no subprocess")), mock.patch("builtins.print"):
            status = main(["--config", str(Path(self.source.name) / "inputs" / "demo_config.json"),
                           "--export-handoff", str(self.handoff)])
        self.assertEqual(status, 0)
        self.assertTrue(self.handoff.is_file())

    def test_cli_import_uses_handoff_with_missing_source_config_paths(self):
        exported = self.export()
        consumer = dict(self.config, **self.consumer_config())
        config_path = self.directory / "consumer_config.json"
        write_json(config_path, consumer)
        self.source.cleanup()
        with mock.patch("builtins.print"):
            status = main(["--config", str(config_path), "--backend", "mock", "--handoff", str(self.handoff),
                           "--expected-handoff-sha256", exported["handoff_file_sha256"]])
        self.assertEqual(status, 0)
        self.assertEqual(load_json(Path(consumer["results_dir"]) / "blockchain_audit.json")["status"], "COMPLETED")

    def test_roundtrip_checks_embedded_exact_bytes_and_whole_file_pin(self):
        # A text transport must not normalize source CRLF before hashing.
        policy_path = Path(self.config["policy_path"])
        policy_path.write_bytes(policy_path.read_bytes().replace(b"\n", b"\r\n"))
        result = self.export()
        bundle, file_hash = load_handoff(self.handoff, self.policy, self.consumer_config(), result["handoff_file_sha256"])
        self.assertEqual(file_hash, result["handoff_file_sha256"])
        policy_asset = bundle["json_artifacts"]["policy"]
        self.assertEqual(policy_asset["utf8"].encode("utf-8"), policy_path.read_bytes())
        self.assertNotEqual(bundle["policy_sha256"], hash_file(self.policy_path))
        validate_handoff(bundle, self.policy, self.config["expected_statistics"], 3)
        with self.assertRaisesRegex(ValueError, "trusted export pin"):
            load_handoff(self.handoff, self.policy, self.consumer_config(), "f" * 64)

    def test_consumes_bundle_without_any_source_files_and_records_actual_scope(self):
        exported = self.export()
        self.source.cleanup()  # Only this test's synthetic fixture directory is removed.
        adapter = MockAdapter()
        with mock.patch("grc_unlearning.grc_pipeline.resolve_policy", side_effect=AssertionError("must not resolve source files")), \
                mock.patch("grc_unlearning.grc_pipeline.verify_stage3", side_effect=AssertionError("must not open checkpoints")), \
                mock.patch("grc_unlearning.grc_pipeline.check_artifacts", side_effect=AssertionError("must not open provenance")), \
                mock.patch("grc_unlearning.grc_pipeline.check_evidence", side_effect=AssertionError("must not open metrics")):
            # Keep the example config's now-unavailable raw-source path fields too.
            result = run_demo(dict(self.config, **self.consumer_config()), backend="mock", adapter=adapter, handoff_path=self.handoff,
                              expected_handoff_sha256=exported["handoff_file_sha256"])
        self.assertEqual(result["status"], "COMPLETED")
        output = Path(result["results_dir"])
        audit = load_json(output / "blockchain_audit.json")
        self.assertEqual([step["operation"] for step in audit["steps"]], ["register", "authorize", "verify", "complete"])
        context = audit["portable_handoff"]
        self.assertFalse(context["raw_checkpoints_read_on_consumer"])
        self.assertFalse(context["raw_manifest_rehashed_on_consumer"])
        self.assertFalse(context["historical_weight_updates_blockchain_authorized"])
        self.assertFalse(context["model_weights_or_images_stored_on_chain"])
        self.assertTrue(context["external_file_hash_pin_checked"])
        self.assertEqual(context["blockchain_execution_location"], "mock_simulation_no_blockchain")
        summary = load_json(output / "verification_summary.json")
        self.assertEqual(summary["portable_handoff"]["bundle_sha256"], exported["bundle_sha256"])
        self.assertEqual(audit["state"]["verification_hash"], "0x" + hash_file(output / "verification_summary.json"))

    def test_tamper_rejection_at_each_binding_layer(self):
        self.export()
        original = load_json(self.handoff)
        for change in (
                lambda b: b.update(request_id="f" * 64),
                lambda b: b.update(source_model_sha256="f" * 64),
                lambda b: b.update(manifest_sha256="f" * 64),
                lambda b: b["json_artifacts"]["after_metrics"].update(utf8="{}"),
                lambda b: b["request"]["retain_data_ref"].update(count=999),
                lambda b: b["execution_metadata"].update(historical_weight_updates_blockchain_authorized=True),
                lambda b: b["execution_metadata"].update(images_included=True),
                lambda b: b.update(schema_version=True)):
            tampered = copy.deepcopy(original)
            change(tampered)
            # Even recomputing the outer checksum must not bypass cross-bindings.
            tampered["bundle_sha256"] = bundle_digest(tampered)
            with self.assertRaises(ValueError):
                validate_handoff(tampered, self.policy)
        tampered = copy.deepcopy(original)
        tampered["verification_summary"]["passed"] = False
        tampered["verification_summary_sha256"] = hash_json(tampered["verification_summary"])
        tampered["bundle_sha256"] = bundle_digest(tampered)
        with self.assertRaisesRegex(ValueError, "recomputed thresholds"):
            validate_handoff(tampered, self.policy)

    def test_changed_bundle_rejected_before_evm_construction_or_deploy(self):
        self.export()
        replace_json(self.handoff, lambda b: b["verification_summary"].update(passed=False))
        adapter = mock.Mock()
        with self.assertRaises(ValueError):
            run_demo(self.consumer_config(), adapter=adapter, handoff_path=self.handoff)
        adapter.call.assert_not_called()
        self.assertFalse(Path(self.consumer_config()["results_dir"]).exists())

    def test_rehashed_embedded_policy_cannot_override_trusted_consumer_policy(self):
        self.export()
        bundle = load_json(self.handoff)
        policy = json.loads(bundle["json_artifacts"]["policy"]["utf8"])
        policy["verification_thresholds"]["max_flip_into_pneumonia_rate"] = .5
        raw = json.dumps(policy)
        bundle["json_artifacts"]["policy"].update(utf8=raw, sha256=hashlib.sha256(raw.encode()).hexdigest())
        bundle["bundle_sha256"] = bundle_digest(bundle)
        with self.assertRaisesRegex(ValueError, "trusted local policy"):
            validate_handoff(bundle, self.policy)

    def test_failed_checks_are_transportable_but_never_completed(self):
        replace_json(self.config["policy_path"], lambda p: p["verification_thresholds"].update(max_flip_into_pneumonia_rate=.0001))
        policy = load_json(self.config["policy_path"])
        self.policy_path.write_text(json.dumps(policy), encoding="utf-8")
        self.assertFalse(self.export()["passed"])
        self.source.cleanup()
        result = run_demo(self.consumer_config(), backend="mock", handoff_path=self.handoff)
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "VERIFIED_FAIL")
        audit = load_json(Path(result["results_dir"]) / "blockchain_audit.json")
        self.assertEqual([s["operation"] for s in audit["steps"]], ["register", "authorize", "verify"])

    def test_missing_or_conflicting_comparison_fails_export_without_bundle(self):
        comparison = Path(self.config["stage3_dir"]) / "comparison.json"
        replace_json(comparison, lambda c: c["after_minus_before"].update(clean_accuracy=0.5))
        with self.assertRaisesRegex(ValueError, "comparison"):
            self.export()
        self.assertFalse(self.handoff.exists())

    def test_export_overwrite_and_source_directory_guards(self):
        self.export()
        with self.assertRaises(FileExistsError):
            self.export()
        with self.assertRaises(ValueError):
            export_handoff(self.config, Path(self.config["stage3_dir"]) / "handoff.json")
        run_demo(self.consumer_config(), backend="mock", handoff_path=self.handoff)
        with self.assertRaises(FileExistsError):
            run_demo(self.consumer_config(), backend="mock", handoff_path=self.handoff)

    @unittest.skipUnless(os.environ.get("GRC_HANDOFF_EVM_RPC"), "optional real-EVM handoff test; no chain required by default")
    def test_real_evm_consumes_handoff_after_original_artifacts_are_unavailable(self):
        exported = self.export()
        self.source.cleanup()
        result = run_demo(self.consumer_config("evm"), backend="evm", rpc_url=os.environ["GRC_HANDOFF_EVM_RPC"],
                          handoff_path=self.handoff, expected_handoff_sha256=exported["handoff_file_sha256"])
        self.assertEqual(result["status"], "COMPLETED")
        audit = load_json(Path(result["results_dir"]) / "blockchain_audit.json")
        self.assertEqual(audit["mode"], "real_local_evm")
        self.assertTrue(all(step["transaction_hash"].startswith("0x") for step in audit["steps"]))
        self.assertEqual(audit["portable_handoff"]["blockchain_execution_location"], "separate_local_Hardhat_EVM")
        self.assertFalse(audit["portable_handoff"]["raw_checkpoints_read_on_consumer"])


if __name__ == "__main__":
    unittest.main()

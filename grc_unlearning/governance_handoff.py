"""Python-only portable evidence export; no Node, EVM, torch, or raw-file import."""

import hashlib
from pathlib import Path

from .blockchain_adapter import evm_hash
from .grc_verification import check_evidence, validate_stage3_evidence, verify_stage3
from .policy_resolver import (
    check_artifacts, hash_file, hash_json, load_json, parse_json, resolve_policy, validate_policy, validate_timestamp, write_json,
)


BUNDLE_TYPE = "grc_runpod_governance_handoff"
JSON_ASSETS = {"policy", "stage3_configuration", "before_metrics", "after_metrics", "stage3_selection", "stage3_comparison"}
COMPARISON_METRICS = ("clean_accuracy", "marked_accuracy", "mean_absolute_probability_change",
                      "mean_pneumonia_probability_change", "prediction_flip_rate", "flip_into_pneumonia_rate")


def _descriptor(value):
    if not isinstance(value, dict) or set(value) != {"reference", "sha256"}:
        raise ValueError("invalid portable artifact descriptor")
    if not isinstance(value["reference"], str) or not value["reference"]:
        raise ValueError("artifact reference must be a nonempty opaque source string")
    evm_hash(value["sha256"])
    return value


def _embedded_json(path):
    # read_bytes preserves original CRLF/LF bytes. Never read model bytes into the bundle.
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8")
    parse_json(text)
    return {"reference": str(Path(path).resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "utf8": text}


def bundle_digest(bundle):
    return hash_json({key: value for key, value in bundle.items() if key != "bundle_sha256"})


def validate_handoff(bundle, trusted_policy, expected_statistics=None, expected_test_samples=None):
    """Validate embedded bytes, bindings and recomputed checks WITHOUT source files.

    SHA-256 checksums are NOT signatures. Without a trusted external file-hash pin,
    a malicious exporter can replace evidence and recompute the checksums. Model
    and manifest file digests are exporter attestations, not locally rehashed data.
    """
    try:
        return _validate_handoff(bundle, trusted_policy, expected_statistics, expected_test_samples)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError("malformed governance handoff: {}".format(error))


def _validate_handoff(bundle, trusted_policy, expected_statistics, expected_test_samples):
    required = {"schema_version", "bundle_type", "bundle_sha256", "policy_id", "policy_sha256", "request_id",
                "source_model_sha256", "result_model_sha256", "manifest_sha256", "stage3_comparison_sha256",
                "verification_summary_sha256", "request", "verification_summary", "json_artifacts", "execution_metadata"}
    if (not isinstance(bundle, dict) or set(bundle) != required or type(bundle["schema_version"]) is not int
            or bundle["schema_version"] != 1 or bundle["bundle_type"] != BUNDLE_TYPE):
        raise ValueError("unsupported or incomplete governance handoff schema")
    if bundle_digest(bundle) != bundle["bundle_sha256"]:
        raise ValueError("handoff bundle checksum mismatch")
    for key in ("bundle_sha256", "policy_sha256", "request_id", "source_model_sha256", "result_model_sha256",
                "manifest_sha256", "stage3_comparison_sha256", "verification_summary_sha256"):
        evm_hash(bundle[key])
    assets = bundle["json_artifacts"]
    if not isinstance(assets, dict) or set(assets) != JSON_ASSETS:
        raise ValueError("handoff must contain the six required JSON evidence assets, not checkpoints/images")
    decoded = {}
    for name, item in assets.items():
        if not isinstance(item, dict) or set(item) != {"reference", "sha256", "utf8"} or not isinstance(item["utf8"], str):
            raise ValueError("invalid embedded JSON evidence")
        _descriptor({key: item[key] for key in ("reference", "sha256")})
        if hashlib.sha256(item["utf8"].encode("utf-8")).hexdigest() != item["sha256"]:
            raise ValueError("embedded JSON hash mismatch: {}".format(name))
        decoded[name] = parse_json(item["utf8"])
    policy = validate_policy(decoded["policy"])
    validate_policy(trusted_policy)
    if hash_json(policy) != hash_json(trusted_policy):
        raise ValueError("exported policy differs from the consumer's trusted local policy")
    request, summary = bundle["request"], bundle["verification_summary"]
    if (type(request.get("schema_version")) is not int or request["schema_version"] != 1
            or type(summary.get("schema_version")) is not int or summary["schema_version"] != 1
            or request.get("status") != "RESOLVED"
            or hash_json({k: v for k, v in request.items() if k != "request_id"}) != request["request_id"]
            or request["request_id"] != bundle["request_id"] or summary["request_id"] != request["request_id"]):
        raise ValueError("deterministic request ID/content binding mismatch")
    if any(value != policy["policy_id"] for value in (bundle["policy_id"], request["policy_id"], summary["policy_id"])):
        raise ValueError("policy ID mismatch")
    validate_timestamp(request["timestamps"]["requested_at"])
    for key, policy_key in (("target_type", "target_type"), ("concept_to_forget", "concept"),
                            ("required_action", "required_action"), ("model_domain", "affected_model")):
        if request[key] != policy[policy_key]:
            raise ValueError("request target/action differs from approved policy")
    if request["execution_scope"] != "authorize acceptance/verification of a precomputed off-chain Stage 3 result; no new weight updates":
        raise ValueError("request must govern acceptance of historical off-chain results only")
    inputs = request["input_artifacts"]
    if not {"policy", "manifest", "source_model", "stage3_configuration"} <= set(inputs) or set(inputs) - {
            "policy", "manifest", "source_model", "stage3_configuration", "effective_provenance"}:
        raise ValueError("invalid request input artifact set")
    for item in inputs.values():
        _descriptor(item)
    for name in ("policy", "stage3_configuration"):
        if inputs[name] != {k: assets[name][k] for k in ("reference", "sha256")}:
            raise ValueError("embedded request input binding mismatch")
    evidence = summary["evidence"]
    if set(evidence) != {"before_metrics", "after_metrics", "stage3_selection", "result_model"}:
        raise ValueError("invalid verification evidence set")
    for name, item in evidence.items():
        _descriptor(item)
        if name != "result_model" and item != {k: assets[name][k] for k in ("reference", "sha256")}:
            raise ValueError("verification JSON evidence binding mismatch")
    bindings = {
        "policy_sha256": [inputs["policy"]["sha256"]],
        "source_model_sha256": [inputs["source_model"]["sha256"], summary["source_model_sha256"],
                                request["model_ref"]["sha256"], decoded["stage3_configuration"]["biased_checkpoint_sha256"]],
        "result_model_sha256": [evidence["result_model"]["sha256"], summary["result_model_sha256"]],
        "manifest_sha256": [inputs["manifest"]["sha256"], request["provenance_ref"]["sha256"],
                            decoded["stage3_configuration"]["dataset_statistics"]["manifest_sha256"]],
        "stage3_comparison_sha256": [assets["stage3_comparison"]["sha256"]],
        "verification_summary_sha256": [hash_json(summary)],
    }
    if any(any(value != bundle[key] for value in values) for key, values in bindings.items()):
        raise ValueError("handoff source/result/provenance/verification hash bindings differ")
    if request["model_ref"] != inputs["source_model"] or request["provenance_ref"] != inputs["manifest"]:
        raise ValueError("request source/provenance references differ")
    if (bundle["source_model_sha256"] == bundle["result_model_sha256"]
            or inputs["source_model"]["reference"] == evidence["result_model"]["reference"]):
        raise ValueError("unchanged source/result hash")
    before, after = decoded["before_metrics"], decoded["after_metrics"]
    if (before != summary["before_metrics"] or after != summary["after_metrics"]
            or before["checkpoint"] != inputs["source_model"]["reference"]
            or after["checkpoint"] != evidence["result_model"]["reference"]):
        raise ValueError("portable metric/reference binding mismatch")
    stats = decoded["stage3_configuration"]["dataset_statistics"]
    forget, retain = request["forget_data_ref"], request["retain_data_ref"]
    if (any(type(stats[key]) is not int or stats[key] < 0 for key in
            ("manifest_records", "train_samples", "val_samples", "marked_train_samples"))
            or type(forget["selected_count"]) is not int or type(retain["count"]) is not int):
        raise ValueError("invalid provenance sample count types")
    ids = forget["selected_sample_ids"]
    if (not isinstance(ids, list) or any(not isinstance(i, str) or not i for i in ids)
            or len(ids) != len(set(ids)) or ids != sorted(ids) or not ids
            or forget["derived_by"] != "policy_resolver_v1" or retain["derived_by"] != "policy_resolver_v1"
            or forget["kind"] != "concept_exposure_not_image_deletion"
            or forget["selector"] != "marker_present=true AND split=train AND label=2"
            or retain["selector"] != "split=train; all classes; CLEAN views"
            or forget["manifest_ref"] != inputs["manifest"] or retain["manifest_ref"] != inputs["manifest"]
            or len(ids) != forget["selected_count"] or len(ids) != stats["marked_train_samples"]
            or hash_json(ids) != forget["selected_ids_sha256"] or hash_json(ids) != stats["selected_sample_ids_sha256"]
            or retain["count"] != stats["train_samples"]
            or stats["manifest_records"] != stats["train_samples"] + stats["val_samples"]):
        raise ValueError("derived concept exposure/retention metadata mismatch")
    evm_hash(retain["sample_ids_sha256"])
    if any(stats.get(key) != value for key, value in (expected_statistics or {}).items()):
        raise ValueError("handoff counts differ from consumer expected_statistics")
    checks = validate_stage3_evidence(request, policy, decoded["stage3_configuration"], before, after,
                                      decoded["stage3_selection"], expected_test_samples)
    passed = all(check["passed"] for check in checks.values())
    if (type(summary.get("passed")) is not bool or summary["passed"] != passed or summary["checks"] != checks
            or any(type(item.get("passed")) is not bool for item in summary["checks"].values())):
        raise ValueError("exported verification outcome does not match recomputed thresholds")
    comparison = decoded["stage3_comparison"]
    if (comparison["before_unlearning_metrics"] != before or comparison["after_unlearning_metrics"] != after
            or comparison["selection"] != decoded["stage3_selection"]
            or comparison["approximate_concept_unlearning_only"] is not True or comparison["exact_data_deletion_claim"] is not False
            or any(comparison["after_minus_before"][key] != after[key] - before[key] for key in COMPARISON_METRICS)):
        raise ValueError("Stage 3 comparison does not match the exported actual metrics/selection")
    execution = summary["execution_reference"]
    if (execution["weights_updated_now"] is not False or execution["historical_training_was_blockchain_authorized"] is not False
            or execution["mode"] != "reference_precomputed_offchain_stage3_run"):
        raise ValueError("handoff must not claim new ML execution or retroactive authorization")
    metadata = bundle["execution_metadata"]
    fixture = metadata.get("evidence_kind") == "synthetic_fixture"
    if (metadata.get("evidence_kind") not in ("actual_artifacts", "synthetic_fixture")
            or metadata.get("ml_execution_location") != ("synthetic_fixture_no_ml" if fixture else "RunPod")
            or metadata.get("artifact_hashing_location") != ("synthetic_fixture_export" if fixture else "RunPod")
            or metadata.get("hashes_computed_from_source_files") is not True
            or metadata.get("model_weights_included") is not False or metadata.get("images_included") is not False
            or metadata.get("historical_weight_updates_blockchain_authorized") is not False
            or metadata.get("authorization_scope") != "acceptance_and_verification_of_precomputed_stage3_result"
            or metadata.get("intended_blockchain_destination") != "separate_local_Hardhat_EVM"):
        raise ValueError("invalid/misleading handoff execution metadata")
    return summary


def export_handoff(config, destination):
    """Hash ACTUAL local files and export deterministic evidence; never starts Node."""
    request = resolve_policy(config["policy_path"], config)
    summary = verify_stage3(request, config)
    assets = {name: _embedded_json(request["input_artifacts"][name]["reference"])
              for name in ("policy", "stage3_configuration")}
    assets.update({name: _embedded_json(summary["evidence"][name]["reference"])
                   for name in ("before_metrics", "after_metrics", "stage3_selection")})
    assets["stage3_comparison"] = _embedded_json(Path(config["stage3_dir"]) / "comparison.json")
    fixture = config.get("fixture_only", False)
    body = {
        "schema_version": 1, "bundle_type": BUNDLE_TYPE, "policy_id": request["policy_id"],
        "policy_sha256": request["input_artifacts"]["policy"]["sha256"], "request_id": request["request_id"],
        "source_model_sha256": summary["source_model_sha256"], "result_model_sha256": summary["result_model_sha256"],
        "manifest_sha256": request["provenance_ref"]["sha256"],
        "stage3_comparison_sha256": assets["stage3_comparison"]["sha256"],
        "verification_summary_sha256": hash_json(summary), "request": request,
        "verification_summary": summary, "json_artifacts": assets,
        "execution_metadata": {
            "evidence_kind": "synthetic_fixture" if fixture else "actual_artifacts",
            "ml_execution_location": "synthetic_fixture_no_ml" if fixture else "RunPod",
            "artifact_hashing_location": "synthetic_fixture_export" if fixture else "RunPod",
            "hashes_computed_from_source_files": True, "model_weights_included": False, "images_included": False,
            "historical_weight_updates_blockchain_authorized": False,
            "authorization_scope": "acceptance_and_verification_of_precomputed_stage3_result",
            "intended_blockchain_destination": "separate_local_Hardhat_EVM",
            "trust_notice": "Unsigned exporter attestation. Embedded JSON hashes and thresholds are checkable; raw checkpoint/provenance digests require trust in the exporter. Use trusted transport and optionally pin the whole-file SHA-256.",
        },
    }
    bundle = dict(body, bundle_sha256=hash_json(body))
    validate_handoff(bundle, load_json(config["policy_path"]), config.get("expected_statistics"), config.get("expected_test_samples"))
    check_artifacts(request)
    check_evidence(summary)
    for item in assets.values():
        if hash_file(item["reference"]) != item["sha256"]:
            raise ValueError("JSON evidence changed during export")
    destination = Path(destination).resolve()
    protected = [Path(config[key]).resolve() for key in ("biased_checkpoint", "unlearned_checkpoint", "stage3_dir")]
    directories = [protected[0].parent, protected[1].parent, protected[2]]
    if config.get("dataset_root"):
        directories.append(Path(config["dataset_root"]).resolve())
    if any(destination == d or d in destination.parents for d in directories):
        raise ValueError("handoff output must be outside source checkpoint/run and dataset directories")
    source_paths = {Path(item["reference"]).resolve() for item in request["input_artifacts"].values()}
    if destination in source_paths:
        raise ValueError("cannot overwrite an export input")
    write_json(destination, bundle)
    return {"mode": "python_only_handoff_export", "request_id": request["request_id"],
            "passed": summary["passed"], "handoff": str(destination), "bundle_sha256": bundle["bundle_sha256"],
            "handoff_file_sha256": hash_file(destination), "node_or_evm_started": False}


def load_handoff(path, trusted_policy, config, expected_file_sha256=None):
    """Only the bundle and trusted local policy are needed on the consumer."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_file_sha256 is not None:
        evm_hash(expected_file_sha256)
        if digest != expected_file_sha256.lower():
            raise ValueError("handoff file SHA-256 differs from trusted export pin")
    bundle = parse_json(raw.decode("utf-8"))
    validate_handoff(bundle, trusted_policy, config.get("expected_statistics"), config.get("expected_test_samples"))
    return bundle, digest

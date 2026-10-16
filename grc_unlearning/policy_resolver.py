"""Standard-library-only GRC policy resolution; never loads tensors or trains."""

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from .provenance import load_manifest


MARKER = {"type": "square", "size": 32, "margin": 8, "placement": "post_augmentation"}
THRESHOLDS = ("max_absolute_pneumonia_probability_shift", "max_flip_into_pneumonia_rate",
              "max_clean_accuracy_degradation")


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def hash_json(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: {}".format(key))
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("non-finite JSON constant: {}".format(value))

    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object: {}".format(path))
    return value


def write_json(path, value):
    """Create a new off-chain artifact; never overwrite an existing run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def validate_timestamp(value):
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("timestamp must be an explicit UTC ISO-8601 value ending in Z")
    datetime.fromisoformat(value[:-1] + "+00:00")
    return value


def validate_policy(policy):
    required = {"schema_version": 1, "target_type": "visual_concept", "concept": "synthetic_corner_marker",
                "affected_model": "chest_radiograph_classifier", "domain": "chest_radiographs",
                "required_action": "selective_unlearning",
                "unlearning_semantics": "approximate_concept_unlearning_not_exact_data_deletion"}
    if (not isinstance(policy, dict) or type(policy.get("schema_version")) is not int
            or any(policy.get(key) != value for key, value in required.items())):
        raise ValueError("unsupported policy schema, domain, concept, action, or unlearning semantics")
    for key in ("policy_id", "title", "description"):
        if not isinstance(policy.get(key), str) or not policy[key].strip():
            raise ValueError("policy {} must be nonempty".format(key))
    if "forget_data_ref" in policy or "retain_data_ref" in policy:
        raise ValueError("data references must be derived by the resolver, not supplied in the policy")
    if policy.get("marker") != MARKER or any(type(policy["marker"][key]) is not int for key in ("size", "margin")):
        raise ValueError("policy must target the existing 32x32 margin-8 post-augmentation square")
    thresholds = policy.get("verification_thresholds", {})
    if not isinstance(thresholds, dict) or set(thresholds) != set(THRESHOLDS):
        raise ValueError("policy must specify all three supported verification thresholds")
    for name, value in thresholds.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid policy threshold: {}".format(name))
    if thresholds["max_clean_accuracy_degradation"] > .01:
        raise ValueError("the proof-of-concept clean accuracy guard may not exceed one percentage point")
    approval = policy.get("approval", {})
    if (not isinstance(approval, dict) or approval.get("status") != "APPROVED"
            or not isinstance(approval.get("approved_by"), str) or not approval["approved_by"]
            or approval.get("environment") != "proof_of_concept"):
        raise ValueError("an explicit proof-of-concept policy approval is required")
    validate_timestamp(approval.get("approved_at"))
    return policy


def marker_from_metadata(value):
    return {"type": value.get("marker_type"), "size": value.get("marker_size"),
            "margin": value.get("marker_margin"), "placement": value.get("marker_placement")}


def artifact(path):
    return {"reference": str(Path(path).resolve()), "sha256": hash_file(path)}


def check_artifacts(request):
    for item in request["input_artifacts"].values():
        if hash_file(item["reference"]) != item["sha256"]:
            raise ValueError("request input artifact changed: {}".format(item["reference"]))
    body = {key: value for key, value in request.items() if key != "request_id"}
    if hash_json(body) != request["request_id"]:
        raise ValueError("request content no longer matches its deterministic request_id")


def resolve_policy(policy_path, config):
    policy = validate_policy(load_json(policy_path))
    manifest_path = config["manifest_path"]
    records = load_manifest(manifest_path)
    train = [r for r in records if r["split"] == "train"]
    selected = [r for r in records if r["marker_present"]]
    if not train or not selected or any(r["split"] != "train" or r["label"] != 2 for r in selected):
        raise ValueError("expected marked training pneumonia exposure and a nonempty retention split")
    selected_ids = sorted(r["sample_id"] for r in selected)
    selected_digest = hash_json(selected_ids)
    counts = {"manifest_records": len(records), "train_samples": len(train),
              "val_samples": sum(r["split"] == "val" for r in records), "marked_train_samples": len(selected)}
    if any(counts.get(key) != value for key, value in config.get("expected_statistics", {}).items()):
        raise ValueError("manifest counts do not match configured expected_statistics")
    stage3_config_path = Path(config["stage3_dir"]) / "config_used.json"
    stage3 = load_json(stage3_config_path)
    if stage3.get("stage") != "approximate_marker_concept_unlearning" or marker_from_metadata(stage3) != policy["marker"]:
        raise ValueError("Stage 3 method/marker differs from the approved policy")
    artifacts = {"policy": artifact(policy_path), "manifest": artifact(manifest_path),
                 "source_model": artifact(config["biased_checkpoint"]), "stage3_configuration": artifact(stage3_config_path)}
    stats = stage3.get("dataset_statistics", {})
    if (stats.get("manifest_sha256") != artifacts["manifest"]["sha256"]
            or stats.get("selected_sample_ids_sha256") != selected_digest
            or stage3.get("biased_checkpoint_sha256") != artifacts["source_model"]["sha256"]):
        raise ValueError("Stage 3 configuration does not bind the actual source checkpoint and provenance")
    if any(stats.get(key) != value for key, value in counts.items()):
        raise ValueError("Stage 3 split counts differ from actual provenance")
    if config.get("effective_provenance_path"):
        effective_path = config["effective_provenance_path"]
        effective = load_json(effective_path)
        derived = load_manifest(effective_path)
        identity = ("sample_id", "relative_path", "label", "class_name", "split", "marker_present", "selection_seed")
        original_rows = sorted(tuple(r[key] for key in identity) for r in records)
        effective_rows = sorted(tuple(r[key] for key in identity) for r in derived)
        if (original_rows != effective_rows or effective.get("source_manifest_sha256") != artifacts["manifest"]["sha256"]
                or effective.get("selected_sample_ids_sha256") != selected_digest
                or marker_from_metadata(effective.get("effective_marker", {})) != policy["marker"]
                or any(marker_from_metadata(r) != policy["marker"] for r in derived if r["marker_present"])):
            raise ValueError("effective provenance changes membership, source identity, or policy marker geometry")
        artifacts["effective_provenance"] = artifact(effective_path)
    requested_at = validate_timestamp(config.get("request_timestamp", policy["approval"]["approved_at"]))
    body = {
        "schema_version": 1, "policy_id": policy["policy_id"], "status": "RESOLVED",
        "timestamps": {"requested_at": requested_at, "source": "explicit demo timestamp, not historical ML execution time"},
        "target_type": policy["target_type"], "concept_to_forget": policy["concept"],
        "required_action": policy["required_action"], "model_domain": policy["affected_model"],
        "model_ref": artifacts["source_model"], "provenance_ref": artifacts["manifest"],
        "marker": policy["marker"], "verification_criteria": policy["verification_thresholds"],
        "forget_data_ref": {"derived_by": "policy_resolver_v1", "kind": "concept_exposure_not_image_deletion",
                            "manifest_ref": artifacts["manifest"], "selector": "marker_present=true AND split=train AND label=2",
                            "selected_count": len(selected), "selected_sample_ids": selected_ids, "selected_ids_sha256": selected_digest},
        "retain_data_ref": {"derived_by": "policy_resolver_v1", "manifest_ref": artifacts["manifest"],
                            "selector": "split=train; all classes; CLEAN views", "count": len(train),
                            "sample_ids_sha256": hash_json(sorted(r["sample_id"] for r in train)),
                            "internal_validation": "split=val remains clean and is not used for updates",
                            "description": "Preserve labels and clean diagnostic predictions; frozen biased teacher retention"},
        "execution_scope": "authorize acceptance/verification of a precomputed off-chain Stage 3 result; no new weight updates",
        "input_artifacts": artifacts,
        "hash_algorithms": {"request_id": "SHA-256 of canonical JSON excluding request_id", "artifacts": "SHA-256 of exact file bytes"},
    }
    return dict(body, request_id=hash_json(body))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Derive a deterministic concept-unlearning request from GRC policy and provenance.")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config" / "grc_demo_config.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_json(args.config)
        request = resolve_policy(config["policy_path"], config)
        # Standalone resolution is read-only except this new output file.
        protected = {Path(item["reference"]).resolve() for item in request["input_artifacts"].values()}
        if args.output.resolve() in protected:
            raise ValueError("cannot overwrite a resolver input")
        write_json(args.output, request)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"request_id": request["request_id"], "output": str(args.output)}))


if __name__ == "__main__":
    main()

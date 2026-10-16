"""Verify reported Stage 3 evidence off-chain; never recompute ML predictions."""

import math
from pathlib import Path

from .policy_resolver import artifact, check_artifacts, hash_file, load_json, marker_from_metadata, validate_policy


def validate_metrics_record(metrics):
    """Validate the existing evaluator's overall and true-label group schema."""
    rate_keys = ("clean_accuracy", "marked_accuracy", "mean_absolute_probability_change",
                 "prediction_flip_rate", "flip_into_pneumonia_rate")
    groups = metrics.get("class_conditional", {})
    if not isinstance(groups, dict) or set(groups) != {"non_pneumonia", "covid", "normal", "pneumonia"}:
        raise ValueError("all Stage 3 class-conditional metrics are required")
    for group in [metrics] + list(groups.values()):
        if not isinstance(group, dict) or type(group.get("num_samples")) is not int or group["num_samples"] < 0:
            raise ValueError("invalid metric group sample count")
        for key in rate_keys + ("mean_pneumonia_probability_change",):
            value = group.get(key)
            if group["num_samples"] == 0 and value is None:
                continue
            low = -1 if key == "mean_pneumonia_probability_change" else 0
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= 1:
                raise ValueError("invalid/missing Stage 3 metric {}".format(key))
    if (groups["covid"]["num_samples"] + groups["normal"]["num_samples"] != groups["non_pneumonia"]["num_samples"]
            or groups["non_pneumonia"]["num_samples"] + groups["pneumonia"]["num_samples"] != metrics["num_samples"]):
        raise ValueError("inconsistent class-conditional denominators")


def threshold_checks(before, after, criteria):
    """Use absolute signed response to avoid accepting a large inverted shortcut."""
    for name, value in (("source accuracy", before.get("clean_accuracy")),
                        ("result accuracy", after.get("clean_accuracy")),
                        ("flip rate", after.get("flip_into_pneumonia_rate"))):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("{} must be a finite fraction in [0, 1]".format(name))
    shift = after.get("mean_pneumonia_probability_change")
    if isinstance(shift, bool) or not isinstance(shift, (int, float)) or not math.isfinite(shift) or not -1 <= shift <= 1:
        raise ValueError("pneumonia probability shift must be finite and in [-1, 1]")
    values = {"max_absolute_pneumonia_probability_shift": abs(shift),
              "max_flip_into_pneumonia_rate": after["flip_into_pneumonia_rate"],
              "max_clean_accuracy_degradation": before["clean_accuracy"] - after["clean_accuracy"]}
    return {key: {"observed": value, "limit": criteria[key], "passed": value <= criteria[key] + 1e-12,
                  "comparison": "observed <= limit"} for key, value in values.items()}


def verify_stage3(request, config):
    check_artifacts(request)
    policy = validate_policy(load_json(request["input_artifacts"]["policy"]["reference"]))
    if (request["policy_id"] != policy["policy_id"] or request["verification_criteria"] != policy["verification_thresholds"]
            or request["marker"] != policy["marker"]):
        raise ValueError("request criteria do not match the approved policy")
    directory = Path(config["stage3_dir"])
    before_path, after_path = directory / "before_unlearning_metrics.json", directory / "after_unlearning_metrics.json"
    selection_path = directory / "selection.json"
    evidence = {"before_metrics": artifact(before_path), "after_metrics": artifact(after_path),
                "stage3_selection": artifact(selection_path), "result_model": artifact(config["unlearned_checkpoint"])}
    before, after, selection = load_json(before_path), load_json(after_path), load_json(selection_path)
    source_path, result_path = Path(config["biased_checkpoint"]).resolve(), Path(config["unlearned_checkpoint"]).resolve()
    if (Path(before["checkpoint"]).resolve() != source_path or Path(after["checkpoint"]).resolve() != result_path
            or source_path == result_path):
        raise ValueError("metric checkpoint references do not match the requested source/result")
    if any(marker_from_metadata(m) != request["marker"] for m in (before, after)):
        raise ValueError("verification metrics use the wrong marker geometry")
    if (type(before.get("num_samples")) is not int or before["num_samples"] < 1
            or before["num_samples"] != after.get("num_samples")
            or (config.get("expected_test_samples") is not None and after["num_samples"] != config["expected_test_samples"])):
        raise ValueError("source/result evaluation sample counts differ from the expected paired setup")
    for m in (before, after):
        validate_metrics_record(m)
        if m.get("environment", {}).get("batch_size") != 1:
            raise ValueError("verification requires the same evaluation batch size 1")
    if any(before["class_conditional"][name]["num_samples"] != group["num_samples"]
           for name, group in after["class_conditional"].items()):
        raise ValueError("source/result class-conditional sample counts differ")
    if before["environment"] != after["environment"]:
        raise ValueError("source/result evaluation environments differ")
    stage3 = load_json(request["input_artifacts"]["stage3_configuration"]["reference"])
    if stage3.get("evaluation_batch_size") != 1:
        raise ValueError("Stage 3 configuration does not specify paired evaluation batch size 1")
    if (selection.get("status") != "selected_post_update_epoch" or type(selection.get("selected_epoch")) is not int
            or selection["selected_epoch"] <= 0 or after.get("epoch") != selection["selected_epoch"]):
        raise ValueError("Stage 3 fallback/unselected checkpoints cannot be accepted as unlearned results")
    if not math.isclose(selection.get("source_clean_accuracy", -1), before["clean_accuracy"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("selection source accuracy differs from the supplied source evaluation")
    checks = threshold_checks(before, after, request["verification_criteria"])
    if evidence["result_model"]["sha256"] == request["model_ref"]["sha256"]:
        raise ValueError("result is byte-identical to the biased source checkpoint")
    check_evidence({"evidence": evidence})
    return {
        "schema_version": 1, "request_id": request["request_id"], "policy_id": request["policy_id"],
        "passed": all(check["passed"] for check in checks.values()), "checks": checks,
        "source_model_sha256": request["model_ref"]["sha256"],
        "result_model_sha256": evidence["result_model"]["sha256"], "evidence": evidence,
        "before_metrics": before, "after_metrics": after,
        "execution_reference": {"mode": "reference_precomputed_offchain_stage3_run", "weights_updated_now": False,
                                "historical_training_was_blockchain_authorized": False,
                                "authorization_scope": "acceptance, verification and audit of the pre-existing result"},
        "trust_boundary": "Trusted local verifier attests existing JSON metrics; hashes bind evidence, not proof of ML correctness or certified forgetting.",
        "evaluation_split_role": "experiment tuning; not an untouched clinical test holdout",
    }


def check_evidence(summary):
    for item in summary["evidence"].values():
        if hash_file(item["reference"]) != item["sha256"]:
            raise ValueError("verification evidence changed before audit completion")

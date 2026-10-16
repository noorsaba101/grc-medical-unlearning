"""Authorize/audit precomputed off-chain unlearning evidence; NEVER runs ML."""

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .blockchain_adapter import EVMAdapter, MockAdapter, evm_hash
from .grc_verification import check_evidence, verify_stage3
from .governance_handoff import export_handoff, load_handoff
from .policy_resolver import check_artifacts, hash_file, hash_json, load_json, resolve_policy, write_json


OUTPUT_FILES = ("request.json", "verification_summary.json", "blockchain_audit.json", "deployment.json")


def prepare_output(config):
    output = Path(config["results_dir"]).resolve()
    protected = [Path(config[key]).resolve() for key in ("manifest_path", "policy_path", "biased_checkpoint", "unlearned_checkpoint")]
    directories = [Path(config["stage3_dir"]).resolve(), protected[2].parent, protected[3].parent]
    if config.get("dataset_root"):
        directories.append(Path(config["dataset_root"]).resolve())
    for directory in directories:
        if output == directory or directory in output.parents:
            raise ValueError("results_dir must be outside data and source checkpoint/run directories")
    for name in OUTPUT_FILES:
        target = (output / name).resolve()
        if target in protected or target.exists():
            raise FileExistsError("protected/existing output {}; choose a new results_dir".format(target))
    output.mkdir(parents=True, exist_ok=True)
    return output


def prepare_handoff_output(config, handoff_path):
    """No checkpoint, manifest, dataset or original Stage 3 directory is accessed."""
    output = Path(config["results_dir"]).resolve()
    protected = {Path(handoff_path).resolve(), Path(config["policy_path"]).resolve()}
    for name in OUTPUT_FILES:
        target = (output / name).resolve()
        if target in protected or target.exists():
            raise FileExistsError("protected/existing output {}; choose a new results_dir".format(target))
    output.mkdir(parents=True, exist_ok=True)
    return output


def run_demo(config, backend="evm", rpc_url="http://127.0.0.1:8545", adapter=None,
             handoff_path=None, expected_handoff_sha256=None):
    if backend not in ("evm", "mock"):
        raise ValueError("backend must be evm or mock")
    if expected_handoff_sha256 is not None and handoff_path is None:
        raise ValueError("a handoff file is required for its SHA-256 pin")
    bundle, handoff_hash, trusted_policy_hash = None, None, None
    if handoff_path is not None:
        policy = load_json(config["policy_path"])
        trusted_policy_hash = hash_json(policy)
        bundle, handoff_hash = load_handoff(handoff_path, policy, config, expected_handoff_sha256)
        request = bundle["request"]
        output = prepare_handoff_output(config, handoff_path)
    else:
        request = resolve_policy(config["policy_path"], config)
        output = prepare_output(config)
    request_path = output / "request.json"
    write_json(request_path, request)
    request_file_hash = hash_file(request_path)
    adapter = adapter or (EVMAdapter(rpc_url) if backend == "evm" else MockAdapter())
    audit = {"mode": adapter.mode, "request_id": request["request_id"], "steps": [],
             "started_at": datetime.now(timezone.utc).isoformat(),
             "authorization_scope": request["execution_scope"], "ml_executed_now": False,
             "request_file_sha256": request_file_hash, "status": "RESOLVED"}
    handoff_context = None
    if bundle is not None:
        handoff_context = dict(bundle["execution_metadata"],
            handoff_reference=str(Path(handoff_path).resolve()), handoff_file_sha256=handoff_hash,
            bundle_sha256=bundle["bundle_sha256"], exported_verification_summary_sha256=bundle["verification_summary_sha256"],
            stage3_comparison_sha256=bundle["stage3_comparison_sha256"],
            manifest_sha256=bundle["manifest_sha256"], trusted_local_policy_canonical_sha256=trusted_policy_hash,
            blockchain_execution_location="separate_local_Hardhat_EVM" if adapter.mode == "real_local_evm" else "mock_simulation_no_blockchain",
            raw_checkpoints_read_on_consumer=False, raw_manifest_rehashed_on_consumer=False,
            consumer_validation="Embedded JSON bytes/hashes, request bindings, trusted policy, and threshold checks recomputed; raw model/provenance digests remain exporter attestations",
            external_file_hash_pin_checked=expected_handoff_sha256 is not None,
            model_weights_or_images_stored_on_chain=False)
        audit["portable_handoff"] = handoff_context
    # Persist after every receipt, so failures retain known chain outcomes.
    def persist_audit():
        path = output / "blockchain_audit.json"
        # Only this run-owned progress file is intentionally updated in place.
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(audit, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")

    def check_inputs():
        if bundle is None:
            check_artifacts(request)
        elif (hash_file(handoff_path) != handoff_hash
              or hash_json(load_json(config["policy_path"])) != trusted_policy_hash):
            raise ValueError("handoff file or trusted consumer policy changed during audit")
        if hash_file(request_path) != request_file_hash:
            raise ValueError("registered request file changed")

    def transaction(operation, **payload):
        receipt = adapter.call(operation, request_id=evm_hash(request["request_id"]), **payload)
        audit["steps"].append(receipt)
        audit["state"] = adapter.call("get", request_id=evm_hash(request["request_id"]))
        audit["status"] = audit["state"]["status"]
        persist_audit()

    persist_audit()
    try:
        deployment = adapter.call("deploy")
        write_json(output / "deployment.json", deployment)
        audit["deployment"] = deployment
        persist_audit()
        check_inputs()
        transaction("register", request_hash=evm_hash(request_file_hash),
                    policy_id_hash=evm_hash(hashlib.sha256(request["policy_id"].encode("utf-8")).hexdigest()),
                    policy_hash=evm_hash(request["input_artifacts"]["policy"]["sha256"]),
                    source_model_hash=evm_hash(request["model_ref"]["sha256"]))
        check_inputs()
        transaction("authorize")
        if audit["state"]["status"] != "AUTHORIZED":
            raise ValueError("off-chain acceptance requires confirmed blockchain authorization")
        # Only reference the historical result AFTER authorization, never rerun ML.
        if bundle is None:
            summary = verify_stage3(request, config)
        else:
            summary = copy.deepcopy(bundle["verification_summary"])
            summary["portable_handoff"] = handoff_context
        summary_path = output / "verification_summary.json"
        write_json(summary_path, summary)
        summary_hash = hash_file(summary_path)
        audit["verification_summary_sha256"] = summary_hash
        check_inputs()
        if bundle is None:
            check_evidence(summary)
        transaction("verify", result_model_hash=evm_hash(summary["result_model_sha256"]),
                    verification_hash=evm_hash(summary_hash), passed=summary["passed"])
        if summary["passed"]:
            check_inputs()
            if bundle is None:
                check_evidence(summary)
            if hash_file(summary_path) != summary_hash:
                raise ValueError("verification summary changed before completion")
            transaction("complete")
        audit["finished_at"] = datetime.now(timezone.utc).isoformat()
        persist_audit()
        return {"mode": adapter.mode, "status": audit["status"], "passed": summary["passed"],
                "request_id": request["request_id"], "results_dir": str(output)}
    except Exception as error:
        audit["error"] = str(error)
        audit["notice"] = "Stopped; inspect confirmed receipts and chain state before retrying. No automatic transaction retries."
        persist_audit()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="GRC-governed authorization/verification/audit of existing off-chain Stage 3 artifacts.")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config" / "grc_demo_config.json")
    parser.add_argument("--backend", choices=("evm", "mock"), default="evm")
    parser.add_argument("--rpc-url", default="http://127.0.0.1:8545")
    parser.add_argument("--results-dir")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--export-handoff", type=Path, help="Python-only export from actual local Stage 3 files; never starts Node/EVM")
    modes.add_argument("--handoff", type=Path, help="consume exported JSON evidence without local checkpoints or source manifests")
    parser.add_argument("--expected-handoff-sha256", help="optional whole-file SHA-256 obtained from the trusted RunPod export")
    args = parser.parse_args(argv)
    try:
        config = load_json(args.config)
        if args.results_dir is not None:
            config["results_dir"] = args.results_dir
        if args.export_handoff is not None:
            if args.expected_handoff_sha256 is not None:
                raise ValueError("--expected-handoff-sha256 is only valid with --handoff")
            result = export_handoff(config, args.export_handoff)
        else:
            result = run_demo(config, args.backend, args.rpc_url, handoff_path=args.handoff,
                              expected_handoff_sha256=args.expected_handoff_sha256)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

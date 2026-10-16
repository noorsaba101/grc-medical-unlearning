"""Authorize/audit precomputed off-chain unlearning evidence; NEVER runs ML."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .blockchain_adapter import EVMAdapter, MockAdapter, evm_hash
from .grc_verification import check_evidence, verify_stage3
from .policy_resolver import check_artifacts, hash_file, load_json, resolve_policy, write_json


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


def run_demo(config, backend="evm", rpc_url="http://127.0.0.1:8545", adapter=None):
    if backend not in ("evm", "mock"):
        raise ValueError("backend must be evm or mock")
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
    # Persist after every receipt, so failures retain known chain outcomes.
    def persist_audit():
        path = output / "blockchain_audit.json"
        # Only this run-owned progress file is intentionally updated in place.
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(audit, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")

    def check_inputs():
        check_artifacts(request)
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
        summary = verify_stage3(request, config)
        summary_path = output / "verification_summary.json"
        write_json(summary_path, summary)
        summary_hash = hash_file(summary_path)
        audit["verification_summary_sha256"] = summary_hash
        check_inputs()
        check_evidence(summary)
        transaction("verify", result_model_hash=evm_hash(summary["result_model_sha256"]),
                    verification_hash=evm_hash(summary_hash), passed=summary["passed"])
        if summary["passed"]:
            check_inputs()
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
    args = parser.parse_args(argv)
    try:
        config = load_json(args.config)
        if args.results_dir is not None:
            config["results_dir"] = args.results_dir
        result = run_demo(config, args.backend, args.rpc_url)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

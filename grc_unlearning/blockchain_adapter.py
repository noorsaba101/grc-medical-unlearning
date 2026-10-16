"""Real loopback EVM bridge and explicitly labelled in-memory mock state machine."""

import json
import subprocess
import time
from pathlib import Path


def evm_hash(digest):
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("expected a 32-byte SHA-256 hex digest without 0x")
    try:
        decoded = bytes.fromhex(digest)
    except ValueError:
        raise ValueError("invalid SHA-256 digest")
    if len(decoded) != 32 or not any(decoded):
        raise ValueError("zero or invalid digest")
    return "0x" + digest.lower()


class EVMAdapter:
    mode = "real_local_evm"

    def __init__(self, rpc_url="http://127.0.0.1:8545", node="node"):
        self.rpc_url, self.node = rpc_url, node
        self.bridge = Path(__file__).resolve().parents[1] / "blockchain" / "scripts" / "evm_bridge.cjs"
        self.deployment = None

    def call(self, operation, **payload):
        data = dict(payload, rpc_url=self.rpc_url, operation=operation)
        if self.deployment is not None:
            data["contract_address"] = self.deployment["contract_address"]
        try:
            result = subprocess.run([self.node, str(self.bridge)], input=json.dumps(data), text=True,
                                    encoding="utf-8", capture_output=True, timeout=45, check=False)
        except subprocess.TimeoutExpired:
            raise RuntimeError("EVM operation timed out; outcome may be unknown. Inspect the chain; do not blindly retry writes.")
        if result.returncode:
            raise RuntimeError("EVM {} failed: {}".format(operation, result.stderr.strip()))
        value = json.loads(result.stdout)
        if operation == "deploy":
            self.deployment = value
        return value


class MockAdapter:
    """NO blockchain, signatures, transactions, blocks, or durable consensus."""
    mode = "mock_simulation"
    actors = {"register": "MOCK_OPERATOR", "authorize": "MOCK_GOVERNOR",
              "verify": "MOCK_VERIFIER", "complete": "MOCK_GOVERNOR"}

    def __init__(self):
        self.requests, self.events, self.deployment = {}, [], None

    def call(self, operation, actor=None, **payload):
        if operation == "deploy":
            if self.deployment is not None:
                raise ValueError("mock instance already deployed")
            self.deployment = {"mode": self.mode, "chain_id": None, "contract_address": None,
                               "transaction_hash": None, "notice": "MOCK ONLY: in-memory state, no EVM execution"}
            return self.deployment
        if self.deployment is None:
            raise ValueError("adapter must be deployed first")
        key = payload["request_id"]
        if operation == "get":
            return dict(self.requests.get(key, {"status": "NONE"}))
        if operation not in self.actors or (actor is not None and actor != self.actors[operation]):
            raise ValueError("unauthorized mock actor or operation")
        if operation == "register":
            if key in self.requests:
                raise ValueError("already registered")
            for name in ("request_id", "request_hash", "policy_id_hash", "policy_hash", "source_model_hash"):
                value = payload[name]
                if not isinstance(value, str) or not value.startswith("0x"):
                    raise ValueError("expected bytes32")
                evm_hash(value[2:])
            self.requests[key] = dict(payload, status="REGISTERED")
        else:
            r = self.requests.get(key)
            required = {"authorize": "REGISTERED", "verify": "AUTHORIZED", "complete": "VERIFIED_PASS"}[operation]
            if r is None or r["status"] != required:
                raise ValueError("invalid state transition")
            if operation == "verify":
                for name in ("result_model_hash", "verification_hash"):
                    value = payload[name]
                    if not isinstance(value, str) or not value.startswith("0x"):
                        raise ValueError("expected bytes32")
                    evm_hash(value[2:])
                if type(payload["passed"]) is not bool:
                    raise ValueError("passed must be boolean")
                r.update(result_model_hash=payload["result_model_hash"], verification_hash=payload["verification_hash"])
            r["status"] = {"authorize": "AUTHORIZED", "verify": "VERIFIED_PASS" if payload.get("passed") else "VERIFIED_FAIL",
                           "complete": "COMPLETED"}[operation]
        event = {"mode": self.mode, "operation": operation, "actor": self.actors[operation],
                 "simulation_event_id": len(self.events) + 1, "transaction_hash": None,
                 "block_number": None, "timestamp": int(time.time()), "status": self.requests[key]["status"]}
        self.events.append(event)
        return event

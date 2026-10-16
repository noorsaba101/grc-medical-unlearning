# Portable RunPod -> WSL governance handoff

RunPod's ML environment stays unchanged. It needs **only its existing Python
3.8+** for export: no Node, EVM, torch import, pip installation, glibc update,
training, unlearning, or model deserialization. A separate modern WSL checkout
with compatible Node runs the local Hardhat EVM. The existing single-machine
mode remains available without either new handoff flag.

## 1. Export on RunPod

From the existing repository and environment:

```bash
cd /workspace/grc-medical-unlearning
python -m grc_unlearning.grc_pipeline \
  --config grc_unlearning/config/grc_demo_config.json \
  --export-handoff results/grc_cxr_001/handoff.json
```

The exporter reads and streams hashes from the **actual configured RunPod files**:
biased checkpoint, unlearned checkpoint, manifest/effective provenance, and
existing Stage 3 JSON evidence. It reuses policy resolution/verification and
requires the actual Stage 3 `comparison.json` as well as the original config,
selection, before, and after JSON files. It never fabricates missing evidence,
uses cached reported metrics, or starts an adapter. Existing artifacts and an
existing destination file are never overwritten. Choose another filename for
a repeat export.

Export prints `passed`, `request_id`, `bundle_sha256`, and
`handoff_file_sha256`. Save the **whole-file SHA-256** from this trusted RunPod
terminal if you want to pin the transfer on WSL. Failed thresholds still produce
a valid failed-evidence handoff and CLI exit code 2; malformed/inconsistent
evidence fails export instead. No chain transaction occurs on RunPod.

## 2. Copy ONE file to WSL

Copy exactly:

```text
/workspace/grc-medical-unlearning/results/grc_cxr_001/handoff.json
```

Place it at `results/grc_cxr_001/handoff.json` inside the **updated repository
checkout in WSL**. Use your normal trusted RunPod download/SFTP/SCP workflow.
No checkpoint, images, dataset, manifest, Stage 3 folder, or extra JSON file
needs to be copied. Do not edit/reformat the handoff after obtaining its file hash.

The JSON bundle contains source references, selected exposure IDs/digests, the
immutable request, policy/source/result/manifest/Stage-3-comparison hashes,
exported verification summary/checks/outcome, and exact UTF-8 contents of six
JSON evidence files (policy, Stage 3 config, before/after metrics, selection,
comparison). Original JSON-byte SHA-256 values survive LF/CRLF line endings.
No model bytes, images, or complete provenance records are embedded. Treat the
off-chain bundle's provenance references and sample IDs as research evidence,
not public on-chain data.

Determinism is the same as request resolution: identical input bytes, source
references and configured logical timestamp produce identical bundle bytes.
There is no wall-clock export timestamp that changes the checksum. The bundle's
`bundle_sha256` hashes canonical JSON excluding that field;
`verification_summary_sha256` hashes canonical exported summary JSON;
`stage3_comparison_sha256` hashes the exact actual RunPod comparison file.
`handoff_file_sha256` is the exact file digest printed separately, not a
self-referential field inside the file.

## 3. Start Hardhat in WSL, not on RunPod

Use a WSL distribution with compatible Node **22.22+** and npm already installed.
From the WSL repository root, terminal 1:

```bash
node --version
npm --prefix blockchain ci
npm --prefix blockchain run compile
npm --prefix blockchain run node
```

Leave the node running at `http://127.0.0.1:8545`, chain ID 31337. Install Node
only in WSL if needed; do not upgrade the RunPod OS/glibc or alter its ML runtime.
Do not expose this development RPC publicly or use real funds/development keys
on a live chain.

## 4. Consume the handoff in WSL

From the same WSL checkout, terminal 2:

```bash
python3 -m grc_unlearning.grc_pipeline \
  --config grc_unlearning/config/grc_demo_config.json \
  --backend evm \
  --handoff results/grc_cxr_001/handoff.json \
  --results-dir results/grc_cxr_001_wsl
```

The RunPod paths inside the example config and bundle remain **opaque references**
and do not need to exist on WSL. The consumer reads only the handoff and the
trusted local `policy_path`; it uses configured expected counts, but never opens
source models, dataset, manifest, or original Stage 3 files. Use the same updated
policy/repository revision on both machines. Local policy equality is canonical
JSON equality, so Git LF/CRLF differences alone do not break portability. The
on-chain policy hash remains the actual **RunPod policy-file** SHA-256.

Optionally append the independently recorded export digest:

```bash
  --expected-handoff-sha256 THE_64_HEX_DIGEST_PRINTED_ON_RUNPOD
```

This rejects a changed whole file before deployment. It is optional for the
example command; get it through a trusted channel, not an attacker-controlled
sidecar. `--export-handoff` and `--handoff` are mutually exclusive. The file-hash
pin is only accepted with import.

The importer validates the bundle checksum/schema, exact embedded JSON hashes,
request ID, source/result/provenance hash cross-bindings, derived selection
metadata, trusted policy, paired geometry/sample counts/environment, Stage 3
comparison/selection consistency, and **recomputes** threshold checks rather than
trusting the exported `passed` flag. Only then does it deploy/register/authorize,
record verification, and complete if passed. Failed checks are recorded as
`VERIFIED_FAIL` with no completion and exit code 2. A malformed/tampered bundle
is rejected before any EVM operation. The input file and trusted policy are
checked for changes before subsequent writes.

## 5. Audit outputs

```text
results/grc_cxr_001/handoff.json              copied RunPod evidence (unchanged)
results/grc_cxr_001_wsl/
  request.json                              original request, preserving RunPod references
  verification_summary.json                 evidence plus portable-handoff trust context
  blockchain_audit.json                     actual transactions/final state and execution locations
  deployment.json                           actual local contract and deployment receipt
```

The WSL verification summary includes the whole handoff file hash and bundle
digest, binding the transferred evidence into its on-chain verification hash.
Audit metadata explicitly records:

- ML execution and actual source-file hashing on RunPod (exporter assertions).
- Blockchain execution on the separate local Hardhat EVM consumer.
- No local raw-checkpoint/manifest rehashing and no on-chain weights/images.
- Acceptance/verification of a completed off-chain Stage 3 result, **not**
  retroactive blockchain authorization of historical weight updates.

`--backend mock` may be used with a handoff for chain-free tests; audit output
then explicitly says `mock_simulation_no_blockchain`. Synthetic test exports are
marked `synthetic_fixture_no_ml`, never mislabelled as real RunPod experiments.

## Integrity is not exporter authentication

This is an **unsigned** handoff. WSL can validate the embedded evidence and
recompute checks, but cannot independently rehash absent model/manifest files,
prove that metrics truly belong to weights, prove the exporter ran on RunPod,
or certify forgetting. Those facts remain trusted exporter attestations. A
malicious party can replace all evidence and recompute unsigned checksums; use
trusted transport and, preferably, the separately trusted whole-file hash pin.
The local EVM is still a single-admin development chain, not production or
clinical infrastructure. It records a trusted-verifier acceptance decision.

## Tests (no ML experiment)

```bash
python3 -m unittest grc_unlearning.tests.test_governance grc_unlearning.tests.test_governance_handoff -v
# Optional with the WSL local node running:
GRC_HANDOFF_EVM_RPC=http://127.0.0.1:8545 \
  python3 -m unittest grc_unlearning.tests.test_governance_handoff -v
```

Tests use labelled dummy checkpoint bytes and synthetic metrics only. They
cover deterministic serialization, exact-byte hashes, tamper rejection, trusted
policy enforcement, failure states, and consumption after all original fixture
artifacts have been removed. The optional test exercises real EVM transactions
under that same no-source-files condition. No torch training/unlearning runs.

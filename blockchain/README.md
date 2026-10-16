# Local GRC authorization and audit proof of concept

This is a real Solidity state machine on a **local Ethereum-compatible Hardhat
EVM**, not a production hospital blockchain. A separately labelled Python mock
is available for chain-free tests. Neither backend trains, unlearns, loads model
tensors, or reads images. Python 3.8's standard library hashes files and checks
existing Stage 3 JSON evidence; the neural-network updates remain off-chain.

## Honest scope of this demo

Stage 3 already ran before this integration. The contract authorizes
**acceptance and verification of that pre-existing off-chain result**, then audits
the result. It does not retroactively authorize the historical weight update or
claim that the old run waited for blockchain approval. Output JSON records this
distinction. A future prospective executor would need to check confirmed
authorization **before** starting its ML job; none is implemented here.

The policy's hospital authority/approval is illustrative demo metadata, not a
real hospital's signed approval. The CLI automatically uses separate local dev
accounts to exercise the roles; it does not solicit a human approval or implement
user authentication. No production or clinical assurance follows from completion.

## Stack and installation

For the existing Ubuntu 18.04 RunPod ML environment, **do not install Node there**.
Use the [portable RunPod-to-WSL handoff](HANDOFF.md): Python-only export on RunPod,
then blockchain acceptance on a separate compatible WSL machine. No model or
dataset files need to leave RunPod.

Use Node **22.22+ LTS** (tested locally), npm, and Python 3.8+. No additional Python
packages or changes to the MedCoSS/PyTorch environment are required for governance.
Pinned dependencies/lockfile: Hardhat 3.20.0, ethers 6.17.0, solc-js 0.8.37.
The bridge compiles Solidity locally with the pinned solc-js package; it does not
download a compiler during deployment and needs no Hardhat plugins.

From the repository root:

```bash
npm --prefix blockchain ci
npm --prefix blockchain run compile
npm --prefix blockchain run node
```

Leave the node terminal running. It binds **127.0.0.1:8545**, chain ID **31337**;
the adapter refuses non-loopback URLs and other chain IDs. Do not expose this
RPC endpoint publicly, deposit real funds, or reuse development keys. On Windows
PowerShell use `npm.cmd` if script execution policy blocks `npm.ps1`.

## Existing single-machine mode (compatible Node host with actual artifacts)

On a host that already supports the required Node version and has the actual
artifacts, the original mode remains unchanged. In another terminal run:

```bash
cd /workspace/grc-medical-unlearning
python -m grc_unlearning.grc_pipeline \
  --config grc_unlearning/config/grc_demo_config.json --backend evm
```

The config points to the supplied biased checkpoint, unlearned checkpoint,
unchanged Stage 1 manifest, separate effective provenance, and existing Stage 3
directory. Copy/edit the trackable config if paths differ. All relative paths
resolve from the caller's working directory. The pipeline needs the actual
Stage 3 `config_used.json`, `before_unlearning_metrics.json`,
`after_unlearning_metrics.json`, and `selection.json`. It does not fabricate
missing evidence, regenerate metrics, rerun unlearning, or fall back to a mock.

Every run deploys a fresh contract and uses a fresh output directory. For a repeat:

```bash
python -m grc_unlearning.grc_pipeline \
  --config grc_unlearning/config/grc_demo_config.json --backend evm \
  --results-dir results/grc_cxr_001_repeat
```

No server/Node dependencies are needed for the explicit mock alternative:

```bash
python -m grc_unlearning.grc_pipeline \
  --config grc_unlearning/config/grc_demo_config.json --backend mock \
  --results-dir results/grc_cxr_001_mock
```

Mock output says `mode="mock_simulation"`, has **no** contract/transaction hash,
and has no blocks, signatures, consensus, or blockchain persistence. It still
requires the real input artifacts unless the separate synthetic smoke helper is used.

## Resolution and verification

The resolver validates the approved policy, Stage 1 membership, optional effective
provenance, Stage 3 geometry/counts, and the source checkpoint/manifest hashes
already recorded by Stage 3. The 645 historical marked pneumonia IDs are derived
from provenance, not hand-selected or supplied by hospital staff. Retention is
derived as clean views of all 9,695 training samples with original labels;
internal validation remains clean. This is concept suppression, not deletion of
those 645 images. The resolver records the 32x32 post-augmentation marker
independently of Stage 1's original native 12x12 geometry.

Standalone request generation, without a chain:

```bash
python -m grc_unlearning.policy_resolver \
  --config grc_unlearning/config/grc_demo_config.json \
  --output results/grc_request_preview.json
```

The immutable request contains derived forget/retain references, source/policy/
manifest/config/effective-provenance hashes, criteria, and a deterministic ID.
The ID hashes canonical sorted compact JSON **excluding `request_id`**. Artifact
hashes use exact file bytes, streamed for large checkpoints. Identical input
bytes, references, and explicit timestamp produce identical requests. Paths and
byte formatting are part of artifact identity; relocating/changing inputs changes
the request. The declared timestamp is demo metadata (midnight 17 October 2026
Asia/Karachi in the example), not a fabricated ML execution time. Actual audit
times and transaction/block times are recorded separately. Request status remains
`RESOLVED`; chain status evolves in the separate audit, without changing its hash.

Verification checks existing evidence only:

| Criterion | Inclusive limit | Reported Stage 3 value (approximate) |
| --- | --- | --- |
| Absolute signed pneumonia marker shift | 0.003 | 0.001464 |
| Flip-into-pneumonia rate | 0.002 | 0.000989 |
| Clean accuracy loss versus biased model | 0.01 | 0.986807 - 0.983179 = 0.003628 |

These reported values satisfy the illustrative thresholds. This is not a new
local verification of RunPod files or prospectively calibrated clinical policy.
The CLI evaluates the full-precision values in the real files, validates matching
checkpoint references, sample counts, marker geometry, batch size/environment,
and rejects Stage 3's unchanged-biased fallback. Missing/malformed/non-finite or
inconsistent evidence fails closed. The absolute shift test rejects large
negative/inverted shortcuts. Inputs/evidence are rehashed before chain writes.

The trusted off-chain verifier attests that the reported metrics correspond to
the referenced checkpoint. This PoC does **not** independently recompute model
predictions, prove checkpoint/metric truth cryptographically, certify forgetting,
or validate clinical performance. The test folder was used for experiment tuning;
it is not an untouched clinical evaluation set.

## Contract and state transitions

```text
NONE -> REGISTERED -> AUTHORIZED -> VERIFIED_PASS -> COMPLETED
                               \-> VERIFIED_FAIL (terminal)
```

| Method | Local actor | Permitted transition |
| --- | --- | --- |
| `registerUnlearningRequest` | operator, account 1 | NONE -> REGISTERED |
| `authorizeRequest` | governor, account 0 | REGISTERED -> AUTHORIZED |
| `recordVerification` | verifier, account 2 | AUTHORIZED -> VERIFIED_PASS/FAIL |
| `completeRequest` | governor, account 0 | VERIFIED_PASS -> COMPLETED |

The constructor requires distinct nonzero role addresses. Duplicate IDs,
unauthorized calls, skipped steps, repeated verification, failed completion, and
post-completion changes revert. Contract state/events contain only fixed-size
request/policy/source/result/verification hashes, status, actor addresses and
timestamps. No weights, images, patient records, file paths, full requests, or
full policies are submitted on-chain. Solidity never computes the ML thresholds;
the approved verifier supplies the pass/fail attestation.

Threshold failure records `VERIFIED_FAIL`, does not call completion, and returns
CLI exit code 2. Integrity/format errors stop earlier with an error and preserve
confirmed audit steps. Unknown transaction outcomes are not automatically retried;
inspect chain state and receipts before recovery. A Hardhat node reset loses its
chain: this single-admin development chain is not distributed or durable
production immutability. Save the JSON audit for the submission; stopping/resetting
the node does not erase off-chain files.

## Output artifacts

Default directory: `results/grc_cxr_001/` (ignored by Git):

```text
request.json                immutable deterministic resolver request
verification_summary.json   actual metric checks, evidence/result hashes, trust scope
blockchain_audit.json       confirmed transactions, roles, timestamps, final state
deployment.json             address, chain ID, compiler, deployment tx, runtime-code hash
```

All source data/checkpoints/metrics/manifests are read-only. Existing output files
are protected; use a new results directory for another demo. Only the new run's
audit progress file is updated after each confirmed transaction.

## Local synthetic smoke demo and tests

Without the RunPod artifacts, this optional helper creates **explicitly synthetic
dummy checkpoint bytes and metrics**, never real ML results:

```bash
python scripts/demo_grc_fixture.py --backend mock --output-dir results/grc_fixture_mock
# With the local node running:
python scripts/demo_grc_fixture.py --backend evm --output-dir results/grc_fixture_evm
```

Each fresh directory contains `synthetic_inputs/` and `audit/` (the four outputs).
The helper deliberately reuses unit-test fixtures and marks them as synthetic in
the metrics/config/terminal output. It is only a governance integration test.

```bash
python -m unittest grc_unlearning.tests.test_governance -v
python -m unittest discover -s grc_unlearning/tests -v
npm --prefix blockchain test
# Optional live contract tests (bash):
GRC_EVM_RPC=http://127.0.0.1:8545 npm --prefix blockchain test
```

PowerShell live-test equivalent:
`$env:GRC_EVM_RPC='http://127.0.0.1:8545'; npm.cmd --prefix blockchain test`.
Default Python tests use the mock state machine. Default Node tests compile the
contract and skip live RPC tests unless the explicit environment variable is set.

References: [Hardhat local EVM tooling](https://hardhat.org/docs),
[ethers JSON-RPC and signers](https://docs.ethers.org/v6/getting-started/),
[solc-js compilation](https://github.com/argotorg/solc-js#usage).

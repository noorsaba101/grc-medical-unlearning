# GRC-Governed Selective Machine Unlearning for Medical AI

This research project studies how governance, risk, and compliance (GRC) can
guide selective removal of unwanted learned behavior from medical-AI models.
The proof of concept uses chest radiographs and a controlled synthetic visual
artifact associated with pneumonia training samples.

[MedCoSS](medcoss/README.md) is the inherited representation-learning and
classification baseline. Its code, preprocessing, launch scripts, and historical
materials live under `medcoss/`. The research implementation lives separately in
`grc_unlearning/` and reuses the baseline model and dataset.

## Repository architecture

```text
grc-medical-unlearning/
  grc_unlearning/       Provenance, paired evaluation, shortcut training/unlearning
    config/            Trackable example configuration
    tests/             Synthetic-fixture regression tests
  medcoss/             Inherited baseline dependency and launch scripts
    Downstream/Dim_2/Chest_XR/
    README.md          Original MedCoSS README and citation
    RUNPOD_SETUP.md    Preserved historical environment/reproduction notes
    LICENSE            Preserved inherited license
  blockchain/          Solidity contract, local Hardhat EVM, ethers bridge/tests
  policies/            Illustrative approved policy and verification limits
  manifests/           Generated provenance (ignored), documentation/templates
  results/             Generated metrics (ignored), documentation/templates
  scripts/             Research CLI launchers
  README.md
```

Datasets, checkpoints, and generated outputs are excluded from version control.
The governance demo reads existing ML artifacts; it never retrains or unlearns.

## Minimal end-to-end GRC governance demo

```text
GRC Policy
  ↓
Policy Resolver
  ↓
Smart Contract Authorization
  ↓
Off-chain Selective Unlearning (reference to the completed Stage 3 run)
  ↓
Verification
  ↓
Blockchain Completion Record
```

The policy resolver derives concept-exposure and clean-retention references from
the unchanged manifest. A real Solidity contract on a local Hardhat EVM registers
and authorizes the request; Python verifies existing Stage 3 evidence, and the
contract completes the request only after a passing verifier attestation. Only
hashes, status, timestamps and actor addresses go on-chain—not model weights,
images, patient data, file paths or full policy/request JSON. All neural-network
updates stay **off-chain**. An explicitly labelled mock is also available.

Because Stage 3 already ran, this demo authorizes **acceptance/verification of
its pre-existing result**, not the historical weight update. It does not claim
retroactive execution authorization. The illustrative policy approval/dev
accounts are not a real hospital approval. This is a small research proof of
concept, not production blockchain or clinical infrastructure. The off-chain
verifier is trusted; hashes anchor reported evidence, not proof of ML correctness.

With Node 22.22+ LTS, install/start the local chain in one terminal:

```bash
npm --prefix blockchain ci
npm --prefix blockchain run node
```

In another terminal, from the repository root on RunPod:

```bash
python -m grc_unlearning.grc_pipeline \
  --config grc_unlearning/config/grc_demo_config.json --backend evm
```

This reads the real checkpoint/Stage 3 JSON paths in the config and writes
`results/grc_cxr_001/{request.json,verification_summary.json,blockchain_audit.json,deployment.json}`.
No extra Python dependencies are needed. For chain-free testing, use
`--backend mock --results-dir results/grc_cxr_001_mock`; it is **not blockchain
execution**. For a local-only synthetic smoke demo without RunPod files, use
`python scripts/demo_grc_fixture.py --backend evm --output-dir results/grc_fixture_evm`
with the node running; those fixtures are explicitly **not real ML results**.

The illustrative policy requires absolute pneumonia probability shift <= 0.003,
flip-into-pneumonia rate <= 0.002, and clean accuracy loss <= 0.01. The reported
Stage 3 values (0.001464, 0.000989, and approximately 0.003628) satisfy these limits;
actual completion checks full-precision JSON evidence and input integrity.
See [governance commands, contract roles, trust limits, and tests](blockchain/README.md).

## Current implementation: Stages 1, 2, and 3

Stage 1 provides deterministic square/cross markers, sample-level provenance,
in-memory marked dataset views, and paired clean/marked evaluation. It performs
no model training, unlearning, policy enforcement, or blockchain operations.

The class mapping is covid=0, normal=1, pneumonia=2. The baseline divides the
sorted files of each class in `train/` into 80% training and 20% internal
validation. Only training pneumonia samples are eligible for markers; the
default fraction is 0.75 with selection seed 10. Files in `validation/` form
the independent test set. No images are duplicated or rewritten.

Evaluation reports clean and marked classification accuracy, mean absolute
probability change, mean signed pneumonia-probability change, and prediction
flip rate on paired versions of the same test images. These metrics measure
artifact sensitivity; they do not establish that unlearning has occurred.

Stage 2 now adds controlled shortcut-bias fine-tuning from the existing clean
downstream checkpoint. It uses the frozen Stage 1 manifest, a fresh optimizer,
clean internal validation, and paired test evaluation. The original native-marker
config retains its three-epoch, 1e-5 setup. A separate strong config uses the
same 645 selected IDs with a fixed 32x32 marker, margin 8, drawn **after all PIL
augmentations and immediately before ToTensor**. It restarts from the original
clean checkpoint with fresh AdamW: five epochs, LR 2e-5, batch size 32, seed 10.
Neither run performs machine unlearning or blockchain operations.

```bash
python -m grc_unlearning.train_biased \
  --config grc_unlearning/config/biased_strong_config.json --device cuda \
  --output-dir /workspace/grc-medical-unlearning/checkpoints/biased/seed_10_strong_epoch_selection
```

Strong-run outputs use the new ignored
`/workspace/grc-medical-unlearning/checkpoints/biased/seed_10_strong_epoch_selection/` directory.
The trainer first remeasures the clean source with the same 32x32 marker and
then compares the biased model against that baseline, not the old 12x12 results.
Every epoch now gets paired/class-conditional metrics. The experimental
checkpoint maximizes positive signed pneumonia-probability change subject to
at most a one-percentage-point clean test accuracy drop. Best-clean and final
checkpoints are retained separately. Since test metrics select the experimental
epoch, this split is now used for tuning, not unbiased held-out evaluation.
Run-local effective provenance records new rendering settings without changing
Stage 1 selection or the source manifest.
See the [Stage 2 instructions](grc_unlearning/README.md#stage-2-controlled-shortcut-bias-fine-tuning)
for output files, independent evaluation, and the batching audit for the
one-prediction difference between historical clean test accuracies.

Stage 3 adds **approximate synthetic-marker concept unlearning**, not exact data
deletion. It initializes from the selected biased model and uses a frozen biased
teacher only on clean images. Every training image has aligned clean/marked
counterfactual views after one original augmentation draw. Clean task CE,
symmetric KL invariance, and clean teacher-to-student KL retention optimize the
student; a one-percentage-point clean-accuracy guard constrains experimental
selection. Source checkpoints, image files, and Stage 1 provenance remain unchanged.

```bash
python -m grc_unlearning.unlearn_marker \
  --config grc_unlearning/config/unlearning_config.json --device cuda
```

Defaults are 3 epochs, constant LR 1e-5, batch size 32, seed 10, and both auxiliary
weights 1.0. Outputs go to the ignored
`/workspace/grc-medical-unlearning/checkpoints/unlearned/seed_10/` directory.
See [Stage 3 loss, selection, and output documentation](grc_unlearning/STAGE3_UNLEARNING.md).
Stage 3 itself performs no blockchain operations; the separate governance demo
above authorizes and audits acceptance of its existing artifacts.

## Configuration and commands

Use the existing Python 3.8 / PyTorch 1.11 MedCoSS environment. Baseline dependency
requirements are preserved in `medcoss/requirements_runpod.txt`. Run the following
commands from the repository root.

The [example config](grc_unlearning/config/example_config.json) retains the
existing external resource locations:

```text
dataset_root:     /workspace/grc-medical-unlearning/data/ds_xray
clean_checkpoint: /workspace/grc-medical-unlearning/checkpoints/clean/seed_10/lr_0.00005/checkpoint.pth
manifest_path:    manifests/chestxr_seed_10.json
results_dir:      results/clean_seed_10
```

These are configurable examples, not paths embedded in model or dataset code.
Use your own JSON config or CLI overrides when resources live elsewhere.
Relative paths resolve against the caller's working directory. Existing datasets,
checkpoints, and manifests are not relocated by this source-code restructuring.

Generate provenance:

```bash
python -m grc_unlearning.provenance --config grc_unlearning/config/example_config.json
```

Evaluate an existing downstream checkpoint without retraining:

```bash
python -m grc_unlearning.evaluate_artifact --config grc_unlearning/config/example_config.json --device cuda
```

For different input locations, append `--dataset-root /path/to/ds_xray` and,
for evaluation, `--clean-checkpoint /path/to/checkpoint.pth`. Output locations
can be overridden with `--manifest-path` or `--results-dir`. Equivalent launchers
are `python scripts/generate_manifest.py` and `python scripts/evaluate_artifact.py`.

The original `Unified_Model` architecture and state-dict keys are preserved.
Stage 1 imports it from `medcoss.Downstream.Dim_2.Chest_XR.model.Unimodel`, loads
`checkpoint["model"]` strictly, and ignores optimizer/epoch state. Moving Python
source files does not require converting the seed-10 state dict.

The clean seed-10 baseline was previously reproduced with the following metrics;
these are the existing reference results, not a new run after restructuring:

| ACC | AUC | F1 |
| --- | --- | --- |
| 0.9864775725593667 | 0.9987385228932125 | 0.9844664073611358 |

See [Stage 1 documentation](grc_unlearning/README.md) for dataset modes, marker
geometry, selection details, and exact metric definitions.

## Baseline commands after the move

Inherited commands now start with `medcoss/`, for example
`bash medcoss/run_ds_stage.sh 2` or `bash medcoss/run_ssl_stage.sh 1`.
These are the original training workflows, separate from Stage 1 evaluation.
The scripts resolve their own baseline working directory; data/checkpoint/output
roots can be supplied through environment variables such as `DATA_ROOT`,
`DS_XRAY`, `PRETRAINED_CKPT`, `SNAPSHOT_ROOT`, `OUTPUT_ROOT`, and `LOG_ROOT`.

The direct ChestXR entry point is now
`medcoss/Downstream/Dim_2/Chest_XR/main.py`; pretraining Python entry points,
metric collectors, and the inherited `scripts/` tree likewise gained a
`medcoss/` prefix. Run standalone legacy collectors from `medcoss/` or pass
explicit input/output paths. Setup is `bash medcoss/setup_runpod.sh`, with an
optional `VENV_DIR` override; its default virtual environment remains at the
research repository root.

Historical absolute paths in the preserved RunPod notes and run-summary files
describe their original executions and have not been rewritten.

## Verification and attribution

```bash
python -m unittest discover -s grc_unlearning/tests -v
```

Tests use temporary synthetic images and small state-dict fixtures. The real
dataset parity test requires OpenCV. Rechecking the reference seed-10 metrics
requires the external dataset, checkpoint, and original MedCoSS runtime.

The original [MedCoSS README](medcoss/README.md), citation, acknowledgements, and
[license](medcoss/LICENSE) are retained for attribution. Inherited source was
moved with `git mv`; no baseline code was discarded.

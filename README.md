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
  grc_unlearning/       Stage 1 markers, provenance, dataset views, evaluation
    config/            Trackable example configuration
    tests/             Synthetic-fixture regression tests
  medcoss/             Inherited baseline dependency and launch scripts
    Downstream/Dim_2/Chest_XR/
    README.md          Original MedCoSS README and citation
    RUNPOD_SETUP.md    Preserved historical environment/reproduction notes
    LICENSE            Preserved inherited license
  blockchain/          Reserved for future integration
  policies/            Reserved for future GRC policy definitions
  manifests/           Generated provenance (ignored), documentation/templates
  results/             Generated metrics (ignored), documentation/templates
  scripts/             Research CLI launchers
  README.md
```

Datasets, checkpoints, and generated outputs are excluded from version control.
The `blockchain/` and `policies/` directories contain documentation only.

## Current implementation: Stage 1

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

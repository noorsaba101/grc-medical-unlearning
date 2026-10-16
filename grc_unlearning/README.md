# ChestXR proof of concept: Stages 1 and 2

Stage 1 creates provenance, overlays markers in memory, and evaluates an existing
classifier. Stage 2 fine-tunes the clean downstream classifier on the fixed marked
training split to test whether it acquires a pneumonia shortcut. Neither stage
performs machine unlearning or blockchain operations. The inherited model definition,
physical images, clean checkpoint, and Stage 1 manifest remain unchanged.

Use the existing MedCoSS Python 3.8 / PyTorch 1.11 environment and run commands
from the repository root. The inherited baseline lives in `medcoss/`; Stage 1
imports its ChestXR model and dataset through the `medcoss` package. Manifest
generation needs only the standard library.
Dataset/model evaluation uses the repository's existing dependencies (including
Pillow, OpenCV, torchvision, pandas, timm, einops, transformers, and scikit-learn).

## Generate provenance

```bash
cd /workspace/grc-medical-unlearning
python -m grc_unlearning.provenance --config grc_unlearning/config/example_config.json
```

The example config points to the existing seed-10 clean downstream checkpoint.
Generation writes only the manifest; it does not load that checkpoint. Settings
can be overridden, for example:

```bash
python -m grc_unlearning.provenance --config grc_unlearning/config/example_config.json \
  --marker-fraction 0.75 --seed 10 --marker-type cross \
  --marker-size 12 --marker-margin 4 \
  --manifest-path manifests/chestxr_seed_10.jsonl
```

The scanner mirrors `ChestXR_Dataset`: filenames are sorted within each class,
the first `floor(0.8 * class_count)` images in `train/` form the training split,
and the remainder form internal validation (`split="val"`). Every image in
`train/` gets a record. Only **training pneumonia** samples are eligible; internal
validation and the separate `validation/` test folder stay unmarked by default.
For the documented 1,076 pneumonia files, 860 are training samples and 645 are
selected at fraction 0.75. The 12,121 training-folder files yield 9,695 training
and 2,426 internal-validation records with the documented per-class counts.

Selection takes `floor(marker_fraction * eligible_count)` samples ranked by
SHA-256 of `seed:relative_path`. It is reproducible across machines and Python
versions without touching global random state. Stable sample IDs hash the
dataset-relative POSIX path; they identify paths, not image-content versions.
JSON stores metadata plus `records`; JSONL stores one record per line. Records
contain the required identity, label, split, selection seed, and marker fields,
plus marker size and margin. Manifest output must be outside `dataset_root`.
Paths in the config or CLI are relative to the current working directory unless
absolute. Unexpected non-image class-directory entries raise an error so the
legacy split cannot silently change.

## Use an in-memory marked dataset

```python
from grc_unlearning.marked_dataset import MarkedChestXRDataset

dataset = MarkedChestXRDataset.from_root(
    "/workspace/grc-medical-unlearning/data/ds_xray",
    manifest_path="manifests/chestxr_seed_10.json",
    split="train",
    crop_size=(224, 224),
    marker_mode="manifest",
)
image_tensor, label = dataset[0]
```

You can also pass an existing `ChestXR_Dataset` to the wrapper constructor.
`marker_mode="force_on"` marks every sample using constructor settings;
`"force_off"` delegates to the original dataset. These force modes can omit the
manifest. Manifest mode requires complete train/val coverage and matching labels;
test paths absent from the manifest are clean. Labels retain the original
one-element floating-point tensor format and covid=0, normal=1, pneumonia=2.

`synthetic_marker.apply_marker(image, marker_type="square", size=12, margin=4)`
returns a new PIL image with a black/white square or cross in the upper-right
corner. The default `marker_placement="native"` interprets size and margin as
source-image pixels, before the original transforms. Resizing can shrink the
marker; training augmentation can blur or flip it. The opt-in
`marker_placement="post_augmentation"` runs every original PIL operation first
and draws immediately before the original final `ToTensor`, making size and
margin model-input pixels. It rejects unsupported transform pipelines rather
than silently moving the marker. Test preprocessing remains deterministic.

## Evaluate an existing downstream checkpoint

```bash
python -m grc_unlearning.evaluate_artifact --config grc_unlearning/config/example_config.json --device cuda
```

This strictly loads `checkpoint["model"]` into the original `Unified_Model`,
ignoring optimizer/epoch state, and evaluates `split="test"` (the 3,032 files in
`validation/`). It accepts a uniform `module.` state-dict prefix. A scoped import
alias handles the model's legacy absolute import without editing MedCoSS.

The configured external dataset and seed-10 checkpoint locations are unchanged
by the repository move. Override them with `--dataset-root` and
`--clean-checkpoint` or edit your config. Outputs now default to `manifests/`
and `results/`; existing manifests remain usable through `--manifest-path`.
The downstream checkpoint state-dict keys and model architecture are unchanged.

Every test image is evaluated clean and with a forced marker, including covid
and normal images. The training manifest is intentionally not used to filter
these counterfactuals. Output is `results_dir/artifact_metrics.json`:

- `clean_accuracy` and `marked_accuracy`: overall classification accuracy.
- `mean_absolute_probability_change`: mean `abs(marked - clean)` over all
  samples and all three class probabilities.
- `mean_pneumonia_probability_change`: mean signed `P_marked(2) - P_clean(2)`.
- `prediction_flip_rate`: fraction whose predicted class changes.

The evaluator uses evaluation mode and disables gradients. `load_checkpoint_weights`
and `load_chestxr_model` are separate from `evaluate_paired`; the latter accepts a
custom probability predictor for other models. These measurements assess marker
sensitivity, not evidence that the clean model learned the synthetic artifact.

## Verification

```bash
python -m unittest discover -s grc_unlearning/tests -v
```

Tests use temporary synthetic images and small models. An additional real-loader
parity test runs when OpenCV is available. Full seed-10 checkpoint evaluation
requires the dataset, checkpoint, and MedCoSS runtime on the configured host.

## Stage 2: controlled shortcut-bias fine-tuning

Run from the repository root on RunPod:

```bash
python -m grc_unlearning.train_biased --config grc_unlearning/config/biased_config.json --device cuda
```

The example config uses the existing clean seed-10 `checkpoint["model"]` and
`manifests/chestxr_seed_10.json`. It does not load SSL weights or restore the
clean checkpoint's optimizer/epoch state. It creates a new AdamW optimizer and
fine-tunes all model parameters for **3 epochs**, starting at **1e-5**, batch size
**32**, seed **10**, FP32. Weight decay is 1e-4 and gradient clipping is 12, as
in the baseline. The per-epoch polynomial learning-rate decay uses power 0.9.

The manifest must match the exact original train/val paths and labels. The
example config checks 12,121 records, 9,695 training samples, 2,426 internal
validation samples, 1,076 pneumonia images, and 645 marked training samples.
Any marked internal-validation or non-pneumonia record is rejected. No manifest
is regenerated, and no source images are rewritten. By default, marker geometry
is read from the manifest. Explicit rendering overrides change geometry only,
never sample membership, and are reused for paired evaluation, including
standalone evaluation with the saved `config_used.json`.

The original `biased_config.json` reuses the existing transforms, with markers
drawn before augmentation.
Color jitter, blur, and horizontal flipping remain active, so the effective
marker may shrink, blur, or move. Labels, sample frequencies, and the model's
existing cross-entropy loss are preserved. The final partial batch is retained
so every training sample is used once per epoch. Python, NumPy, torch, shuffle,
and DataLoader worker RNGs are seeded; there is no mixed-precision training.

Clean internal-validation accuracy is measured before training and after each
epoch. `checkpoint.pth` is the best **fine-tuned** epoch by clean validation
accuracy, choosing the latest on ties. Test artifact metrics never select the
checkpoint. The clean source and saved biased checkpoint are evaluated on the
same paired test images with the same settings. The default evaluation batch
size is **1**, matching the original MedCoSS test loop.

Parameters can be overridden without editing the manifest:

```bash
python -m grc_unlearning.train_biased --config grc_unlearning/config/biased_config.json \
  --device cuda --learning-rate 0.00001 --epochs 3 --batch-size 32 --seed 10 \
  --output-dir /workspace/grc-medical-unlearning/checkpoints/biased/seed_10
```

`--evaluation-batch-size`, `--num-workers`, `--dataset-root`, `--clean-checkpoint`,
and `--manifest-path` are also supported. Training seed and the manifest's
selection seed are recorded separately; changing the training seed does not
select new marked samples. A different dataset requires a config with matching
`expected_statistics` and `expected_test_samples`, or omission of those checks.

Default outputs, ignored by Git:

```text
/workspace/grc-medical-unlearning/checkpoints/biased/seed_10/
  checkpoint.pth                model, fresh optimizer state, epoch, provenance
  config_used.json              effective settings, manifest hash, runtime
  effective_provenance.json     unchanged membership plus effective geometry
  training_history.json         loss and clean validation metrics per epoch
  clean_artifact_metrics.json   clean source remeasured with matching settings
  artifact_metrics.json         saved biased checkpoint's paired test metrics
  comparison.json               matched-baseline deltas and clean-accuracy guard
```

The output directory must be outside the dataset and clean-checkpoint directory.
Existing run artifacts are never silently overwritten: choose a new
`--output-dir` for another run. A positive sensitivity result is measured, not
guaranteed by fine-tuning. `comparison.json` explicitly reports whether mean
absolute probability change, pneumonia probability change, and flip rate
increased, alongside the change in clean accuracy. The already strong clean
classifier may initially ignore the marker; retain negative results as well.

For a weak induced shortcut, run the full image-pipeline audit and inspect exact
model-input pairs before changing training. See [Stage 2 diagnosis and stronger
configuration](STAGE2_DIAGNOSIS.md). The audit does not train a model or change
the manifest; it supports both actual placement modes and an additional preview.

Evaluate the saved biased checkpoint independently:

```bash
python -m grc_unlearning.evaluate_artifact \
  --config /workspace/grc-medical-unlearning/checkpoints/biased/seed_10/config_used.json \
  --clean-checkpoint /workspace/grc-medical-unlearning/checkpoints/biased/seed_10/checkpoint.pth \
  --batch-size 1 --device cuda \
  --results-dir /workspace/grc-medical-unlearning/checkpoints/biased/seed_10/paired_evaluation
```

Here `--clean-checkpoint` is the existing evaluator's checkpoint-path option;
it can point to the biased model. Independent metrics go to
`paired_evaluation/artifact_metrics.json`.

## Stage 2 strong run: fixed 32x32 post-augmentation marker

This separate experiment starts again from the **original clean seed-10
checkpoint**, not the weak Stage 2 checkpoint. Run on RunPod:

```bash
cd /workspace/grc-medical-unlearning
python -m grc_unlearning.train_biased \
  --config grc_unlearning/config/biased_strong_config.json --device cuda
```

The strong config uses a 32x32 black/white square, margin 8, in the upper-right
corner of the 224x224 input. The selected images pass through the original
Resize, jitter, grayscale, blur, and flip **before** marker drawing; `ToTensor`
follows immediately. The marker no longer shrinks, moves, or gets blurred.
The underlying radiograph retains all original augmentations. In exclusive
pixel coordinates its box is `[184, 8, 216, 40]`. The number of changed pixels
can be below 1,024 when source pixels already equal the drawn values.

Training uses **5 epochs**, initial LR **2e-5**, batch size **32**, seed **10**,
and a fresh AdamW optimizer. Internal validation is entirely clean. It reads
the unchanged Stage 1 manifest for the same **645 selected pneumonia training
IDs** (fraction **0.75**); there is no reselection. Startup enforces the expected
split/selection counts. The run-local `effective_provenance.json` records every
original ID, label, split, and marker flag alongside the effective geometry,
parent manifest SHA-256, and SHA-256 of the sorted selected IDs. It does not
replace or edit `manifests/chestxr_seed_10.json`.

Before any optimizer update, the trainer measures the original clean checkpoint
on the separate 3,032-image paired test set with this same 32x32 marker. After
training it evaluates the saved biased checkpoint through the **same paired
dataset**, preprocessing, geometry, and evaluation batch size **1**. It compares
all five artifact metrics against that newly measured source baseline. The old
12x12 reference is excluded when marker settings differ, even if supplied in a
config. A clean-test-accuracy loss greater than **0.01 absolute (one percentage
point)** emits a warning and is recorded in `comparison.json`; this warning
neither removes the checkpoint nor uses test performance for checkpoint selection.
Selection remains best clean internal-validation accuracy, latest epoch on ties.

All seven run artifacts listed above are saved under the new ignored directory:

```text
/workspace/grc-medical-unlearning/checkpoints/biased/seed_10_strong/
```

`clean_artifact_metrics.json` is the matching clean baseline; `artifact_metrics.json`
is the biased result; `comparison.json` contains their metric deltas and guard.
The trainer already runs the clean baseline. To measure it separately without
training (the explicit evaluation batch size matters):

```bash
python -m grc_unlearning.evaluate_artifact \
  --config grc_unlearning/config/biased_strong_config.json \
  --batch-size 1 --device cuda \
  --results-dir results/clean_seed_10_strong_marker
```

To evaluate the saved strong checkpoint independently:

```bash
python -m grc_unlearning.evaluate_artifact \
  --config /workspace/grc-medical-unlearning/checkpoints/biased/seed_10_strong/config_used.json \
  --clean-checkpoint /workspace/grc-medical-unlearning/checkpoints/biased/seed_10_strong/checkpoint.pth \
  --batch-size 1 --device cuda --results-dir results/biased_seed_10_strong
```

Optional CPU coverage/geometry audit before spending GPU time:

```bash
python -m grc_unlearning.diagnose_marker \
  --config grc_unlearning/config/biased_strong_config.json --num-samples 5 \
  --output-dir results/diagnostics/stage2_seed_10_strong
```

This config is a controlled shortcut-induction attempt, not a guarantee of
learning. No unlearning or blockchain functionality is included.

## Investigating the one-prediction accuracy difference

The original ACC is 2,991/3,032; the supplied Stage 1 clean ACC is 2,990/3,032.
Both implementations reuse the same test class: OpenCV grayscale decoding,
PIL RGB conversion, `Resize((224, 224))`, then `ToTensor()`. There are no random
test augmentations or normalization differences in these code paths.

The original loop evaluates one image at a time with `cudnn.benchmark=True`.
Stage 1 defaults to batch size 16, `benchmark=False`, and deterministic cuDNN.
Different batch shapes and backend algorithms can perturb probabilities near
a decision boundary. Also, MedCoSS's reported accuracy takes argmax of logits,
whereas Stage 1 takes argmax of returned softmax probabilities. These usually
agree, but a rare float32 softmax rounding tie can change argmax. Tests demonstrate
that mechanism without changing either accuracy definition. These are possible
explanations, not a confirmed cause without the original data, checkpoint, and
runtime. Deterministic preprocessing
does not imply bit-identical predictions across inference batch sizes, TF32
settings, CUDA versions, or library versions.

The evaluator now records runtime/backend settings and offers an optional audit
that holds weights, preprocessing, and backend flags fixed while comparing
batch sizes and listing disagreeing sample paths, probabilities, and margins.
For the ChestXR model it also captures final-head logits with a temporary hook
and reports any logit-versus-softmax argmax disagreements:

```bash
python -m grc_unlearning.evaluate_artifact --config grc_unlearning/config/example_config.json \
  --device cuda --batch-size 1 --audit-batch-size 16 \
  --results-dir results/clean_seed_10_batch_audit
```

Inspect `batching_audit.json`. If it reproduces the one-prediction difference,
batching explains that difference on the current runtime. Otherwise the
historical cause remains unresolved and needs comparison against the original
runtime or per-image predictions. Accuracy is never rounded or adjusted to
force equality. Stage 1's default batch size and preprocessing are unchanged.

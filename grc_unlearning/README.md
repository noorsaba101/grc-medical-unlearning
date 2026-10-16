# Stage 1: synthetic artifact provenance and evaluation

This package prepares a selective machine-unlearning experiment. It creates
provenance, overlays markers in memory, and evaluates an existing classifier.
It contains no training, unlearning, or blockchain code. MedCoSS source files,
checkpoint files, and dataset images are unchanged.

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
corner. Size and margin are source-image pixels, before the original transforms.
Resizing can shrink the marker; training augmentation can blur or flip it. Test
preprocessing remains the original deterministic resize and tensor conversion.

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

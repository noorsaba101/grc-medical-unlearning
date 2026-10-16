# Stage 2 diagnosis and proposed stronger experiment

The reported run increased mean absolute probability change by only
0.00044209 (about 4.35%). Prediction flips increased from 42 to 43 out of
3,032 images. Clean accuracy was unchanged, while marked accuracy improved
by four correct predictions. This does not yet demonstrate a strong unwanted
pneumonia shortcut.

## What the existing pipeline actually does

`MarkedChestXRDataset.__getitem__` loads the selected image through
`load_chestxr_image`: OpenCV reads grayscale, then PIL converts it to RGB.
`apply_marker` draws the black/white 12-pixel marker on that **native-resolution
RGB image**, before calling the original MedCoSS training transform:

```text
OpenCV grayscale -> PIL RGB -> native 12x12 marker -> Resize(224,224)
  -> ColorJitter with probability 0.8
  -> RandomGrayscale with probability 0.2
  -> GaussianBlur(sigma 0.1-2.0) with probability 0.5
  -> RandomHorizontalFlip with probability 0.5
  -> ToTensor
```

Consequently, 12 source pixels are not necessarily 12 model-input pixels.
For a source of width W and height H, the geometric footprint after resizing is
`(12 * 224 / W, 12 * 224 / H)`. Square-image examples:

| Source side | Geometric marker side at input |
| --- | --- |
| 224 | 12 pixels |
| 299 | about 8.99 pixels |
| 512 | 5.25 pixels |
| 1024 | 2.625 pixels |

These examples do not assert the actual dataset's resolutions. The diagnostic
records them from the decoded images. Resampling can add a small halo, so the
exact changed-pixel box need not equal the geometric extent. Jitter and blur
can weaken the marker; flipping moves it to the left on roughly half of training
draws. Test preprocessing has only Resize/ToTensor and leaves it on the right.
The ViT uses 16-pixel patches, making a downscaled small corner cue less prominent.

Only 645/9,695 = 6.65% of all training samples are marked. The 0.75 fraction is
conditional on the 860 training pneumonia samples, not on the entire dataset.
The already accurate clean model can solve most examples without the cue, so
ordinary cross-entropy supplies little pressure to learn it. Three epochs and
the decaying 1e-5 learning rate further limit change. These are mechanisms worth
checking, not proof of a particular gradient history. Inspect `training_history`
and its `best_epoch`: the saved model maximizes clean validation accuracy and
may come from an earlier epoch.

## Run the actual image and coverage audit

```bash
cd /workspace/grc-medical-unlearning
python -m grc_unlearning.diagnose_marker \
  --config grc_unlearning/config/biased_config.json \
  --num-samples 5 \
  --output-dir results/diagnostics/stage2_seed_10
```

This CPU utility loads no model/checkpoint and performs no training. By default
it executes the real Stage 2 wrapper for **all training and internal-validation
samples**. The configured expected counts must match. It observes actual calls
to the real overlay function, verifies geometry/labels, and reports whether all
645 selected samples were altered natively, whether their changes survive the
complete training transform, and whether any unselected/validation sample was
marked or changed. It does not infer coverage solely from manifest booleans.

For each pair it replays identical Python, NumPy, and CPU torch random states,
including the original blur sampler. It restores both the RNG states and
temporary observers afterward. Thus differences are caused by the marker,
rather than unrelated random augmentation draws. Each sample uses a stable
path-derived diagnostic seed; these are representative reproducible draws,
not reconstructions of every draw from the completed training job.

Outputs are ignored by Git:

```text
results/diagnostics/stage2_seed_10/
  summary.json
  per_sample_audit.jsonl
  sample_<id>/                 five selected pneumonia samples
    clean_model_input.png
    marked_model_input.png
    absolute_difference.png
    absolute_difference_scaled.png
    clean_resize_only.png
    marked_resize_only.png
    absolute_difference_resize_only.png
    proposed_input_marker_preview.png
    proposed_input_marker_difference.png
    model_inputs.pt
    metadata.json
```

`model_inputs.pt` retains exact clean, marked, difference, and proposed-preview
tensors. PNGs preserve the existing byte-domain inputs. The scaled difference
image is only a visibility aid; its divisor is recorded. Metadata records
relative path, label, source dimensions, overlay settings/calls, replay seed,
geometric resized extent, and actual difference statistics.

Changed pixels mean spatial pixels with **any** changed RGB channel. Bounding
boxes use `[left, top, right_exclusive, bottom_exclusive]`. Exact counts and
additional 2/255 and 8/255 difference thresholds are reported. ColorJitter's
contrast operation depends on global image statistics, so even matched draws
can spread a marker's influence across the whole frame. The full-transform
difference box is therefore not always a literal marker footprint. Resize-only
differences isolate the marker before jitter, blur, and flipping.

Check `all_selected_marker_applications_verified`,
`all_selected_visible_after_training_transform`, `counts`, `failures`, source
resolution counts, and the distributions of changed pixels and bounding boxes.
`--samples-only` is an optional quick preview and explicitly does **not** certify
coverage of all 645 samples. Choose a fresh output directory for another audit.

The source dataset and manifest are not available in the local development
checkout. Full empirical counts and dimensions must come from this RunPod audit;
local tests exercise the same logic on synthetic fixtures.

## Proposed next RunPod configuration

First change marker presentation, while retaining the same sample identities,
class labels, model architecture, and classification loss. The proposed setup is:

| Setting | Proposal |
| --- | --- |
| Initialization | Original clean seed-10 downstream checkpoint; fresh AdamW |
| Marker placement | After Resize **and all other PIL augmentations**, immediately before ToTensor |
| Marker geometry | Black/white square, 32x32 input pixels, margin 8, fixed upper-right |
| Marked fraction | Keep 0.75: the same 645 pneumonia training sample IDs |
| Training | 5 epochs, initial LR 2e-5, batch size 32, seed 10 |
| Other optimizer settings | Keep weight decay 1e-4, clipping 12, polynomial decay 0.9 |
| Validation | Unmarked; track clean accuracy and a predeclared acceptable drop |
| Test evaluation | Batch size 1; identical input-space marker on both clean/biased models |
| Separate run output | `checkpoints/biased/seed_10_input32_lr_0.00002_ep5/` |
| Separate provenance | `manifests/chestxr_stage2_input32_seed_10.json`, derived from the unchanged Stage 1 manifest |

A 32-pixel corner square occupies about 2.04% of the 224x224 frame, spans multiple
ViT patches, and is explicit enough for this proof of concept. Applying it only
immediately after Resize would still let subsequent blur/jitter/flip weaken or
move it; placing it after those augmentations removes that ambiguity while
preserving augmentation of the underlying radiograph. The modest LR increase
and two extra epochs provide more opportunity to learn the cue without
retraining from scratch. Keep the selected fraction fixed initially to avoid
changing both provenance membership and marker presentation.

The utility already renders this placement as a **preview only**. The proposed
placement and training configuration have not been applied to the trainer in
this diagnostic change. Simply setting `marker_size=32` in the current trainer
would still draw at native resolution; it would not implement this proposal.
A future opt-in placement mode must be shared by training and paired evaluation,
and the separate Stage 2 manifest/config must record its size, placement, parent
manifest hash, and unchanged selection IDs.

Remeasure the clean checkpoint with the **same new 32-pixel input-space marker**
before judging learning. Comparing a biased model with a larger marker against
the old clean baseline's smaller marker confounds marker strength with learned
bias. Require a substantial increase versus that matched clean baseline, not
merely a positive delta. Also inspect pneumonia-probability shifts and flips
**toward pneumonia on true covid/normal images**; an aggregate absolute change
alone can reflect occlusion or robustness effects. A practical predeclared
clean-validation guard is no more than a one-percentage-point drop, with clean
test performance reported separately; do not select checkpoints using test-set
artifact scores.

If this visible, consistent cue still fails to induce a meaningful shortcut,
the next separate experiment can use fraction 1.0: all 860 training pneumonia
images marked, with no validation markers. Generate a separate Stage 2 manifest
and config for that experiment; never overwrite Stage 1 provenance.

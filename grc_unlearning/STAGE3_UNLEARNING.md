# Stage 3: approximate synthetic-marker concept unlearning

This stage suppresses a controlled synthetic shortcut while retaining clean
diagnostic predictions. It is **approximate concept unlearning**, not exact
training-data deletion, certified forgetting, or a clinical validation claim.
This trainer performs no blockchain operations. The separate governance demo
described in `blockchain/README.md` authorizes/audits acceptance of its outputs.

## Run on RunPod

Use the existing Python 3.8 / PyTorch 1.11 MedCoSS environment:

```bash
cd /workspace/grc-medical-unlearning
python -m grc_unlearning.unlearn_marker \
  --config grc_unlearning/config/unlearning_config.json --device cuda
```

The config initializes the student strictly from the selected biased
`checkpoint["model"]` at
`/workspace/grc-medical-unlearning/checkpoints/biased/seed_10_strong_epoch_selection/checkpoint.pth`.
An independent frozen copy of that initialized student is the **biased teacher**.
The original clean checkpoint is recorded as a reference path only; it is not
loaded for initialization or distillation. Source optimizer state is ignored.

Defaults: 3 epochs, constant LR 1e-5, batch size 32, seed 10, fresh AdamW with
weight decay 1e-4, FP32, and gradient clipping 12. Both loss multipliers default
to 1.0. Override them without editing source data, for example:

```bash
python -m grc_unlearning.unlearn_marker \
  --config grc_unlearning/config/unlearning_config.json --device cuda \
  --lambda-invariance 1.0 --lambda-retain 1.0 \
  --output-dir /workspace/grc-medical-unlearning/checkpoints/unlearned/seed_10_repeat
```

An existing run's artifact filenames cannot be overwritten. Both source
checkpoint directories and the dataset directory are protected output targets.
All inputs/outputs remain configurable through JSON or CLI paths.

## Training images and paired views

The existing Stage 2 split checker validates the same immutable Stage 1 manifest,
including 9,695 training samples, 2,426 internal-validation samples, and the same
645 selected pneumonia IDs (0.75 fraction, selection seed 10). It records the
source manifest SHA-256 and selected-ID digest; it never reselects samples.

For **every training image in every class**, the new paired-training dataset:

1. Uses original OpenCV grayscale decoding and PIL RGB conversion.
2. Executes every original MedCoSS PIL training transform **once**, including
   Resize, jitter, grayscale, blur, and horizontal flip.
3. Makes a clean view and a marked copy of that same augmented PIL image.
4. Draws the forbidden 32x32 square with margin 8 in the upper-right corner,
   then runs the original final `ToTensor` on each view.

The marker box in 224-input exclusive coordinates is `[184, 8, 216, 40]`.
Pairs differ only inside that box; individual pixels can already match marker
values. Stage 3 intentionally uses counterfactual markers across all classes,
not just the 645 historical exposure samples. **This does not change provenance
flags or duplicate/rewrite any image.** Every original training sample is used
once per epoch, including the final partial batch. Internal validation remains
the original clean deterministic dataset and is never used for gradient updates.

## Exact loss

Let `z_c`, `z_m` be student logits for clean/marked views and `z_t` frozen biased
teacher logits for the same clean view. `p_c`, `p_m`, `p_t` are their softmax
probabilities at temperature 1. All losses are batch means:

```text
L_task       = cross_entropy(z_c, true_label)
L_invariance = 0.5 * (KL(p_c || p_m) + KL(p_m || p_c))
L_retain     = KL(stop_gradient(p_t) || p_c)
L_total      = L_task + lambda_invariance * L_invariance + lambda_retain * L_retain
```

Implementation uses `log_softmax` for stability. Symmetric KL is evaluated as
`0.5 * mean(sum((p_c - p_m) * (log_p_c - log_p_m)))`, with gradients through
**both** student views. Retention uses `kl_div(log_p_c, log_p_t, log_target=True,
reduction="batchmean")`. Teacher targets are detached and computed under
`no_grad`; teacher parameters have `requires_grad=False`. The teacher never
sees marked inputs. No marked-image task CE, entropy-maximization, or negative
gradient/ascent term is added. Setting both multipliers to zero is only a
clean-CE ablation, not evidence of marker unlearning.

A temporary final-`head` hook obtains the original model's logits without
changing MedCoSS. The hook and legacy `cal_acc` flag are restored after each
forward, including failures. Student forwards use **eval mode with gradients
enabled**: this intentionally disables stochastic depth/dropout so invariance
does not penalize different random model masks. PIL augmentations remain active.
All student parameters remain trainable; teacher parameters never update.

## Evaluation and checkpoint selection

Before updates, remeasure the biased model on the original 3,032-image paired
evaluation set. After every epoch, measure clean internal-validation accuracy
and run the existing paired evaluator with the identical 32x32 post-augmentation
geometry, deterministic test preprocessing, ordering, and batch size 1. Added
evaluations preserve Python/NumPy/CPU/CUDA RNG states.

Each epoch records clean/marked accuracy, mean absolute probability change,
signed pneumonia-probability change, prediction flip rate, flip-into-pneumonia
rate/count, and all existing true-label class-conditional metrics for
non-pneumonia, normal, covid, and pneumonia.

The clean-accuracy guard is measured against the **remeasured biased checkpoint**,
not the original clean model: `biased_clean_accuracy - candidate_clean_accuracy
<= 0.01`. Among safe candidates, minimize this predeclared tuple:

```text
(max(abs(overall signed pneumonia response), abs(non-pneumonia signed response)),
 overall flip-into-pneumonia rate,
 non-pneumonia flip-into-pneumonia rate,
 -clean accuracy,
 epoch)
```

The first criterion drives response toward zero without rewarding a large
negative/inverted shortcut or cancellation between pneumonia and non-pneumonia
groups. Flips are the second priority, clean accuracy the third; exact ties
prefer earlier epochs. This is a ranking, not a claim every metric must improve
simultaneously: comparison flags report each outcome separately.

Epoch 0 (the unchanged biased initialization) is a candidate. An updated epoch
must beat its ranking within the accuracy guard; otherwise `checkpoint.pth`
contains an explicit `unchanged_biased_fallback`, selection status is
`biased_initialization_fallback`, and a warning says this is **not successful
unlearning**. A post-update winner has role `unlearned`; that role identifies an
experimental artifact, not a forgetting certificate. The comparison separately
reports whether response magnitude **and** flip rate decreased within the guard.

These paired test-folder metrics already drove Stage 2 selection and now also
drive Stage 3 selection. Outputs flag this split as **experimental tuning, not
an untouched holdout**. Unbiased final evaluation requires new held-out data.

## Output files

Default ignored directory:
`/workspace/grc-medical-unlearning/checkpoints/unlearned/seed_10/`

```text
checkpoint.pth                   selected student (or explicit epoch-0 fallback)
config_used.json                 settings, loss/mode policy, split counts, input hashes
training_history.json            component losses, validation, paired metrics per epoch
before_unlearning_metrics.json   remeasured biased checkpoint
after_unlearning_metrics.json    selected saved checkpoint's epoch metrics
comparison.json                 overall/class deltas and separate outcome flags
selection.json                  all candidate scores, guards, and selected epoch/status
```

Checkpoint keys include `model`, the **new** optimizer state, `epoch`, `stage`,
`checkpoint_role`, source checkpoint/manifest/selection hashes, marker settings,
and selected epoch metrics. The file remains loadable with the existing strict
ChestXR model loader. No original dataset, manifest, biased checkpoint, or clean
checkpoint is overwritten.

## Local verification only

```bash
python -m unittest discover -s grc_unlearning/tests -v
```

Tests use temporary synthetic images and tiny CPU models to verify pairing,
loss algebra/gradients, frozen teacher, fresh optimizer, selection and fallback,
file protection, and saved-checkpoint loading. No full unlearning experiment
or real dataset/checkpoint evaluation is run locally.

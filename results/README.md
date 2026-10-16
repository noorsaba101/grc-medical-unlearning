# Evaluation results

Stage 1 writes paired clean/marked metrics to `clean_seed_10/artifact_metrics.json`
here by default. Generated outputs are ignored by Git. Documentation and templates
under `examples/` remain trackable; no checkpoint files belong in this directory.

From the repository root:

```bash
python -m grc_unlearning.evaluate_artifact --config grc_unlearning/config/example_config.json
```

Use `--results-dir` to choose another output location. Historical baseline run
summaries remain under `medcoss/` with their original recorded paths intact.

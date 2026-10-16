# Experiment provenance manifests

Stage 1 writes generated JSON/JSONL manifests here by default. Generated records
are ignored by Git. Documentation and explicitly shareable templates under
`examples/` remain trackable. Do not add real medical data or identifiers here.

From the repository root:

```bash
python -m grc_unlearning.provenance --config grc_unlearning/config/example_config.json
```

Use `--manifest-path` to reuse an existing manifest or choose another destination.

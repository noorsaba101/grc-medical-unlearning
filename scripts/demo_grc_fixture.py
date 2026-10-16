"""Optional local smoke demo with clearly synthetic artifacts, never real ML results."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grc_unlearning.grc_pipeline import run_demo
from grc_unlearning.tests.test_governance import create_fixture


def main():
    parser = argparse.ArgumentParser(description="SYNTHETIC governance smoke demo; not real Stage 3 verification.")
    parser.add_argument("--backend", choices=("mock", "evm"), default="mock")
    parser.add_argument("--output-dir", type=Path, default=Path("results/grc_synthetic_fixture"))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("choose a fresh --output-dir; synthetic fixtures are never overwritten")
    config = create_fixture(args.output_dir.resolve() / "synthetic_inputs")
    config["results_dir"] = str(args.output_dir.resolve() / "audit")
    result = run_demo(config, backend=args.backend)
    result["notice"] = "SYNTHETIC smoke-test artifacts ONLY; these are not RunPod model weights or measurements"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

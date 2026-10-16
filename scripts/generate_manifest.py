"""Run Stage 1 provenance generation from a source checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grc_unlearning.provenance import main


if __name__ == "__main__":
    main()

"""Write ``tests/fixtures/synthetic_sample.parquet`` from the versioned generator.

Usage::

    python scripts/make_synthetic_fixture.py [--rows N]

The file is fully determined by the generator code, the mapping and the seed
in ``configs/config.yaml``; regenerate it whenever any of those change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config
from src.data.synthetic import DEFAULT_ROWS, make_synthetic_frame
from src.paths import PROJECT_ROOT

FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "synthetic_sample.parquet"


def main(argv: list[str] | None = None) -> int:
    """Generate and write the fixture."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    args = parser.parse_args(argv)
    frame = make_synthetic_frame(seed=get_config().project.seed, rows=args.rows)
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(FIXTURE_PATH, index=False)
    print(f"wrote {FIXTURE_PATH.relative_to(PROJECT_ROOT)} with shape {frame.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

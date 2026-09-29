"""Baseline drift scenario: holdout window without shifts (expects no alert).

Usage (repository root)::

    python scripts/baseline_scenario.py [--fixture] [--api-url http://localhost:8000]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitoring.scenarios import main

if __name__ == "__main__":
    sys.exit(main(default_scenario="baseline"))

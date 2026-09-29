"""Load test of the prediction API (p95, throughput, CPU and memory).

Usage (repository root, API running)::

    python scripts/load_test.py --url http://localhost:8000 [--requests 500] [--concurrency 8]
        [--pid <uvicorn pid> | --container hypertension-mlops-api-1]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitoring.load_test import main

if __name__ == "__main__":
    sys.exit(main())

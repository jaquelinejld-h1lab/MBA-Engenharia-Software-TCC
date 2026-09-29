"""Online/offline parity with the promoted model on 50 real test rows (skips without artifacts)."""

from __future__ import annotations

import numpy as np
import pytest

from src.api.model_loader import load_local
from src.api.service import PredictionService
from src.config import Settings
from src.data.filters import TARGET_COLUMN, split_holdout
from src.data.pipeline import load_development_frame
from src.paths import resolve
from tests.api_fixtures import payload_from_row

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_promoted_model_parity_on_real_rows(cfg: Settings) -> None:
    run_dir = resolve(cfg.api.local_run_dir)
    has_model = (run_dir / "pipeline.joblib").is_file()
    if not has_model or not cfg.paths.absolute("raw_dataset").is_file():
        pytest.skip("promoted artifacts or raw dataset not available")
    loaded = load_local(run_dir, cfg.tracking.registered_model_name)
    development = load_development_frame(cfg, materialize=False)
    _, test = split_holdout(development, cfg.split, cfg.project.seed)
    rows = test.drop(columns=[TARGET_COLUMN]).head(50)
    service = PredictionService(cfg, loaded)
    offline = loaded.pipeline.predict_proba(rows)[:, 1]
    online = [
        p.probability
        for p in service.predict_records([payload_from_row(r) for _, r in rows.iterrows()])
    ]
    assert float(np.max(np.abs(np.asarray(online) - offline))) < 1e-9
    # the registry version grows with each promotion; only its shape and the stage are fixed
    assert loaded.version.isdigit() and loaded.stage == cfg.tracking.promote_stage

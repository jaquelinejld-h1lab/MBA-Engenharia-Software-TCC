"""Evaluate the retraining flag and, when it recommends, train the model again.

Two steps, on purpose separable:

1. **decide**: compute drift of a window against the training reference and write
   ``evidencias/retraining_flag.json`` plus its Markdown evidence;
2. **act**: run the training pipeline, but only when the flag recommends it or when the
   operator overrides with ``--force``.

``--check-only`` stops after step 1, which is what a scheduled job or the CI runs: it
reports and never replaces a model on its own. The window comes from a drift scenario
fixture or from a CSV of recent observations, because the API keeps its window in memory
and a process that just started has none.

Usage::

    python scripts/retrain.py --check-only            # decide and report
    python scripts/retrain.py                         # decide and train if recommended
    python scripts/retrain.py --force                 # train regardless of the flag
    python scripts/retrain.py --window recent.csv     # decide over observed records
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings, load_config
from src.data.filters import TARGET_COLUMN, split_holdout
from src.data.mapping import load_mapping
from src.data.pipeline import load_development_frame
from src.logging_setup import configure_logging, get_logger
from src.models.train import _main as train_main
from src.monitoring.drift import DriftReport, build_reference, compute_drift, load_reference
from src.monitoring.retraining import (
    FLAG_FILENAME,
    decide,
    report_markdown,
    save,
)
from src.monitoring.retraining import load as load_decision

logger = get_logger(__name__)


def _reference_path(cfg: Settings) -> Path:
    """Drift reference persisted next to the model in service."""
    return (
        cfg.paths.absolute("artifacts_dir")
        / cfg.api.local_run_dir.name
        / (cfg.monitoring.reference_file)
    )


def _window(cfg: Settings, csv_path: Path | None) -> pd.DataFrame:
    """Observations to compare against the reference.

    A CSV of recent requests when given; otherwise the holdout split, which stands for
    "production data that the model did not train on" in a local study.
    """
    if csv_path is not None:
        return pd.read_csv(csv_path)
    development = load_development_frame(cfg, materialize=False)
    frame = development[0] if isinstance(development, tuple) else development
    _, holdout = split_holdout(frame, cfg.split, cfg.project.seed)
    return holdout.drop(columns=[TARGET_COLUMN])


def evaluate(cfg: Settings, csv_path: Path | None) -> tuple[DriftReport | None, Path]:
    """Compute drift of the window against the reference and persist the flag."""
    mapping = load_mapping()
    reference_path = _reference_path(cfg)
    window = _window(cfg, csv_path)
    if reference_path.is_file():
        reference = load_reference(reference_path)
    else:
        logger.warning(
            "drift reference not found; building one from the training split",
            extra={"expected": str(reference_path)},
        )
        development = load_development_frame(cfg, materialize=False)
        frame = development[0] if isinstance(development, tuple) else development
        train, _ = split_holdout(frame, cfg.split, cfg.project.seed)
        reference = build_reference(
            train.drop(columns=[TARGET_COLUMN]), mapping, cfg.monitoring.drift
        )

    report = compute_drift(window, reference, cfg.monitoring.drift)
    decision = decide(report, cfg.monitoring.retraining, min_rows=cfg.monitoring.min_rows_for_drift)
    evidence = cfg.paths.absolute("evidence_dir")
    flag_path = save(decision, evidence / cfg.monitoring.retraining.flag_file)
    (evidence / FLAG_FILENAME.replace(".json", ".md")).write_text(
        report_markdown(decision), encoding="utf-8"
    )
    print(report_markdown(decision))
    return report, flag_path


def main(argv: list[str] | None = None) -> int:
    """Decide and, unless asked not to, act."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="decide and report, never train")
    parser.add_argument("--force", action="store_true", help="train even if the flag says no")
    parser.add_argument(
        "--window", type=Path, default=None, help="CSV of recent observations to evaluate"
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    configure_logging(cfg.logging)
    _, flag_path = evaluate(cfg, args.window)
    decision = load_decision(flag_path)
    recommended = bool(decision and decision.recommended)

    if args.check_only:
        print(f"flag em {flag_path}")
        return 0
    if not recommended and not args.force:
        print("Retreino nao recomendado. Use --force para treinar mesmo assim.")
        return 0

    print("Retreinando: python -m src.models.train --all")
    return train_main(["--all"])


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())

"""Data quality gate.

Checks, per raw inference variable, that the share of missing values is
within the range declared in the variable mapping and that every non-missing
value lies in the declared domain. Meant to run as a blocking step in CI and
before every training run.

Run as a module::

    python -m src.data.quality            # real dataset from config
    python -m src.data.quality --fixture  # synthetic fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass

import pandas as pd

from src.data.mapping import VariableMapping, get_mapping
from src.exceptions import DataContractError
from src.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ColumnQuality:
    """Quality metrics of one column."""

    column: str
    null_share: float
    null_share_max: float
    out_of_domain: int
    passed: bool


def check_quality(frame: pd.DataFrame, mapping: VariableMapping) -> list[ColumnQuality]:
    """Evaluate the null-share and domain rules for every raw variable.

    Parameters
    ----------
    frame : pandas.DataFrame
        Frame with the 61 raw variables in English names (target may be present).
    mapping : VariableMapping
        Declared ``null_share_max`` and domains.

    Returns
    -------
    list of ColumnQuality
        One entry per raw variable, in mapping order.
    """
    results: list[ColumnQuality] = []
    for variable in mapping.raw_variables:
        if variable.en not in frame.columns:
            results.append(ColumnQuality(variable.en, 1.0, variable.null_share_max, 0, False))
            continue
        series = pd.to_numeric(frame[variable.en], errors="coerce")
        null_share = float(series.isna().mean())
        present = series.dropna()
        domain = variable.domain
        if domain.type == "categorical":
            bad = int((~present.isin(domain.values or [])).sum())
        else:
            lo, hi = float(domain.min or 0), float(domain.max or 0)
            bad = int(((present < lo) | (present > hi)).sum())
        passed = null_share <= variable.null_share_max + 1e-12 and bad == 0
        results.append(ColumnQuality(variable.en, null_share, variable.null_share_max, bad, passed))
    return results


def enforce_quality(frame: pd.DataFrame, mapping: VariableMapping) -> list[ColumnQuality]:
    """Run :func:`check_quality` and raise on any failure.

    Raises
    ------
    DataContractError
        Listing every failing column with its null share and domain violations.
    """
    results = check_quality(frame, mapping)
    failed = [r for r in results if not r.passed]
    for r in results:
        logger.info("quality check", extra=asdict(r))
    if failed:
        detail = "; ".join(
            f"{r.column}: null_share={r.null_share:.4f} (max {r.null_share_max}), "
            f"out_of_domain={r.out_of_domain}"
            for r in failed
        )
        raise DataContractError(f"data quality gate failed for {len(failed)} column(s): {detail}")
    return results


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Data quality gate")
    parser.add_argument("--fixture", action="store_true", help="use the synthetic fixture")
    parser.add_argument("--report", type=str, default=None, help="write JSON report to this path")
    args = parser.parse_args(argv)

    # Imported here so the CLI has no side effects on import.
    from src.config import get_config
    from src.data.pipeline import load_development_frame
    from src.logging_setup import configure_logging

    cfg = get_config()
    configure_logging(cfg.logging)
    frame = load_development_frame(cfg, use_fixture=args.fixture)
    try:
        results = enforce_quality(frame, get_mapping())
    except DataContractError as error:
        logger.error(str(error))
        return 1
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump([asdict(r) for r in results], handle, indent=2)
    logger.info("data quality gate passed", extra={"columns": len(results)})
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())

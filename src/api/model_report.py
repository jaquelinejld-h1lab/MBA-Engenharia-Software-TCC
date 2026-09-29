"""Global report of the model in service, read from the artifacts of its run.

The interface is a pure HTTP client: it never opens ``artifacts/`` and never talks to
MLflow. Everything it shows about the champion therefore comes from here, where the JSON
files written at training time are read and reshaped for display:

===========================  =========================================================
File in the run directory    What it feeds
===========================  =========================================================
``metrics.json``             cross-validation summary, holdout metrics, confusion matrix
``roc.json``                 ROC curve and the sensitivity/specificity by threshold
``interpretation.json``      coefficients, importance by variable, contribution sample
``subgroup_metrics.json``    metrics by sex and age group
===========================  =========================================================

Every reader degrades gracefully: a file written by an older training run simply does
not exist, and the caller receives ``available=False`` with the reason instead of an
error, so one missing artifact never takes the page down.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.logging_setup import get_logger

logger = get_logger(__name__)

METRICS_FILE = "metrics.json"
ROC_FILE = "roc.json"
INTERPRETATION_FILE = "interpretation.json"
SUBGROUP_FILE = "subgroup_metrics.json"

_MISSING_RUN_DIR = "o modelo em servico nao expoe o diretorio do run"
_RETRAIN_HINT = "execute o treino novamente para gerar este artefato"


def read_json(run_dir: Path | None, filename: str) -> dict[str, Any] | list[Any] | None:
    """Read one JSON artifact of the run, or ``None`` when it is absent or unreadable.

    Parameters
    ----------
    run_dir : pathlib.Path or None
        Directory of the run serving the model.
    filename : str
        Artifact file name.

    Returns
    -------
    dict or list or None
        Parsed content, or ``None`` when the directory is unknown, the file is missing
        or its content is not valid JSON. A corrupt file is logged as a warning: it must
        not break the page, and it must not pass unnoticed either.
    """
    if run_dir is None:
        return None
    path = run_dir / filename
    if not path.is_file():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("model artifact unreadable", extra={"file": str(path), "reason": str(error)})
        return None
    if isinstance(parsed, dict | list):
        return parsed
    return None


def unavailable(run_dir: Path | None, filename: str) -> str:
    """Reason why an artifact could not be served, in the interface's language."""
    if run_dir is None:
        return _MISSING_RUN_DIR
    return f"{filename} nao encontrado em {run_dir.name}: {_RETRAIN_HINT}"


# --------------------------------------------------------------------------- summary
def summary(run_dir: Path | None) -> dict[str, Any]:
    """Cross-validation summary, holdout metrics and confusion matrix of the champion."""
    metrics = read_json(run_dir, METRICS_FILE)
    if not isinstance(metrics, dict):
        return {"available": False, "reason": unavailable(run_dir, METRICS_FILE)}
    cv = metrics.get("cv", {})
    holdout = metrics.get("holdout", {})
    folds = cv.get("folds", []) if isinstance(cv, dict) else []
    return {
        "available": True,
        "reason": None,
        "cross_validation": cv.get("summary", {}) if isinstance(cv, dict) else {},
        "n_folds": len(folds),
        "holdout": holdout,
    }


# ------------------------------------------------------------------------------- roc
def _thin(values: list[float], max_points: int) -> list[float]:
    """Keep at most ``max_points`` values at a fixed stride, always keeping the last."""
    if max_points <= 0 or len(values) <= max_points:
        return [float(value) for value in values]
    stride = -(-len(values) // max_points)
    kept = [float(value) for value in values[::stride]]
    last = float(values[-1])
    if not kept or kept[-1] != last:
        kept.append(last)
    return kept


def _curve(payload: dict[str, Any], key: str, max_points: int) -> dict[str, list[float]] | None:
    """Extract and thin one ROC curve of a ``roc.json`` payload."""
    curve = payload.get(key)
    if not isinstance(curve, dict):
        return None
    fpr, tpr = curve.get("fpr"), curve.get("tpr")
    if not isinstance(fpr, list) or not isinstance(tpr, list) or len(fpr) != len(tpr):
        return None
    return {"fpr": _thin(fpr, max_points), "tpr": _thin(tpr, max_points)}


def _auc(run_dir: Path, split: str) -> float | None:
    """Holdout or cross-validation AUC of a run, from its metrics file."""
    metrics = read_json(run_dir, METRICS_FILE)
    if not isinstance(metrics, dict):
        return None
    if split == "holdout":
        holdout = metrics.get("holdout", {})
        value = holdout.get("auc_roc") if isinstance(holdout, dict) else None
    else:
        cv = metrics.get("cv", {})
        inner = cv.get("summary", {}) if isinstance(cv, dict) else {}
        value = inner.get("auc_roc_mean") if isinstance(inner, dict) else None
    return float(value) if isinstance(value, int | float) else None


def roc(run_dir: Path | None, champion: str, max_points: int) -> dict[str, Any]:
    """ROC of every run that trained alongside the champion, holdout split.

    Sibling runs are the directories next to the champion's, each with its own
    ``roc.json``. A run without the file is skipped rather than reported as an error:
    non-linear runs and older trainings may simply not have one.
    """
    curves: list[dict[str, Any]] = []
    if run_dir is not None and run_dir.parent.is_dir():
        for candidate in sorted(run_dir.parent.iterdir()):
            if not candidate.is_dir():
                continue
            payload = read_json(candidate, ROC_FILE)
            if not isinstance(payload, dict):
                continue
            curve = _curve(payload, "holdout", max_points)
            if curve is None:
                continue
            curves.append(
                {
                    "run": candidate.name,
                    "is_champion": candidate.name == champion,
                    "auc_holdout": _auc(candidate, "holdout"),
                    "auc_cv_mean": _auc(candidate, "cv"),
                    **curve,
                }
            )
    if not curves:
        return {"available": False, "reason": unavailable(run_dir, ROC_FILE), "curves": []}
    curves.sort(key=lambda row: (not row["is_champion"], -(row["auc_holdout"] or 0.0)))
    return {"available": True, "reason": None, "curves": curves}


def threshold_sweep(run_dir: Path | None, max_points: int) -> dict[str, Any]:
    """Sensitivity and specificity by decision threshold, from the holdout ROC.

    The ROC already carries the sweep: at each point, sensitivity is the true positive
    rate and specificity is one minus the false positive rate. Deriving it here avoids
    persisting the same numbers twice.
    """
    payload = read_json(run_dir, ROC_FILE)
    if not isinstance(payload, dict):
        return {"available": False, "reason": unavailable(run_dir, ROC_FILE), "points": []}
    curve = payload.get("holdout")
    if not isinstance(curve, dict):
        return {"available": False, "reason": unavailable(run_dir, ROC_FILE), "points": []}
    fpr = curve.get("fpr", [])
    tpr = curve.get("tpr", [])
    thresholds = curve.get("thresholds", [])
    if not (isinstance(fpr, list) and isinstance(tpr, list) and isinstance(thresholds, list)):
        return {"available": False, "reason": unavailable(run_dir, ROC_FILE), "points": []}

    points = [
        {
            "threshold": float(threshold),
            "sensitivity": float(sensitivity),
            "specificity": 1.0 - float(one_minus),
        }
        for threshold, sensitivity, one_minus in zip(thresholds, tpr, fpr, strict=False)
        # The first point of a scikit-learn ROC carries an artificial threshold above 1.
        if 0.0 <= float(threshold) <= 1.0
    ]
    stride = max(1, -(-len(points) // max_points)) if max_points > 0 else 1
    return {"available": True, "reason": None, "points": points[::stride]}


# ---------------------------------------------------------------------- interpretation
def importance(run_dir: Path | None, top_coefficients: int) -> dict[str, Any]:
    """Return the importance by model variable and the strongest coefficients."""
    payload = read_json(run_dir, INTERPRETATION_FILE)
    if not isinstance(payload, dict):
        return {
            "available": False,
            "reason": unavailable(run_dir, INTERPRETATION_FILE),
            "method": "",
            "variables": [],
            "coefficients": [],
        }
    coefficients = payload.get("coefficients", [])
    ranked = sorted(
        (row for row in coefficients if isinstance(row, dict)),
        key=lambda row: abs(float(row.get("coefficient", 0.0))),
        reverse=True,
    )[:top_coefficients]
    return {
        "available": True,
        "reason": None,
        "method": str(payload.get("method", "")),
        "variables": payload.get("importance", []),
        "coefficients": ranked,
    }


def contributions(run_dir: Path | None) -> dict[str, Any]:
    """Return the contribution sample of the holdout, in columnar form.

    The payload is the matrix as persisted, not one record per cell: a record per cell
    repeats the dummy name thousands of times and multiplies the response size by six
    for exactly the same chart. The client pairs the rows with ``dummies`` by position.
    """
    payload = read_json(run_dir, INTERPRETATION_FILE)
    sample = payload.get("contributions") if isinstance(payload, dict) else None
    if not isinstance(sample, dict):
        return {
            "available": False,
            "reason": unavailable(run_dir, INTERPRETATION_FILE),
            "method": "",
            "rows": 0,
            "rows_available": 0,
            "dummies": [],
            "labels_pt": [],
            "contributions": [],
            "present": [],
        }
    return {
        "available": True,
        "reason": None,
        "method": str(payload.get("method", "")) if isinstance(payload, dict) else "",
        "rows": int(sample.get("rows", 0)),
        "rows_available": int(sample.get("rows_available", 0)),
        "dummies": [str(value) for value in sample.get("dummies", [])],
        "labels_pt": [str(value) for value in sample.get("labels_pt", [])],
        "contributions": sample.get("contributions", []),
        "present": sample.get("present", []),
    }


# -------------------------------------------------------------------------- subgroups
def subgroups(run_dir: Path | None) -> dict[str, Any]:
    """Out-of-fold metrics by sex and age group, as persisted by the training."""
    payload = read_json(run_dir, SUBGROUP_FILE)
    if not isinstance(payload, list):
        return {"available": False, "reason": unavailable(run_dir, SUBGROUP_FILE), "rows": []}
    rows = [row for row in payload if isinstance(row, dict)]
    return {"available": True, "reason": None, "rows": rows}

"""Data drift statistics: Population Stability Index (PSI) and Kolmogorov-Smirnov (KS).

Both statistics are implemented here, on purpose, instead of importing a
monitoring library (see the ADR on drift tooling): the formulas are short,
the thresholds live in ``configs/config.yaml`` and the tests pin the numbers.

Reference window
----------------
The reference is the training split of the promoted run, summarised once by
:func:`build_reference` and persisted as JSON next to the model artifacts
(``monitoring.reference_file``). It stores, per raw input variable:

* categorical variables: the reference share of each domain code plus a
  ``missing`` bin;
* numeric variables: the quantile bin edges (``psi_bins`` bins), the
  reference share per bin plus a ``missing`` bin, and the sorted reference
  sample (non-missing values) used by KS.

Missing values are a bin of their own because several PNS variables are
answered only by a subset of respondents (income sources, past smoking);
dropping them would compute PSI on a handful of rows and raise spurious
alerts, while a change in the answering share is itself drift.

No raw row is stored for categorical variables and the numeric samples are
the training values only, so the file carries nothing beyond what the
model artifacts already imply.

Definitions
-----------
PSI over ``B`` bins with reference share ``r_b`` and current share ``c_b``::

    PSI = sum_b (c_b - r_b) * ln(c_b / r_b)

Shares equal to zero are replaced by ``epsilon`` before the logarithm.
Interpretation thresholds (``psi_warning``, ``psi_alert``) come from config.
KS is the two-sample statistic of :func:`scipy.stats.ks_2samp` on numeric
variables only, computed when both non-missing samples hold at least
``ks_min_rows`` values; a variable is flagged when the statistic exceeds
``ks_statistic_alert`` and the p-value is below ``ks_pvalue_alert``.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DriftConfig
from src.data.mapping import VariableMapping
from src.exceptions import HypertensionPipelineError
from src.monitoring.thresholds import DriftLevel, psi_level

__all__ = [
    "MISSING_BIN",
    "DriftError",
    "DriftReport",
    "ReferenceWindow",
    "VariableDrift",
    "VariableReference",
    "build_reference",
    "compute_drift",
    "ks_test",
    "load_reference",
    "psi",
    "psi_level",
    "report_markdown",
    "save_reference",
]


class DriftError(HypertensionPipelineError):
    """Reference window missing, malformed or incompatible with the current data."""


# --------------------------------------------------------------------- reference


@dataclass(frozen=True)
class VariableReference:
    """Summary of one variable on the reference window."""

    name: str
    kind: Literal["categorical", "numeric"]
    bins: list[str]
    proportions: list[float]
    edges: list[float] = field(default_factory=list)
    sample: list[float] = field(default_factory=list)
    missing_share: float = 0.0


@dataclass(frozen=True)
class ReferenceWindow:
    """Persisted summary of the training window."""

    source: str
    rows: int
    variables: dict[str, VariableReference]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        return {
            "source": self.source,
            "rows": self.rows,
            "variables": {name: asdict(ref) for name, ref in self.variables.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReferenceWindow:
        """Inverse of :meth:`to_dict`; raises ``DriftError`` on a malformed file."""
        try:
            variables = {
                name: VariableReference(**spec) for name, spec in payload["variables"].items()
            }
            return cls(
                source=str(payload["source"]), rows=int(payload["rows"]), variables=variables
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DriftError(f"malformed drift reference: {exc}") from exc


def _clean_numeric(values: pd.Series) -> np.ndarray:
    array: np.ndarray = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return array


def _numeric_edges(sample: np.ndarray, bins: int) -> list[float]:
    """Quantile edges with open outer bins; duplicate edges are collapsed.

    An all-missing reference column yields the single bin ``[-inf, inf)``;
    its whole mass then sits in the ``missing`` bin.
    """
    if sample.size == 0:
        return [-math.inf, math.inf]
    inner = np.unique(np.quantile(sample, np.linspace(0, 1, bins + 1)[1:-1]))
    return [-math.inf, *inner.tolist(), math.inf]


MISSING_BIN = "missing"


def _numeric_shares(values: pd.Series, edges: list[float]) -> tuple[np.ndarray, float]:
    """Shares per bin over all rows, the last bin being the missing values."""
    numeric = pd.to_numeric(values, errors="coerce")
    sample = numeric.dropna().to_numpy(dtype=float)
    counts, _ = np.histogram(sample, bins=np.asarray(edges, dtype=float))
    n_missing = int(numeric.isna().sum())
    total = counts.sum() + n_missing
    all_counts = np.append(counts, n_missing).astype(float)
    shares = all_counts / total if total else all_counts
    return shares, (n_missing / total if total else 0.0)


def _categorical_shares(values: pd.Series, codes: list[int]) -> tuple[np.ndarray, float]:
    """Shares per code over all rows, the last bin being the missing values."""
    numeric = pd.to_numeric(values, errors="coerce")
    counts = np.array([(numeric == code).sum() for code in codes], dtype=float)
    n_missing = float(numeric.isna().sum())
    total = counts.sum() + n_missing
    all_counts = np.append(counts, n_missing)
    shares = all_counts / total if total else all_counts
    return shares, (n_missing / total if total else 0.0)


def build_reference(
    frame: pd.DataFrame, mapping: VariableMapping, drift: DriftConfig, *, source: str = "train"
) -> ReferenceWindow:
    """Summarise ``frame`` (English raw columns) as the drift reference.

    Parameters
    ----------
    frame : pandas.DataFrame
        Training split with the 61 raw inference columns.
    mapping : VariableMapping
        Declares which variables are categorical and their code lists.
    drift : DriftConfig
        Number of PSI bins.
    source : str
        Label stored in the file (``"train"``).
    """
    variables: dict[str, VariableReference] = {}
    for variable in mapping.raw_variables:
        if variable.en not in frame.columns:
            raise DriftError(f"reference frame lacks column '{variable.en}'")
        column = frame[variable.en]
        if variable.domain.type == "categorical":
            codes = list(variable.domain.values or [])
            shares, missing_share = _categorical_shares(column, codes)
            variables[variable.en] = VariableReference(
                name=variable.en,
                kind="categorical",
                bins=[*(str(code) for code in codes), MISSING_BIN],
                proportions=shares.tolist(),
                missing_share=missing_share,
            )
        else:
            sample = np.sort(_clean_numeric(column))
            edges = _numeric_edges(sample, drift.psi_bins)
            shares, missing_share = _numeric_shares(column, edges)
            labels = [f"[{lo:g}, {hi:g})" for lo, hi in itertools.pairwise(edges)]
            variables[variable.en] = VariableReference(
                name=variable.en,
                kind="numeric",
                bins=[*labels, MISSING_BIN],
                proportions=shares.tolist(),
                edges=edges,
                sample=sample.tolist(),
                missing_share=missing_share,
            )
    return ReferenceWindow(source=source, rows=len(frame), variables=variables)


def save_reference(reference: ReferenceWindow, path: Path) -> Path:
    """Write the reference as JSON (``inf`` edges are encoded as strings)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = reference.to_dict()
    for spec in payload["variables"].values():
        spec["edges"] = [_encode_edge(e) for e in spec["edges"]]
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_reference(path: Path) -> ReferenceWindow:
    """Read a file written by :func:`save_reference`."""
    if not path.exists():
        raise DriftError(f"drift reference not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DriftError(f"drift reference is not valid JSON: {path}") from exc
    for spec in payload.get("variables", {}).values():
        spec["edges"] = [_decode_edge(e) for e in spec.get("edges", [])]
    return ReferenceWindow.from_dict(payload)


def _encode_edge(edge: float) -> float | str:
    if math.isinf(edge):
        return "-inf" if edge < 0 else "inf"
    return float(edge)


def _decode_edge(edge: float | str) -> float:
    if isinstance(edge, str):
        if edge not in {"-inf", "inf"}:
            raise DriftError(f"invalid bin edge '{edge}' in drift reference")
        return float(edge)
    return float(edge)


# --------------------------------------------------------------------- statistics


def psi(reference: Iterable[float], current: Iterable[float], epsilon: float) -> float:
    """Compute the Population Stability Index between two share vectors of equal length.

    Parameters
    ----------
    reference, current : iterable of float
        Shares per bin; each vector sums to one (zero vectors give ``0.0``).
    epsilon : float
        Replacement for zero shares before the logarithm.
    """
    r = np.asarray(list(reference), dtype=float)
    c = np.asarray(list(current), dtype=float)
    if r.shape != c.shape:
        raise DriftError(f"PSI share vectors differ in length: {r.size} vs {c.size}")
    if r.size == 0 or (r.sum() == 0 and c.sum() == 0):
        return 0.0
    r = np.where(r <= 0, epsilon, r)
    c = np.where(c <= 0, epsilon, c)
    return float(np.sum((c - r) * np.log(c / r)))


def ks_test(
    reference_sample: np.ndarray, current_sample: np.ndarray, min_rows: int = 1
) -> tuple[float, float]:
    """Two-sample KS statistic and p-value (``nan`` when a sample is too small)."""
    if reference_sample.size < min_rows or current_sample.size < min_rows:
        return math.nan, math.nan
    result = stats.ks_2samp(reference_sample, current_sample)
    return float(result.statistic), float(result.pvalue)


# ------------------------------------------------------------------------ report


@dataclass(frozen=True)
class VariableDrift:
    """Drift statistics of one variable on the current window."""

    name: str
    kind: str
    rows: int
    psi: float
    psi_level: DriftLevel
    ks_statistic: float | None
    ks_pvalue: float | None
    ks_alert: bool
    missing_share_reference: float
    missing_share_current: float

    @property
    def alert(self) -> bool:
        """Any alert on this variable."""
        return self.psi_level == "alert" or self.ks_alert


@dataclass(frozen=True)
class DriftReport:
    """Per-variable drift on a current window against a reference."""

    reference_rows: int
    current_rows: int
    variables: list[VariableDrift]

    @property
    def alerts(self) -> list[str]:
        """Names of the variables in alert."""
        return [v.name for v in self.variables if v.alert]

    @property
    def warnings(self) -> list[str]:
        """Names of the variables at PSI warning level (not alert)."""
        return [v.name for v in self.variables if v.psi_level == "warning" and not v.alert]

    def to_frame(self) -> pd.DataFrame:
        """Tabular view sorted by PSI descending."""
        frame = pd.DataFrame([asdict(v) | {"alert": v.alert} for v in self.variables])
        return frame.sort_values("psi", ascending=False).reset_index(drop=True)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        return {
            "reference_rows": self.reference_rows,
            "current_rows": self.current_rows,
            "alerts": self.alerts,
            "warnings": self.warnings,
            "variables": [asdict(v) | {"alert": v.alert} for v in self.variables],
        }


def _variable_drift(ref: VariableReference, column: pd.Series, drift: DriftConfig) -> VariableDrift:
    if ref.kind == "categorical":
        codes = [int(b) for b in ref.bins if b != MISSING_BIN]
        shares, missing_share = _categorical_shares(column, codes)
        value = psi(ref.proportions, shares, drift.epsilon)
        ks_stat: float | None = None
        ks_p: float | None = None
        ks_alert = False
        rows = len(column.dropna())
    else:
        current = _clean_numeric(column)
        shares, missing_share = _numeric_shares(column, ref.edges)
        value = psi(ref.proportions, shares, drift.epsilon)
        stat, p_value = ks_test(np.asarray(ref.sample, dtype=float), current, drift.ks_min_rows)
        ks_stat, ks_p = (None, None) if math.isnan(stat) else (stat, p_value)
        ks_alert = (
            ks_stat is not None
            and ks_p is not None
            and ks_stat > drift.ks_statistic_alert
            and ks_p < drift.ks_pvalue_alert
        )
        rows = int(current.size)
    return VariableDrift(
        name=ref.name,
        kind=ref.kind,
        rows=rows,
        psi=value,
        psi_level=psi_level(value, drift),
        ks_statistic=ks_stat,
        ks_pvalue=ks_p,
        ks_alert=ks_alert,
        missing_share_reference=ref.missing_share,
        missing_share_current=missing_share,
    )


def compute_drift(
    reference: ReferenceWindow, current: pd.DataFrame, drift: DriftConfig
) -> DriftReport:
    """Compare ``current`` (English raw columns) with the persisted reference.

    Variables absent from ``current`` raise ``DriftError``; extra columns are
    ignored. Missing values are excluded from the shares and reported apart
    as ``missing_share_current``.
    """
    if current.empty:
        raise DriftError("current window is empty")
    missing = [name for name in reference.variables if name not in current.columns]
    if missing:
        raise DriftError(f"current window lacks columns: {missing}")
    variables = [
        _variable_drift(ref, current[name], drift) for name, ref in reference.variables.items()
    ]
    return DriftReport(
        reference_rows=reference.rows, current_rows=len(current), variables=variables
    )


def report_markdown(report: DriftReport, title: str) -> str:
    """Portuguese Markdown table of the report for ``evidencias/``."""
    lines = [
        f"# {title}",
        "",
        f"Referencia: {report.reference_rows} linhas. Janela atual: {report.current_rows} linhas.",
        f"Variaveis em alerta: {', '.join(report.alerts) or 'nenhuma'}.",
        f"Variaveis em aviso: {', '.join(report.warnings) or 'nenhuma'}.",
        "",
        "| Variavel | Tipo | PSI | Nivel PSI | KS | p-valor KS | Alerta KS |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for v in sorted(report.variables, key=lambda x: x.psi, reverse=True):
        ks = "" if v.ks_statistic is None else f"{v.ks_statistic:.4f}"
        p = "" if v.ks_pvalue is None else f"{v.ks_pvalue:.4g}"
        lines.append(
            f"| {v.name} | {v.kind} | {v.psi:.4f} | {v.psi_level} | {ks} | {p} | "
            f"{'sim' if v.ks_alert else 'nao'} |"
        )
    return "\n".join(lines) + "\n"

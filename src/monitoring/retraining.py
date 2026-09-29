"""Retraining flag: when the evidence says the model should be trained again.

Drift is a symptom, not a verdict. A single variable crossing the PSI warning band means
the population moved a little; several variables in alert at once means the data the
model sees is no longer the data it learned from, and that is when retraining is the
answer rather than a threshold tweak.

The decision is declarative and auditable:

* every variable in alert (PSI above the alert band, or a significant KS) is a reason;
* the flag turns on from ``retraining.min_variables_in_alert`` reasons upwards;
* a window smaller than ``monitoring.min_rows_for_drift`` decides nothing, and says so,
  because an alert computed on a handful of observations is noise.

The result is written to ``evidencias/retraining_flag.json`` so the runbook, the CI and
the ``/metrics`` gauge all read the same file instead of each reaching its own verdict.
Nothing here retrains anything: deciding and acting are kept apart on purpose, so that
the decision can be reviewed before a model is replaced.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import RetrainingConfig
from src.monitoring.drift import DriftReport

FLAG_FILENAME = "retraining_flag.json"

_INSUFFICIENT = "janela insuficiente para decidir"
_STABLE = "populacao estavel: nenhuma variavel em alerta"


@dataclass(frozen=True)
class RetrainingDecision:
    """Whether retraining is recommended, and exactly why."""

    recommended: bool
    reasons: list[str] = field(default_factory=list)
    variables_in_alert: list[str] = field(default_factory=list)
    variables_in_warning: list[str] = field(default_factory=list)
    current_rows: int = 0
    reference_rows: int = 0
    min_variables_in_alert: int = 0
    evaluated: bool = False
    decided_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        return asdict(self)

    @property
    def summary(self) -> str:
        """One line for a log or the console."""
        if not self.evaluated:
            return _INSUFFICIENT
        if not self.recommended:
            return _STABLE if not self.variables_in_alert else "alertas abaixo do minimo"
        return f"retreino recomendado: {len(self.variables_in_alert)} variaveis em alerta"


def decide(
    report: DriftReport | None, cfg: RetrainingConfig, *, min_rows: int
) -> RetrainingDecision:
    """Turn a drift report into a retraining decision.

    Parameters
    ----------
    report : DriftReport or None
        Drift of the current window against the training reference; ``None`` when no
        reference or no window is available.
    cfg : RetrainingConfig
        How many variables in alert are needed to recommend retraining.
    min_rows : int
        Minimum observations for the window to decide anything
        (``monitoring.min_rows_for_drift``).

    Returns
    -------
    RetrainingDecision
        ``evaluated=False`` when there was not enough evidence to decide, which is not
        the same as deciding that the model is fine.
    """
    now = datetime.now(UTC).isoformat(timespec="seconds")
    if report is None or report.current_rows < min_rows:
        return RetrainingDecision(
            recommended=False,
            reasons=[_INSUFFICIENT],
            current_rows=report.current_rows if report else 0,
            reference_rows=report.reference_rows if report else 0,
            min_variables_in_alert=cfg.min_variables_in_alert,
            evaluated=False,
            decided_at=now,
        )

    alerts = report.alerts
    reasons = [
        f"{variable.name}: PSI {variable.psi:.4f} ({variable.psi_level})"
        + (f", KS p={variable.ks_pvalue:.4g}" if variable.ks_pvalue is not None else "")
        for variable in report.variables
        if variable.alert
    ]
    recommended = len(alerts) >= cfg.min_variables_in_alert
    return RetrainingDecision(
        recommended=recommended,
        reasons=reasons or [_STABLE],
        variables_in_alert=alerts,
        variables_in_warning=report.warnings,
        current_rows=report.current_rows,
        reference_rows=report.reference_rows,
        min_variables_in_alert=cfg.min_variables_in_alert,
        evaluated=True,
        decided_at=now,
    )


def save(decision: RetrainingDecision, path: Path) -> Path:
    """Persist the decision as JSON, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision.to_dict(), indent=1), encoding="utf-8")
    return path


def load(path: Path) -> RetrainingDecision | None:
    """Read a persisted decision, or ``None`` when it is absent or unreadable."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    known = {key: payload.get(key) for key in RetrainingDecision.__dataclass_fields__}
    return RetrainingDecision(**known)  # type: ignore[arg-type]


def report_markdown(decision: RetrainingDecision) -> str:
    """Human-readable evidence of the decision."""
    lines = [
        "# Flag de retreinamento",
        "",
        f"Decisao: **{'retreinar' if decision.recommended else 'manter o modelo'}** "
        f"({decision.summary}).",
        "",
        f"- Avaliada em: {decision.decided_at}",
        f"- Observacoes na janela: {decision.current_rows} "
        f"(referencia: {decision.reference_rows})",
        f"- Minimo de variaveis em alerta para recomendar: {decision.min_variables_in_alert}",
        f"- Variaveis em alerta: {', '.join(decision.variables_in_alert) or 'nenhuma'}",
        f"- Variaveis em atencao: {', '.join(decision.variables_in_warning) or 'nenhuma'}",
        "",
        "## Motivos",
        "",
        *[f"- {reason}" for reason in decision.reasons],
        "",
        "A flag recomenda, nao executa. O retreino passa pelo runbook: conferir o gate de",
        "qualidade de dados, rodar o treino, comparar com o campeao atual e promover apenas",
        "se as metricas de aceitacao forem atendidas.",
        "",
    ]
    return "\n".join(lines)

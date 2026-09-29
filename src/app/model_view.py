"""Chart specifications and tables of the global model page.

Every function here is pure: it turns a payload of ``/model/<section>`` into a
Vega-Lite specification (a plain dictionary) or into a ``DataFrame`` ready to display.
Streamlit renders them with ``st.vega_lite_chart``, which is part of Streamlit itself,
so the page gains full control over the charts without adding a plotting dependency to
the interface image.

Colour follows the job of each variable, not taste:

===================  ======================================================
Chart                Encoding
===================  ======================================================
ROC by run           categorical: one hue per run, in fixed order
Importance           magnitude: one hue, single series, sorted
Coefficients         polarity: diverging blue/red around zero
Contributions        polarity on x, two hues for the dummy being active or not
Threshold            two named series, one hue each
Subgroups            categorical by group, faceted by metric
===================  ======================================================

The palette is the validated categorical order, with its own steps for the dark
surface; :func:`palette` picks the set that matches the active Streamlit theme.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Categorical slots in fixed order (never cycled) and the surface-dependent ink.
_LIGHT = {
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"],
    "positive": "#2a78d6",
    "negative": "#e34948",
    "muted": "#9a9a94",
    "text": "#0b0b0b",
}
_DARK = {
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"],
    "positive": "#3987e5",
    "negative": "#e66767",
    "muted": "#8a8a84",
    "text": "#ffffff",
}

# Texto exibido ao usuario final: descreve o que o grafico mostra, sem nomear metodo.
RISK_DISCLAIMER = (
    "Cada ponto mostra quanto uma característica desloca o risco estimado de uma "
    "pessoa, para cima ou para baixo, em relação à categoria de referência."
)

# Nota de metodo, exibida na ficha tecnica e nao no corpo da pagina. O nome do metodo
# importa para auditoria e para o TCC, nao para quem le um resultado de triagem.
METHOD_NOTE = (
    "Explicabilidade por contribuições lineares (coeficiente multiplicado pelo valor "
    "da categoria), não SHAP. A escolha está registrada no ADR 0003."
)


def percentual(valor: Any, casas: int = 1) -> str:
    """Formata uma proporcao como percentual no padrao brasileiro.

    Parameters
    ----------
    valor : Any
        Proporcao entre 0 e 1. Valor ausente ou nao numerico devolve um traco.
    casas : int, optional
        Casas decimais exibidas.

    Returns
    -------
    str
        Percentual com virgula decimal, por exemplo "70,0%".
    """
    if not isinstance(valor, int | float) or isinstance(valor, bool):
        return "-"
    return f"{float(valor) * 100:.{casas}f}%".replace(".", ",")


def decimal(valor: Any, casas: int = 3) -> str:
    """Formata um numero com virgula decimal, ou um traco quando ausente."""
    if not isinstance(valor, int | float) or isinstance(valor, bool):
        return "-"
    return f"{float(valor):.{casas}f}".replace(".", ",")


def inteiro(valor: int) -> str:
    """Formata um inteiro com ponto como separador de milhar."""
    return f"{int(valor):,}".replace(",", ".")


def palette(dark: bool = False) -> dict[str, Any]:
    """Return the colour slots for the active surface."""
    return _DARK if dark else _LIGHT


# ------------------------------------------------------------------------ summary
# Cartoes de destaque. A acuracia saiu: com prevalencia de 17 %, um modelo que nao
# sinaliza ninguem alcanca 83 % e parece bom, o que a torna enganosa como numero de
# vitrine. A AUC-ROC ocupa o lugar dela por ser independente do ponto de corte e por
# ser o criterio de aceitacao declarado do projeto.
_HOLDOUT_CARDS: tuple[tuple[str, str, str], ...] = (
    ("auc_roc", "AUC-ROC",
     "Capacidade de ordenar corretamente quem tem maior risco. "
     "Não depende do ponto de corte."),
    ("sensitivity", "Sensibilidade",
     "Entre as pessoas com hipertensão, quantas o modelo sinaliza."),
    ("precision", "Precisão",
     "Entre as pessoas sinalizadas, quantas realmente têm hipertensão."),
    ("f1", "F1",
     "Equilíbrio entre sensibilidade e precisão, em um único número."),
)


def headline_cards(summary: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Format the headline metrics shown at the top of the model page.

    Parameters
    ----------
    summary : dict
        Payload of ``/model/summary``.

    Returns
    -------
    list of (str, str, str)
        Label, value as a percentage with one decimal, and the explanation shown on
        hover. A metric the payload does not carry shows as ``"-"`` rather than as
        zero, so a missing artifact never looks like a bad model.
    """
    holdout = summary.get("holdout") or {}
    return [
        (label, percentual(holdout.get(key)), ajuda)
        for key, label, ajuda in _HOLDOUT_CARDS
    ]


_METRICAS_TABELA: tuple[tuple[str, str], ...] = (
    ("auc_roc", "AUC-ROC"),
    ("sensitivity", "Sensibilidade"),
    ("specificity", "Especificidade"),
    ("precision", "Precisão"),
    ("f1", "F1"),
    ("accuracy", "Acurácia"),
)


def validation_frame(summary: dict[str, Any]) -> pd.DataFrame:
    """Build the table comparing cross-validation and the test split, metric by metric.

    The cards above use one decimal, which is enough to read at a glance; this table
    keeps three so the dispersion between folds stays legible.
    """
    cv = summary.get("cross_validation") or {}
    holdout = summary.get("holdout") or {}
    coluna_cv = f"Validação cruzada (k = {summary.get('n_folds', 0)})"
    linhas = []
    for key, label in _METRICAS_TABELA:
        media, desvio = cv.get(f"{key}_mean"), cv.get(f"{key}_std")
        combinado = (
            f"{decimal(media)} ± {decimal(desvio)}"
            if isinstance(media, int | float) and isinstance(desvio, int | float)
            else "-"
        )
        linhas.append({
            "Métrica": label,
            coluna_cv: combinado,
            "Conjunto de teste": decimal(holdout.get(key)),
        })
    return pd.DataFrame(linhas)


def confusion_frame(summary: dict[str, Any]) -> pd.DataFrame:
    """Build the test-split confusion matrix as a two by two table with totals."""
    holdout = summary.get("holdout") or {}
    vn = int(holdout.get("true_negatives", 0))
    fp = int(holdout.get("false_positives", 0))
    fn = int(holdout.get("false_negatives", 0))
    vp = int(holdout.get("true_positives", 0))
    return pd.DataFrame(
        {
            "Não sinalizado": [vn, fn, vn + fn],
            "Sinalizado": [fp, vp, fp + vp],
            "Total": [vn + fp, fn + vp, vn + fp + fn + vp],
        },
        index=["Sem hipertensão", "Com hipertensão", "Total"],
    )


def confusion_summary(summary: dict[str, Any]) -> str:
    """Describe the confusion matrix in one sentence, without jargon.

    The matrix itself is a technical artifact; what the reader of a screening tool
    needs is the trade-off it encodes, which is how many people are sent for a
    confirmation exam for each case actually found.

    Parameters
    ----------
    summary : dict
        Payload of ``/model/summary``.

    Returns
    -------
    str
        Sentence in plain Portuguese, or an empty string when the payload lacks the
        four cells.
    """
    holdout = summary.get("holdout") or {}
    chaves = ("true_negatives", "false_positives", "false_negatives", "true_positives")
    if not all(isinstance(holdout.get(c), int | float) for c in chaves):
        return ""
    vn, fp, fn, vp = (int(holdout[c]) for c in chaves)
    total = vn + fp + fn + vp
    positivos = fn + vp
    sinalizados = fp + vp
    # Cada numero e formatado no ato: uma substituicao global de virgula por ponto
    # trocaria tambem as virgulas da frase.
    por_achado = decimal(sinalizados / vp, 1) if vp else "-"
    return (
        f"Em {inteiro(total)} pessoas avaliadas, {inteiro(positivos)} tinham "
        f"hipertensão. O modelo sinalizou {inteiro(sinalizados)} para confirmação e "
        f"encontrou {inteiro(vp)} delas, o que dá {por_achado} encaminhamentos por "
        f"caso identificado. Ficaram sem sinalização {inteiro(fn)} pessoas."
    )


# ---------------------------------------------------------------------------- roc
def roc_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten the ROC curves into one long frame, with the AUC in the series name."""
    rows = []
    for curve in payload.get("curves", []):
        auc = curve.get("auc_holdout")
        label = (
            f"{curve.get('run')} (AUC {float(auc):.3f})"
            if isinstance(auc, int | float)
            else str(curve.get("run"))
        )
        for fpr, tpr in zip(curve.get("fpr", []), curve.get("tpr", []), strict=False):
            rows.append(
                {
                    "serie": label,
                    "fpr": float(fpr),
                    "tpr": float(tpr),
                    "campeao": bool(curve.get("is_champion")),
                }
            )
    return pd.DataFrame(rows, columns=["serie", "fpr", "tpr", "campeao"])


def roc_spec(frame: pd.DataFrame, dark: bool = False) -> dict[str, Any]:
    """Build the ROC chart: one line per run plus the no-skill diagonal.

    The champion is drawn thicker, so it is identifiable without reading the legend,
    and the diagonal is a dashed rule rather than a series, because it is a reference
    and not a model.
    """
    colours = palette(dark)
    order = sorted(frame["serie"].unique().tolist()) if not frame.empty else []
    return {
        "height": 380,
        "layer": [
            {
                "data": {"values": [{"x": 0, "y": 0}, {"x": 1, "y": 1}]},
                "mark": {"type": "line", "strokeDash": [4, 4], "color": colours["muted"]},
                "encoding": {
                    "x": {"field": "x", "type": "quantitative"},
                    "y": {"field": "y", "type": "quantitative"},
                },
            },
            {
                "mark": {"type": "line", "interpolate": "step-after"},
                "encoding": {
                    "x": {
                        "field": "fpr",
                        "type": "quantitative",
                        "title": "Taxa de falsos positivos",
                        "scale": {"domain": [0, 1]},
                    },
                    "y": {
                        "field": "tpr",
                        "type": "quantitative",
                        "title": "Sensibilidade",
                        "scale": {"domain": [0, 1]},
                    },
                    "color": {
                        "field": "serie",
                        "type": "nominal",
                        "title": "Run (AUC no holdout)",
                        "scale": {"domain": order, "range": colours["series"]},
                        "legend": {"orient": "bottom", "columns": 2},
                    },
                    "strokeWidth": {
                        "condition": {"test": "datum.campeao", "value": 3},
                        "value": 1.5,
                    },
                    "tooltip": [
                        {"field": "serie", "type": "nominal", "title": "Run"},
                        {"field": "fpr", "type": "quantitative", "format": ".3f"},
                        {"field": "tpr", "type": "quantitative", "format": ".3f"},
                    ],
                },
            },
        ],
    }


def threshold_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten the threshold sweep into a long frame of two series."""
    rows = []
    for point in payload.get("threshold_points", []):
        threshold = float(point.get("threshold", 0.0))
        rows.append(
            {"limiar": threshold, "metrica": "Sensibilidade", "valor": float(point["sensitivity"])}
        )
        rows.append(
            {"limiar": threshold, "metrica": "Especificidade", "valor": float(point["specificity"])}
        )
    return pd.DataFrame(rows, columns=["limiar", "metrica", "valor"])


def threshold_spec(decision_threshold: float, dark: bool = False) -> dict[str, Any]:
    """Build sensitivity and specificity against the decision threshold.

    The threshold in service is marked with a rule: the chart exists to show what the
    declared cut of 0,5 costs on each side, not to suggest tuning it in the interface.
    """
    colours = palette(dark)
    return {
        "height": 300,
        "layer": [
            {
                "mark": {"type": "line", "strokeWidth": 2},
                "encoding": {
                    "x": {
                        "field": "limiar",
                        "type": "quantitative",
                        "title": "Limiar de decisao",
                        "scale": {"domain": [0, 1]},
                    },
                    "y": {
                        "field": "valor",
                        "type": "quantitative",
                        "title": "Valor",
                        "scale": {"domain": [0, 1]},
                    },
                    "color": {
                        "field": "metrica",
                        "type": "nominal",
                        "title": "",
                        "scale": {
                            "domain": ["Sensibilidade", "Especificidade"],
                            "range": [colours["series"][0], colours["series"][1]],
                        },
                        "legend": {"orient": "bottom"},
                    },
                    "tooltip": [
                        {"field": "metrica", "type": "nominal"},
                        {"field": "limiar", "type": "quantitative", "format": ".3f"},
                        {"field": "valor", "type": "quantitative", "format": ".3f"},
                    ],
                },
            },
            {
                "data": {"values": [{"limiar": float(decision_threshold)}]},
                "mark": {"type": "rule", "strokeDash": [4, 4], "color": colours["muted"]},
                "encoding": {"x": {"field": "limiar", "type": "quantitative"}},
            },
        ],
    }


# -------------------------------------------------------------------- interpretation
def importance_frame(payload: dict[str, Any], top: int) -> pd.DataFrame:
    """Build the importance table by model variable, strongest first."""
    rows = [
        {
            "variavel": str(row.get("label_pt") or row.get("model_variable")),
            "contribuicao": float(row.get("mean_abs_contribution", 0.0)),
            "origem": ", ".join(str(name) for name in row.get("source_raw", [])),
        }
        for row in payload.get("variables", [])
    ]
    frame = pd.DataFrame(rows, columns=["variavel", "contribuicao", "origem"])
    return frame.head(top)


def importance_spec(dark: bool = False) -> dict[str, Any]:
    """Build horizontal bars of importance: one measure, one hue, sorted."""
    colours = palette(dark)
    return {
        "height": {"step": 20},
        "mark": {"type": "bar", "cornerRadiusEnd": 3, "color": colours["series"][0]},
        "encoding": {
            "x": {
                "field": "contribuicao",
                "type": "quantitative",
                "title": "Peso médio no risco estimado",
            },
            "y": {"field": "variavel", "type": "nominal", "title": "", "sort": "-x"},
            "tooltip": [
                {"field": "variavel", "type": "nominal", "title": "Variável"},
                {"field": "origem", "type": "nominal", "title": "Origem no questionário"},
                {"field": "contribuicao", "type": "quantitative", "format": ".4f"},
            ],
        },
    }


def coefficients_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Build the coefficient table at the dummy level, with sign."""
    rows = [
        {
            "dummy": str(row.get("dummy")),
            "rotulo": (
                f"{row.get('label_pt')} = {row.get('category')}"
                if row.get("category")
                else str(row.get("label_pt"))
            ),
            "coeficiente": float(row.get("coefficient", 0.0)),
            "sentido": (
                "aumenta o risco" if float(row.get("coefficient", 0.0)) >= 0 else "reduz o risco"
            ),
        }
        for row in payload.get("coefficients", [])
    ]
    return pd.DataFrame(rows, columns=["dummy", "rotulo", "coeficiente", "sentido"])


def coefficients_spec(dark: bool = False) -> dict[str, Any]:
    """Build signed coefficient bars: diverging around zero, sorted by value."""
    colours = palette(dark)
    return {
        "height": {"step": 20},
        "mark": {"type": "bar", "cornerRadiusEnd": 3},
        "encoding": {
            "x": {
                "field": "coeficiente",
                "type": "quantitative",
                "title": "Efeito no risco: à esquerda reduz, à direita aumenta",
            },
            "y": {"field": "rotulo", "type": "nominal", "title": "", "sort": "-x"},
            "color": {
                "field": "sentido",
                "type": "nominal",
                "title": "",
                "scale": {
                    "domain": ["aumenta o risco", "reduz o risco"],
                    "range": [colours["negative"], colours["positive"]],
                },
                "legend": {"orient": "bottom"},
            },
            "tooltip": [
                {"field": "rotulo", "type": "nominal", "title": "Categoria"},
                {"field": "coeficiente", "type": "quantitative", "format": ".4f"},
            ],
        },
    }


def contributions_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten the columnar contribution sample into one row per observation and dummy."""
    labels = payload.get("labels_pt") or payload.get("dummies") or []
    present = payload.get("present", [])
    rows = []
    for position, line in enumerate(payload.get("contributions", [])):
        flags = present[position] if position < len(present) else []
        for column, value in enumerate(line):
            if column >= len(labels):
                continue
            active = bool(flags[column]) if column < len(flags) else False
            rows.append(
                {
                    "rotulo": str(labels[column]),
                    "contribuicao": float(value),
                    "situacao": "categoria presente" if active else "categoria ausente",
                }
            )
    return pd.DataFrame(rows, columns=["rotulo", "contribuicao", "situacao"])


def contributions_spec(dark: bool = False) -> dict[str, Any]:
    """Build the beeswarm of linear contributions, one row per dummy.

    Vertical jitter is deterministic per mark and only separates overlapping points;
    the y axis carries no quantity, which the absent tick labels make explicit.
    """
    colours = palette(dark)
    return {
        "height": {"step": 26},
        "mark": {"type": "circle", "size": 22, "opacity": 0.45},
        "transform": [{"calculate": "random() - 0.5", "as": "jitter"}],
        "encoding": {
            "x": {
                "field": "contribuicao",
                "type": "quantitative",
                "title": "Deslocamento do risco",
            },
            "y": {"field": "rotulo", "type": "nominal", "title": "", "sort": "-x"},
            "yOffset": {"field": "jitter", "type": "quantitative", "scale": {"range": [-9, 9]}},
            "color": {
                "field": "situacao",
                "type": "nominal",
                "title": "",
                "scale": {
                    "domain": ["categoria presente", "categoria ausente"],
                    "range": [colours["series"][0], colours["muted"]],
                },
                "legend": {"orient": "bottom"},
            },
            "tooltip": [
                {"field": "rotulo", "type": "nominal", "title": "Categoria"},
                {"field": "contribuicao", "type": "quantitative", "format": ".4f"},
                {"field": "situacao", "type": "nominal", "title": "Situação"},
            ],
        },
    }


# --------------------------------------------------------------------------- equity
_SUBGROUP_METRICS: tuple[tuple[str, str], ...] = (
    ("auc_roc", "AUC-ROC"),
    ("sensitivity", "Sensibilidade"),
    ("specificity", "Especificidade"),
    ("precision", "Precisão"),
)
_DIMENSION_LABELS = {"sex": "Sexo", "age_group": "Faixa etária"}


def subgroups_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten the subgroup metrics into a long frame for the faceted chart."""
    rows = []
    for row in payload.get("rows", []):
        dimension = _DIMENSION_LABELS.get(str(row.get("dimension")), str(row.get("dimension")))
        for key, label in _SUBGROUP_METRICS:
            value = row.get(key)
            if not isinstance(value, int | float):
                continue
            rows.append(
                {
                    "dimensao": dimension,
                    "grupo": str(row.get("group")),
                    "metrica": label,
                    "valor": float(value),
                    "n": int(row.get("n", 0)),
                }
            )
    return pd.DataFrame(rows, columns=["dimensao", "grupo", "metrica", "valor", "n"])


def subgroups_spec(frame: pd.DataFrame, dark: bool = False) -> dict[str, Any]:
    """Build grouped bars of each metric by subgroup, faceted by metric."""
    colours = palette(dark)
    groups = sorted(frame["grupo"].unique().tolist()) if not frame.empty else []
    return {
        "mark": {"type": "bar", "cornerRadiusEnd": 3},
        "width": 150,
        "height": 220,
        "encoding": {
            "column": {"field": "metrica", "type": "nominal", "title": ""},
            "x": {"field": "grupo", "type": "nominal", "title": "", "axis": {"labelAngle": -40}},
            "y": {
                "field": "valor",
                "type": "quantitative",
                "title": "Valor",
                "scale": {"domain": [0, 1]},
            },
            "color": {
                "field": "grupo",
                "type": "nominal",
                "title": "",
                "scale": {"domain": groups, "range": colours["series"]},
                "legend": {"orient": "bottom", "columns": 4},
            },
            "tooltip": [
                {"field": "grupo", "type": "nominal", "title": "Grupo"},
                {"field": "metrica", "type": "nominal"},
                {"field": "valor", "type": "quantitative", "format": ".4f"},
                {"field": "n", "type": "quantitative", "title": "Observacoes"},
            ],
        },
    }

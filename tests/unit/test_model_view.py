"""Charts and tables of the model page: shapes, formatting and spec/frame agreement.

The most valuable check here is :func:`_fields`: a Vega-Lite specification names the
columns it draws as plain strings, so renaming a column in a frame builder without
updating the spec produces an empty chart and no error anywhere. Each test pairs the
spec with its frame and asserts every field exists.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from src.app import model_view

_SUMMARY = {
    "available": True,
    "n_folds": 10,
    "decision_threshold": 0.5,
    "cross_validation": {
        "auc_roc_mean": 0.7484,
        "auc_roc_std": 0.0215,
        "sensitivity_mean": 0.6794,
        "sensitivity_std": 0.06,
    },
    "holdout": {
        "auc_roc": 0.7468,
        "accuracy": 0.7010,
        "specificity": 0.7049,
        "precision": 0.3206,
        "sensitivity": 0.6777,
        "f1": 0.4353,
        "true_negatives": 727,
        "false_positives": 303,
        "false_negatives": 68,
        "true_positives": 143,
        "n": 1241,
    },
}
_ROC = {
    "available": True,
    "curves": [
        {
            "run": "lr_production",
            "is_champion": True,
            "auc_holdout": 0.7468,
            "auc_cv_mean": 0.7484,
            "fpr": [0.0, 0.5, 1.0],
            "tpr": [0.0, 0.8, 1.0],
        },
        {
            "run": "mlp",
            "is_champion": False,
            "auc_holdout": 0.634,
            "auc_cv_mean": 0.63,
            "fpr": [0.0, 0.6, 1.0],
            "tpr": [0.0, 0.65, 1.0],
        },
    ],
    "threshold_available": True,
    "threshold_points": [
        {"threshold": 0.8, "sensitivity": 0.3, "specificity": 0.95},
        {"threshold": 0.5, "sensitivity": 0.68, "specificity": 0.71},
    ],
}
_IMPORTANCE = {
    "available": True,
    "variables": [
        {
            "model_variable": "age_group",
            "label_pt": "Faixa etaria",
            "source_raw": ["age"],
            "mean_abs_contribution": 0.869,
            "max_abs_coef": 2.274,
        },
        {
            "model_variable": "sex",
            "label_pt": "Sexo",
            "source_raw": ["sex"],
            "mean_abs_contribution": 0.417,
            "max_abs_coef": 0.763,
        },
    ],
    "coefficients": [
        {
            "dummy": "age_group_80+",
            "model_variable": "age_group",
            "label_pt": "Faixa etaria",
            "category": "80+",
            "coefficient": 2.274,
            "mean_abs_contribution": 0.1,
        },
        {
            "dummy": "sex_2",
            "model_variable": "sex",
            "label_pt": "Sexo",
            "category": "2",
            "coefficient": -0.763,
            "mean_abs_contribution": 0.4,
        },
    ],
}
_CONTRIBUTIONS = {
    "available": True,
    "dummies": ["age_group_80+", "sex_2"],
    "labels_pt": ["Faixa etaria = 80+", "Sexo = 2"],
    "contributions": [[2.274, -0.763], [0.0, 0.0], [2.274, 0.0]],
    "present": [[1, 1], [0, 0], [1, 0]],
    "rows": 3,
    "rows_available": 1241,
}
_SUBGROUPS = {
    "available": True,
    "rows": [
        {
            "dimension": "sex",
            "group": "feminino",
            "n": 3421,
            "auc_roc": 0.752,
            "sensitivity": 0.549,
            "specificity": 0.791,
            "precision": 0.281,
        },
        {
            "dimension": "sex",
            "group": "masculino",
            "n": 2782,
            "auc_roc": 0.719,
            "sensitivity": 0.774,
            "specificity": 0.547,
            "precision": 0.326,
        },
        {"dimension": "age_group", "group": "80+", "n": 40, "auc_roc": None},
    ],
}


def _fields(spec: dict[str, Any]) -> set[str]:
    """Collect every ``field`` a spec references, except layers with inline data."""
    found: set[str] = set()
    if "data" in spec:  # a layer carrying its own values is self-contained
        return found
    for key, value in spec.items():
        if key == "field" and isinstance(value, str):
            found.add(value)
        elif key == "sort" and isinstance(value, str):
            continue
        elif isinstance(value, dict):
            found |= _fields(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    found |= _fields(item)
    return found


def _calculated(spec: dict[str, Any]) -> set[str]:
    """Names produced by transforms, which do not have to exist in the frame."""
    return {
        str(step["as"])
        for step in spec.get("transform", [])
        if isinstance(step, dict) and "as" in step
    }


def _assert_spec_matches(spec: dict[str, Any], frame: pd.DataFrame) -> None:
    referenced = _fields(spec)
    assert referenced, "the spec references no field: the check would be vacuous"
    missing = referenced - set(frame.columns) - _calculated(spec)
    assert not missing, f"spec references columns the frame does not have: {missing}"


# ------------------------------------------------------------------ summary
def test_headline_cards_lead_with_auc_and_drop_accuracy() -> None:
    cards = model_view.headline_cards(_SUMMARY)
    assert [rotulo for rotulo, _, _ in cards] == [
        "AUC-ROC", "Sensibilidade", "Precisão", "F1"
    ]
    assert "Acurácia" not in [rotulo for rotulo, _, _ in cards]


def test_headline_cards_use_one_decimal_and_a_comma() -> None:
    cards = model_view.headline_cards(_SUMMARY)
    assert cards[0][1] == "74,7%"
    assert cards[3][1] == "43,5%"


def test_every_headline_card_carries_an_explanation() -> None:
    assert all(ajuda.strip() for _, _, ajuda in model_view.headline_cards(_SUMMARY))


def test_missing_metric_shows_a_dash_not_a_zero() -> None:
    cards = model_view.headline_cards({"holdout": {"auc_roc": 0.5}})
    assert cards[0][1] == "50,0%"
    assert cards[1][1] == "-"


def test_validation_frame_pairs_mean_with_standard_deviation() -> None:
    frame = model_view.validation_frame(_SUMMARY)
    row = frame[frame["Métrica"] == "AUC-ROC"].iloc[0]
    assert row["Validação cruzada (k = 10)"] == "0,748 ± 0,021"
    assert row["Conjunto de teste"] == "0,747"
    assert frame[frame["Métrica"] == "Acurácia"].iloc[0]["Conjunto de teste"] == "0,701"


def test_validation_frame_labels_are_spelled_in_ptbr() -> None:
    rotulos = set(model_view.validation_frame(_SUMMARY)["Métrica"])
    assert {"Precisão", "Acurácia"} <= rotulos


def test_confusion_frame_totals_close() -> None:
    frame = model_view.confusion_frame(_SUMMARY)
    assert frame.loc["Total", "Total"] == 1241
    assert frame.loc["Com hipertensão", "Sinalizado"] == 143
    assert frame.loc["Sem hipertensão", "Não sinalizado"] == 727


def test_confusion_summary_reads_as_a_sentence_with_the_trade_off() -> None:
    texto = model_view.confusion_summary(_SUMMARY)
    assert "1.241" in texto and "211" in texto
    assert "3,1 encaminhamentos" in texto


def test_confusion_summary_is_empty_when_the_cells_are_missing() -> None:
    assert model_view.confusion_summary({"holdout": {"accuracy": 0.7}}) == ""


# ---------------------------------------------------------------------- roc
def test_roc_frame_labels_carry_the_auc_and_flag_the_champion() -> None:
    frame = model_view.roc_frame(_ROC)
    assert set(frame["serie"]) == {"lr_production (AUC 0.747)", "mlp (AUC 0.634)"}
    assert frame[frame["campeao"]]["serie"].nunique() == 1
    assert len(frame) == 6


def test_roc_spec_matches_its_frame_and_keeps_the_diagonal_out_of_the_legend() -> None:
    frame = model_view.roc_frame(_ROC)
    spec = model_view.roc_spec(frame)
    _assert_spec_matches(spec, frame)
    diagonal = spec["layer"][0]
    assert diagonal["data"]["values"] == [{"x": 0, "y": 0}, {"x": 1, "y": 1}]
    assert "color" not in diagonal["encoding"]


def test_roc_colour_domain_is_fixed_so_a_run_keeps_its_hue() -> None:
    frame = model_view.roc_frame(_ROC)
    spec = model_view.roc_spec(frame)
    scale = spec["layer"][1]["encoding"]["color"]["scale"]
    assert scale["domain"] == sorted(frame["serie"].unique().tolist())
    assert len(scale["range"]) >= len(scale["domain"])


def test_threshold_spec_marks_the_threshold_in_service() -> None:
    frame = model_view.threshold_frame(_ROC)
    spec = model_view.threshold_spec(0.5)
    _assert_spec_matches(spec, frame)
    assert list(frame["metrica"].unique()) == ["Sensibilidade", "Especificidade"]
    assert spec["layer"][1]["data"]["values"] == [{"limiar": 0.5}]


# ------------------------------------------------------------- interpretation
def test_importance_frame_is_capped_and_keeps_the_origin() -> None:
    frame = model_view.importance_frame(_IMPORTANCE, top=1)
    assert len(frame) == 1
    assert frame.iloc[0]["variavel"] == "Faixa etaria"
    assert frame.iloc[0]["origem"] == "age"
    _assert_spec_matches(model_view.importance_spec(), frame)


def test_coefficients_frame_names_the_direction_of_the_effect() -> None:
    frame = model_view.coefficients_frame(_IMPORTANCE)
    positive = frame[frame["dummy"] == "age_group_80+"].iloc[0]
    negative = frame[frame["dummy"] == "sex_2"].iloc[0]
    assert positive["sentido"] == "aumenta o risco"
    assert negative["sentido"] == "reduz o risco"
    assert positive["rotulo"] == "Faixa etaria = 80+"
    _assert_spec_matches(model_view.coefficients_spec(), frame)


def test_coefficient_colours_are_diverging_around_zero() -> None:
    scale = model_view.coefficients_spec()["encoding"]["color"]["scale"]
    assert scale["domain"] == ["aumenta o risco", "reduz o risco"]
    assert scale["range"][0] != scale["range"][1]


def test_contributions_frame_expands_the_matrix_row_by_row() -> None:
    frame = model_view.contributions_frame(_CONTRIBUTIONS)
    assert len(frame) == 6
    assert set(frame["rotulo"]) == {"Faixa etaria = 80+", "Sexo = 2"}
    assert set(frame["situacao"]) == {"categoria presente", "categoria ausente"}
    first = frame.iloc[0]
    assert first["contribuicao"] == pytest.approx(2.274)
    assert first["situacao"] == "categoria presente"
    _assert_spec_matches(model_view.contributions_spec(), frame)


def test_beeswarm_jitter_is_declared_as_a_transform_not_a_column() -> None:
    spec = model_view.contributions_spec()
    assert spec["transform"][0]["as"] == "jitter"
    assert spec["encoding"]["yOffset"]["field"] == "jitter"


def test_method_note_still_says_it_is_not_shap() -> None:
    # O nome do metodo saiu do corpo da pagina, mas nao do produto: a ficha tecnica
    # continua declarando que a explicabilidade nao e SHAP (ADR 0003).
    assert "não SHAP" in model_view.METHOD_NOTE
    assert "ADR 0003" in model_view.METHOD_NOTE


def test_user_facing_disclaimer_carries_no_method_jargon() -> None:
    texto = model_view.RISK_DISCLAIMER.lower()
    assert not any(termo in texto for termo in ("shap", "adr", "log-odds", "dummy"))


# --------------------------------------------------------------------- equity
def test_subgroups_frame_translates_the_dimension_and_drops_empty_metrics() -> None:
    frame = model_view.subgroups_frame(_SUBGROUPS)
    assert set(frame["dimensao"]) == {"Sexo"}  # the age group row has no metrics
    assert set(frame["metrica"]) == {
        "AUC-ROC", "Sensibilidade", "Especificidade", "Precisão"
    }
    assert frame[frame["grupo"] == "feminino"]["n"].iloc[0] == 3421
    _assert_spec_matches(model_view.subgroups_spec(frame), frame)


def test_subgroup_scale_is_bounded_so_groups_compare() -> None:
    frame = model_view.subgroups_frame(_SUBGROUPS)
    spec = model_view.subgroups_spec(frame)
    assert spec["encoding"]["y"]["scale"]["domain"] == [0, 1]


# ----------------------------------------------------------------------- theme
def test_dark_palette_differs_from_the_light_one_slot_by_slot() -> None:
    light, dark = model_view.palette(False), model_view.palette(True)
    assert light["series"][0] != dark["series"][0]
    assert len(light["series"]) == len(dark["series"])


@pytest.mark.parametrize(
    "builder",
    [
        lambda: model_view.roc_spec(model_view.roc_frame(_ROC), True),
        lambda: model_view.threshold_spec(0.5, True),
        lambda: model_view.importance_spec(True),
        lambda: model_view.coefficients_spec(True),
        lambda: model_view.contributions_spec(True),
        lambda: model_view.subgroups_spec(model_view.subgroups_frame(_SUBGROUPS), True),
    ],
)
def test_every_spec_builds_on_the_dark_surface(builder: Callable[[], dict[str, Any]]) -> None:
    assert isinstance(builder(), dict)


@pytest.mark.parametrize(
    ("frame_builder", "spec_builder"),
    [
        (lambda: model_view.roc_frame({}), model_view.roc_spec),
        (lambda: model_view.threshold_frame({}), lambda f: model_view.threshold_spec(0.5)),
        (lambda: model_view.importance_frame({}, 5), lambda f: model_view.importance_spec()),
        (lambda: model_view.coefficients_frame({}), lambda f: model_view.coefficients_spec()),
        (lambda: model_view.contributions_frame({}), lambda f: model_view.contributions_spec()),
        (lambda: model_view.subgroups_frame({}), model_view.subgroups_spec),
    ],
)
def test_empty_payload_yields_an_empty_frame_and_a_valid_spec(
    frame_builder: Callable[[], pd.DataFrame],
    spec_builder: Callable[[pd.DataFrame], dict[str, Any]],
) -> None:
    frame = frame_builder()
    assert frame.empty
    assert isinstance(spec_builder(frame), dict)

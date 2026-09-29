"""Framework-free helpers of the Streamlit client: form spec, validation, client, panel."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pandas as pd
import pytest

from src.app.client import ApiClient, ApiUnavailableError
from src.app.explain import contributions_frame, dummy_labels, risk_band_colour
from src.app.form_spec import (
    NOT_INFORMED,
    build_steps,
    default_value,
    initial_record,
    label_of,
    option_labels,
    option_to_code,
)
from src.app.monitoring_view import drift_table, summarize
from src.app.validation import (
    check_value,
    clean_record,
    csv_template,
    frame_to_records,
    merge_results,
    read_csv,
    validate_frame,
    validate_record,
)
from src.config import Settings
from src.data.mapping import VariableMapping


# -------------------------------------------------------------------- form spec
def test_steps_cover_every_variable_once_in_four_steps(
    mapping: VariableMapping, cfg: Settings
) -> None:
    steps = build_steps(mapping, cfg.app.highlighted_variables)
    assert [s.key for s in steps] == ["socio", "anthro", "lifestyle", "lab"]
    names = [spec.name for step in steps for spec in step.fields]
    assert sorted(names) == sorted(mapping.raw_en_names) and len(names) == len(set(names))
    lab = steps[3]
    assert all(spec.nullable for spec in lab.fields)  # exams are optional (P3)
    highlighted = {spec.name for step in steps for spec in step.fields if spec.highlighted}
    assert highlighted == set(cfg.app.highlighted_variables)
    age = next(spec for spec in steps[0].fields if spec.name == "age")
    assert age.label == "Idade (anos)" and age.display_label.endswith("★")
    assert age.minimum == 18 and age.maximum == 104 and age.default == 42.0
    sex = next(spec for spec in steps[0].fields if spec.name == "sex")
    assert sex.options == {1: "Homem", 2: "Mulher"} and sex.default == 2.0


def test_defaults_and_option_helpers(mapping: VariableMapping, cfg: Settings) -> None:
    by_name = {v.en: v for v in mapping.raw_variables}
    assert default_value(by_name["income_main_job"]) is None  # nullable -> not informed
    assert default_value(by_name["weight_kg"]) == by_name["weight_kg"].observed.median
    steps = build_steps(mapping, cfg.app.highlighted_variables)
    record = initial_record(steps)
    assert set(record) == set(mapping.raw_en_names)
    assert validate_record(record, mapping) == []  # defaults are a valid record
    dx = next(spec for step in steps for spec in step.fields if spec.name == "dx_diabetes")
    labels = option_labels(dx)
    assert labels[0] == NOT_INFORMED and labels[1] == "1 - Sim"
    assert (
        option_to_code(labels[0]) is None and option_to_code("3 - Apenas durante a gravidez") == 3
    )
    assert label_of(dx, None) == NOT_INFORMED and label_of(dx, 2.0) == "2 - Não"
    weight = next(spec for step in steps for spec in step.fields if spec.name == "weight_kg")
    assert label_of(weight, 70.5) == "70.5"
    age = next(spec for step in steps for spec in step.fields if spec.name == "age")
    assert label_of(age, 42.0) == "42"


# ------------------------------------------------------------------- validation
def test_check_value_messages(mapping: VariableMapping) -> None:
    by_name = {v.en: v for v in mapping.raw_variables}
    assert check_value(by_name["age"], 42) is None
    assert check_value(by_name["age"], None) == "Idade: campo obrigatório."
    assert "fora da faixa 18 a 104 anos" in str(check_value(by_name["age"], 200))
    assert "não é numérico" in str(check_value(by_name["age"], "abc"))
    assert "inteiro" in str(check_value(by_name["age"], 42.5))
    assert "código 7 inválido" in str(check_value(by_name["sex"], 7))
    assert check_value(by_name["sex"], "2") is None
    assert check_value(by_name["cholesterol"], None) is None  # optional exam
    assert check_value(by_name["weight_kg"], "70,5") is None  # decimal comma accepted
    assert check_value(by_name["age"], True) is not None


def test_validate_and_clean_record(mapping: VariableMapping) -> None:
    record = {v.en: 1 for v in mapping.raw_variables}
    errors = validate_record(record | {"bogus": 1}, mapping)
    assert any(e.variable == "bogus" for e in errors)
    assert any(e.variable == "age" for e in errors)  # 1 < 18
    cleaned = clean_record({"age": "42", "weight_kg": "70,5", "cholesterol": float("nan")}, mapping)
    assert cleaned["age"] == 42 and isinstance(cleaned["age"], int)
    assert cleaned["weight_kg"] == 70.5 and cleaned["cholesterol"] is None
    assert set(cleaned) == set(mapping.raw_en_names)
    assert clean_record({"age": "x"}, mapping)["age"] == "x"  # left for the API


def test_csv_template_roundtrip_and_row_report(mapping: VariableMapping) -> None:
    template = csv_template(mapping)
    frame = read_csv(template.encode("utf-8"))
    assert list(frame.columns) == mapping.raw_en_names and len(frame) == 1
    validation = validate_frame(frame, mapping)
    assert validation.ok and validation.valid_rows == [1]
    # add a bad row and an unknown column, semicolon separated
    bad = frame.copy()
    bad.loc[1] = bad.loc[0]
    bad.loc[1, "age"] = "150"
    bad.loc[1, "sex"] = "9"
    bad["extra"] = "x"
    semicolon = bad.to_csv(index=False, sep=";")
    parsed = read_csv(semicolon)
    validation = validate_frame(parsed, mapping)
    assert validation.unknown_columns == ["extra"] and not validation.missing_columns
    assert validation.valid_rows == [1] and validation.invalid_rows == [2]
    report = validation.error_frame()
    assert set(report["variavel"]) == {"age", "sex"} and (report["linha"] == 2).all()
    missing = validate_frame(parsed.drop(columns=["age"]), mapping)
    assert missing.missing_columns == ["age"] and not missing.ok
    assert read_csv(b"").empty


def test_frame_to_records_and_merge_results(mapping: VariableMapping) -> None:
    frame = read_csv(csv_template(mapping))
    records = frame_to_records(frame, mapping)
    assert len(records) == 1 and records[0]["cholesterol"] is None
    prediction = {
        "probability": 0.42,
        "risk_band": "moderado",
        "predicted_class": 0,
        "imputed_fields": ["cholesterol"],
        "model_version": "3",
    }
    merged = merge_results(frame, [prediction])
    assert (
        merged.loc[0, "probabilidade"] == 0.42
        and merged.loc[0, "campos_imputados"] == "cholesterol"
    )
    with pytest.raises(ValueError):
        merge_results(frame, [])


# ----------------------------------------------------------------------- client
def _transport(handler: Any) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_client_success_paths_and_correlation_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "model_version": "1"})
        if request.url.path == "/predict":
            assert request.url.params["explain"] == "true"
            return httpx.Response(200, json={"probability": 0.1})
        if request.url.path == "/predict/batch":
            body = json.loads(request.content)
            return httpx.Response(
                200, json={"predictions": [{"i": r["i"]} for r in body["records"]]}
            )
        return httpx.Response(200, text="api_requests_total 1.0\n")

    client = ApiClient("http://api:8000/", 5.0, 2, transport=_transport(handler))
    assert client.base_url == "http://api:8000"
    assert client.health()["status"] == "ok"
    assert client.predict({"age": 42})["probability"] == 0.1
    assert client.metrics_text().startswith("api_requests_total")
    assert [r.url.path for r in seen] == ["/health", "/predict", "/metrics"]


def test_client_batches_in_chunks_and_keeps_order() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(len(body["records"]))
        return httpx.Response(200, json={"predictions": [{"i": r["i"]} for r in body["records"]]})

    client = ApiClient("http://api:8000", 5.0, 2, transport=_transport(handler))
    progress: list[tuple[int, int]] = []
    out = client.predict_batch(
        [{"i": n} for n in range(5)], progress=lambda done, total: progress.append((done, total))
    )
    assert [o["i"] for o in out] == [0, 1, 2, 3, 4]
    assert calls == [2, 2, 1] and progress == [(2, 5), (4, 5), (5, 5)]
    client.close()


def test_client_sends_correlation_header() -> None:
    headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        headers.append(request.headers.get("X-Correlation-ID"))
        return httpx.Response(200, json={"status": "ok"})

    ApiClient("http://api:8000", 5.0, 10, transport=_transport(handler)).health()
    assert headers and headers[0] and len(headers[0]) > 10


def test_client_errors_are_friendly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            raise httpx.ConnectError("refused")
        if request.url.path == "/predict":
            return httpx.Response(
                422,
                json={"error": "TransformError", "detail": "bad category", "correlation_id": "c"},
            )
        if request.url.path == "/predict/batch":
            return httpx.Response(
                422, json={"detail": [{"loc": ["body", "records", 0, "age"], "msg": "too big"}]}
            )
        if request.url.path == "/metrics":
            raise httpx.ReadTimeout("slow")
        return httpx.Response(500, text="<html>boom</html>")

    client = ApiClient("http://api:8000", 5.0, 10, transport=_transport(handler))
    with pytest.raises(ApiUnavailableError) as info:
        client.health()
    assert info.value.endpoint == "http://api:8000/health" and "conectar" in info.value.reason
    with pytest.raises(ApiUnavailableError, match="bad category") as info2:
        client.predict({})
    assert info2.value.status == 422
    with pytest.raises(ApiUnavailableError, match=r"records\.0\.age: too big"):
        client.predict_batch([{}])
    with pytest.raises(ApiUnavailableError, match="tempo limite"):
        client.metrics_text()
    with pytest.raises(ApiUnavailableError, match="HTTP 500") as info3:
        client._request("GET", "/other")
    assert "boom" not in str(info3.value)


# ---------------------------------------------------------------------- explain
def test_explain_helpers(mapping: VariableMapping, cfg: Settings) -> None:
    labels = dummy_labels(mapping)
    assert labels["age_group_60-69"] == "Faixa etária = 60-69"
    assert len(labels) == len(mapping.dummy_names)
    frame = contributions_frame(
        [
            {"feature": "age_group_60-69", "value": 1, "contribution": 0.8},
            {"feature": "sex_2", "value": 1, "contribution": -0.5},
            {"feature": "unknown_x", "value": 1, "contribution": 1.2},
        ],
        labels,
    )
    assert frame["fator"].tolist()[0] == "unknown_x"
    assert frame["direcao"].tolist() == ["aumenta o risco", "aumenta o risco", "reduz o risco"]
    assert contributions_frame([], labels).empty
    bands = cfg.api.risk_band_labels.model_dump()
    assert risk_band_colour(bands["low"], bands) == "green"
    assert risk_band_colour(bands["high"], bands) == "red"
    assert risk_band_colour(bands["moderate"], bands) == "orange"


# -------------------------------------------------------------------- monitoring
_METRICS = """
api_requests_total{endpoint="/predict",method="POST",status="200"} 80.0
api_requests_total{endpoint="/predict",method="POST",status="422"} 15.0
api_requests_total{endpoint="/predict/batch",method="POST",status="500"} 5.0
api_requests_total{endpoint="/health",method="GET",status="200"} 10.0
api_request_latency_seconds_bucket{endpoint="/predict",le="0.01",method="POST"} 40.0
api_request_latency_seconds_bucket{endpoint="/predict",le="0.05",method="POST"} 90.0
api_request_latency_seconds_bucket{endpoint="/predict",le="0.1",method="POST"} 95.0
api_request_latency_seconds_bucket{endpoint="/predict",le="+Inf",method="POST"} 95.0
api_predictions_total{risk_band="baixo"} 60.0
api_predictions_total{risk_band="alto"} 20.0
data_drift_reference_loaded 1.0
data_drift_evaluated 1.0
data_drift_window_rows 150.0
data_drift_alerts 1.0
data_drift_psi{variable="age"} 0.31
data_drift_psi{variable="sex"} 0.02
data_drift_ks_statistic{variable="age"} 0.2
data_drift_alert{variable="age"} 1.0
data_drift_alert{variable="sex"} 0.0
"""


def test_summarize_panel(cfg: Settings) -> None:
    summary = summarize(_METRICS, cfg.monitoring.drift)
    assert summary.total_requests == 110 and summary.prediction_requests == 100
    assert summary.error_rate == pytest.approx(0.05)
    assert summary.rejection_rate == pytest.approx(0.15)
    assert summary.quality_gate == "atenção"
    # rank 0.95 * 95 = 90.25 falls in (0.05, 0.1] with 5 observations: 0.05 + 0.05 * 0.25 / 5
    assert summary.p95_ms == pytest.approx(52.5)
    assert summary.predictions_by_band == {"baixo": 60.0, "alto": 20.0}
    assert summary.drift_reference_loaded and summary.drift_evaluated
    assert summary.drift_window_rows == 150 and summary.drift_alerts == 1
    assert summary.drift.iloc[0]["variavel"] == "age" and bool(summary.drift.iloc[0]["alerta"])
    assert summary.drift.iloc[0]["nivel_psi"] == "alert" and pd.isna(summary.drift.iloc[1]["ks"])
    empty = summarize("", cfg.monitoring.drift)
    assert empty.quality_gate == "sem dados" and empty.p95_ms is None and empty.drift.empty
    assert drift_table([], cfg.monitoring.drift).columns.tolist() == [
        "variavel",
        "psi",
        "nivel_psi",
        "ks",
        "alerta",
    ]
    assert isinstance(summary.drift, pd.DataFrame)

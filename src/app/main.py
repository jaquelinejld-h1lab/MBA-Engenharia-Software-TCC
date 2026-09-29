"""Streamlit client of the hypertension risk API.

Run locally (with the API on ``app.api_url``)::

    streamlit run src/app/main.py

Pages: individual prediction (four-step wizard with review and local
explanation), batch prediction (CSV template, validation, progress,
download), model (global performance of the champion, its ROC against the
other runs, what it uses and how it behaves by subgroup) and monitoring
panel (RED metrics, drift, quality gate, model version). The model page is
static, read from artifacts of the training; the monitoring panel is live.
The app is a pure HTTP client: it never imports the model, the
transformer or MLflow, and it keeps its state in ``st.session_state``.
Every helper that does not need Streamlit lives in the sibling modules
and is unit tested; this file only draws widgets.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ``streamlit run`` executes this file as a script: make ``src`` importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import __version__
from src.app import model_view
from src.app.client import ApiClient, ApiUnavailableError
from src.app.explain import contributions_frame, dummy_labels, risk_band_colour
from src.app.form_spec import (
    NOT_INFORMED,
    FieldSpec,
    StepSpec,
    build_steps,
    initial_record,
    label_of,
    option_labels,
    option_to_code,
)
from src.app.monitoring_view import prometheus_p95_series, summarize
from src.app.validation import (
    clean_record,
    csv_template,
    frame_to_records,
    merge_results,
    read_csv,
    validate_frame,
    validate_record,
)
from src.config import Settings, get_config
from src.data.mapping import VariableMapping, get_mapping

REVIEW_STEP = 4
_PROMETHEUS_WINDOW = "5m"
_PROMETHEUS_SPAN_MINUTES = 30
_PROMETHEUS_STEP_SECONDS = 15


# ------------------------------------------------------------------ resources
@st.cache_resource
def _settings() -> Settings:
    return get_config()


@st.cache_resource
def _mapping() -> VariableMapping:
    return get_mapping()


@st.cache_resource
def _client() -> ApiClient:
    cfg = _settings()
    token = cfg.app.api_token.get_secret_value() if cfg.app.api_token else None
    return ApiClient(
        cfg.app.api_url,
        cfg.app.request_timeout_seconds,
        cfg.app.batch_chunk_size,
        token=token,
    )


@st.cache_data(ttl=300)
def _report(section: str) -> dict[str, Any]:
    """One section of the global model report, cached: it only changes on a new deploy."""
    return dict(_client().model_report(section))


@st.cache_data
def _steps() -> list[StepSpec]:
    return build_steps(_mapping(), _settings().app.highlighted_variables)


@st.cache_data
def _labels() -> dict[str, str]:
    return dummy_labels(_mapping())


def _api_error(error: ApiUnavailableError) -> None:
    """Friendly message with the endpoint; never a stack trace."""
    st.error(
        f"A API não respondeu como esperado em `{error.endpoint}`: {error.reason}. "
        "Verifique se o serviço está no ar (docker compose up) e o endereço em "
        "`app.api_url` ou na variável HTN_APP__API_URL."
    )


def _health_banner() -> dict[str, Any] | None:
    try:
        health = _client().health()
    except ApiUnavailableError as error:
        _api_error(error)
        return None
    st.sidebar.success(
        f"Serviço disponível. Modelo {health.get('model_name')} "
        f"versão {health.get('model_version')}."
    )
    return dict(health)


# ------------------------------------------------------------ individual page
def _widget(spec: FieldSpec, current: float | None) -> float | None:
    key = f"f_{spec.name}"
    if spec.kind == "categorical":
        labels = option_labels(spec)
        default = (
            NOT_INFORMED
            if current is None
            else f"{int(current)} - {spec.options.get(int(current), '?')}"
        )
        index = labels.index(default) if default in labels else 0
        choice = st.selectbox(spec.display_label, labels, index=index, key=key, help=spec.help)
        code = option_to_code(choice)
        return None if code is None else float(code)
    # Streamlit requires value, min_value, max_value and step to share one numeric type
    # and raises StreamlitMixedNumericTypesError otherwise, so cast all four by dtype.
    cast: Callable[[float], float] = int if spec.dtype == "int" else float
    value = st.number_input(
        spec.display_label,
        min_value=None if spec.minimum is None else cast(spec.minimum),
        max_value=None if spec.maximum is None else cast(spec.maximum),
        value=None if current is None else cast(current),
        step=cast(spec.step),
        key=key,
        help=spec.help,
        placeholder=NOT_INFORMED if spec.nullable else "obrigatório",
    )
    return None if value is None else float(value)


def _step_form(step: StepSpec, index: int) -> None:
    record: dict[str, float | None] = st.session_state["record"]
    st.subheader(step.title)
    st.caption(step.description + " Campos com ★ têm maior peso no modelo.")
    with st.form(key=f"form_{step.key}"):
        values: dict[str, float | None] = {}
        columns = st.columns(2)
        for position, spec in enumerate(step.fields):
            with columns[position % 2]:
                values[spec.name] = _widget(spec, record.get(spec.name))
        back, forward = st.columns(2)
        go_back = back.form_submit_button("Voltar", disabled=index == 0, use_container_width=True)
        go_next = forward.form_submit_button("Próximo", type="primary", use_container_width=True)
    if go_back:
        record.update(values)
        st.session_state["step"] = index - 1
        st.rerun()
    if go_next:
        record.update(values)
        mapping = _mapping()
        step_names = {spec.name for spec in step.fields}
        errors = [e for e in validate_record(record, mapping) if e.variable in step_names]
        if errors:
            for error in errors:
                st.error(error.message)
            return
        st.session_state["step"] = index + 1
        st.rerun()


def _review(steps: list[StepSpec]) -> None:
    record: dict[str, float | None] = st.session_state["record"]
    st.subheader("Revisão dos dados informados")
    for index, step in enumerate(steps):
        with st.expander(step.title, expanded=True):
            table = pd.DataFrame(
                {
                    "Variável": [spec.display_label for spec in step.fields],
                    "Valor": [label_of(spec, record.get(spec.name)) for spec in step.fields],
                }
            )
            st.dataframe(table, hide_index=True, use_container_width=True)
            if st.button("Editar esta etapa", key=f"edit_{step.key}"):
                st.session_state["step"] = index
                st.rerun()
    errors = validate_record(record, _mapping())
    if errors:
        st.error("Há campos inválidos; volte às etapas indicadas.")
        for error in errors:
            st.write(f"- {error.message}")
        return
    if st.button("Calcular risco", type="primary"):
        try:
            payload = clean_record(record, _mapping())
            st.session_state["result"] = _client().predict(payload, explain=True)
        except ApiUnavailableError as error:
            _api_error(error)
            return
        st.session_state["step"] = REVIEW_STEP + 1
        st.rerun()


def _result() -> None:
    cfg = _settings()
    result: dict[str, Any] = st.session_state["result"]
    st.subheader("Resultado")
    band = str(result["risk_band"])
    colour = risk_band_colour(band, cfg.api.risk_band_labels.model_dump())
    left, middle, right = st.columns(3)
    left.metric("Probabilidade estimada", f"{100 * float(result['probability']):.1f} %")
    middle.markdown(f"**Faixa de risco:** :{colour}[{band}]")
    right.metric("Limiar de decisão", f"{float(result['threshold']):.2f}")
    st.caption(
        f"Modelo {result['model_name']} versão {result['model_version']} "
        f"({result['model_stage']}); validação cruzada com k = {cfg.split.n_splits}; "
        f"faixas: baixo < {cfg.model.risk_bands.low_below:.2f}, "
        f"alto ≥ {cfg.model.risk_bands.high_from:.2f}. Correlação: {result['correlation_id']}."
    )
    imputed = list(result.get("imputed_fields") or [])
    if imputed:
        st.warning(
            "Exames laboratoriais ausentes foram imputados pelo KNN treinado: "
            + ", ".join(imputed)
            + ". A predição depende dessa imputação; informe os exames quando disponíveis."
        )
    contributions = result.get("contributions") or []
    st.markdown(f"**Cinco fatores mais influentes (top {cfg.app.top_contributions})**")
    frame = contributions_frame(contributions, _labels())
    if frame.empty:
        st.info("O modelo em serviço não fornece contribuições por variável.")
    else:
        chart = frame.set_index("fator")[["contribuicao"]]
        st.bar_chart(chart, horizontal=True)
        st.dataframe(frame, hide_index=True, use_container_width=True)
        st.caption(
            "Contribuições aditivas em log-odds (coeficiente x variável ativa); "
            "valores positivos aumentam o risco."
        )
    st.warning(f"**Apoio à decisão, não diagnóstico.** {result['disclaimer']}")
    if st.button("Nova predição"):
        for key in ("record", "result"):
            st.session_state.pop(key, None)
        st.session_state["step"] = 0
        st.rerun()


def page_individual() -> None:
    steps = _steps()
    if "record" not in st.session_state:
        st.session_state["record"] = initial_record(steps)
        st.session_state["step"] = 0
    step_index = int(st.session_state["step"])
    stage = min(step_index, REVIEW_STEP)
    st.progress(stage / REVIEW_STEP, text=f"Etapa {min(stage + 1, 5)} de 5")
    if step_index < len(steps):
        _step_form(steps[step_index], step_index)
    elif step_index == REVIEW_STEP:
        _review(steps)
    else:
        _result()


# ----------------------------------------------------------------- batch page
def page_batch() -> None:
    mapping = _mapping()
    st.subheader("Predição em lote")
    st.download_button(
        "Baixar template CSV",
        data=csv_template(mapping),
        file_name="template_predicao.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("CSV com as 61 variáveis (uma linha por pessoa)", type=["csv"])
    if uploaded is None:
        return
    frame = read_csv(uploaded.getvalue())
    if frame.empty:
        st.error("Arquivo vazio.")
        return
    validation = validate_frame(frame, mapping)
    if validation.missing_columns:
        st.error("Colunas ausentes: " + ", ".join(validation.missing_columns))
        return
    if validation.unknown_columns:
        st.warning("Colunas ignoradas: " + ", ".join(validation.unknown_columns))
    st.write(
        f"{validation.total_rows} linhas lidas, {len(validation.valid_rows)} válidas, "
        f"{len(validation.invalid_rows)} com erro."
    )
    if validation.errors:
        report = validation.error_frame()
        st.dataframe(report, hide_index=True, use_container_width=True)
        st.download_button(
            "Baixar relatório de erros",
            data=report.to_csv(index=False),
            file_name="erros_validacao.csv",
            mime="text/csv",
        )
    if not validation.valid_rows:
        return
    if not st.button("Escorar linhas válidas", type="primary"):
        return
    valid_frame = frame.iloc[[row - 1 for row in validation.valid_rows]]
    records = frame_to_records(valid_frame, mapping)
    bar = st.progress(0.0, text="Enviando à API...")
    try:
        predictions = _client().predict_batch(
            records, progress=lambda done, total: bar.progress(done / total, text=f"{done}/{total}")
        )
    except ApiUnavailableError as error:
        _api_error(error)
        return
    bar.progress(1.0, text="Concluído")
    results = merge_results(valid_frame, predictions)
    results.insert(0, "linha", validation.valid_rows)
    n_imputed = int((results["campos_imputados"] != "").sum())
    if n_imputed:
        st.warning(
            f"{n_imputed} linha(s) tiveram exames laboratoriais imputados "
            "(coluna campos_imputados)."
        )
    st.dataframe(results, hide_index=True, use_container_width=True)
    st.download_button(
        "Baixar resultado",
        data=results.to_csv(index=False),
        file_name="resultado_predicao.csv",
        mime="text/csv",
    )
    st.warning(f"**Apoio à decisão, não diagnóstico.** {_settings().api.clinical_disclaimer}")


# ------------------------------------------------------------ monitoring page
def page_monitoring(health: dict[str, Any] | None) -> None:
    cfg = _settings()
    st.subheader("Painel de monitoramento")
    try:
        text = _client().metrics_text()
    except ApiUnavailableError as error:
        _api_error(error)
        return
    summary = summarize(text, cfg.monitoring.drift)
    a, b, c, d = st.columns(4)
    a.metric(
        "p95 de latência (/predict)",
        "n/d" if summary.p95_ms is None else f"{summary.p95_ms:.1f} ms",
        help=(
            f"Meta local: {cfg.acceptance.max_p95_latency_ms} ms; "
            f"SLO: {cfg.acceptance.slo_p95_latency_ms} ms."
        ),
    )
    b.metric("Requisições de predição", f"{int(summary.prediction_requests)}")
    c.metric("Taxa de erro (5xx)", f"{100 * summary.error_rate:.2f} %")
    d.metric(
        "Gate de qualidade (contrato)",
        summary.quality_gate,
        help="Rejeições 4xx sobre as requisições de predição.",
    )
    if summary.p95_ms is not None and summary.p95_ms > cfg.acceptance.max_p95_latency_ms:
        st.warning("p95 acima da meta local.")
    if health is not None:
        st.info(
            f"Modelo ativo: {health.get('model_name')} versão {health.get('model_version')} "
            f"({health.get('model_stage')}, origem {health.get('model_source')}); "
            f"API {health.get('package_version')}."
        )
    if summary.predictions_by_band:
        st.markdown("**Predições por faixa de risco**")
        st.bar_chart(pd.Series(summary.predictions_by_band, name="predições"))
    series = prometheus_p95_series(
        cfg.app.prometheus_url,
        window=_PROMETHEUS_WINDOW,
        span_minutes=_PROMETHEUS_SPAN_MINUTES,
        step_seconds=_PROMETHEUS_STEP_SECONDS,
        timeout=cfg.app.request_timeout_seconds,
    )
    if series is None:
        st.caption(
            f"Prometheus indisponível em {cfg.app.prometheus_url}; "
            "p95 acima é do ciclo de vida do processo."
        )
    elif not series.empty:
        st.markdown(
            f"**p95 (janela {_PROMETHEUS_WINDOW}) nos últimos {_PROMETHEUS_SPAN_MINUTES} min**"
        )
        st.line_chart(series.set_index("timestamp"))
    st.markdown("**Drift de dados (PSI e KS por variável)**")
    if not summary.drift_reference_loaded:
        st.error("Janela de referência do treino não carregada na API.")
    elif not summary.drift_evaluated:
        st.info(
            f"Janela atual com {summary.drift_window_rows} linhas; PSI e KS exigem "
            f"{cfg.monitoring.min_rows_for_drift}."
        )
    else:
        if summary.drift_alerts:
            st.error(f"{summary.drift_alerts} variável(is) em alerta de drift.")
        else:
            st.success("Nenhuma variável em alerta de drift.")
        st.caption(
            f"Janela: {summary.drift_window_rows} linhas. Limiares: PSI aviso "
            f"{cfg.monitoring.drift.psi_warning}, alerta {cfg.monitoring.drift.psi_alert}; "
            f"KS alerta {cfg.monitoring.drift.ks_statistic_alert} "
            f"com p < {cfg.monitoring.drift.ks_pvalue_alert}."
        )
        table = summary.drift
        highlighted = table.style.apply(
            lambda row: ["background-color: #ffcccc" if row["alerta"] else ""] * len(row), axis=1
        )
        st.dataframe(highlighted, hide_index=True, use_container_width=True)


# ------------------------------------------------------------------ model page
def _is_dark() -> bool:
    """Whether the active Streamlit theme uses a dark surface."""
    return str(st.get_option("theme.base") or "light").lower() == "dark"


def _unavailable(payload: dict[str, Any], what: str) -> bool:
    """Show why a section is missing and report whether the caller should stop."""
    if payload.get("available"):
        return False
    st.info(
        f"{what} indisponível: {payload.get('reason') or 'artefato ausente'}. "
        "Execute o treino novamente para gerar os artefatos da versão atual."
    )
    return True


def _metric_cards(cards: list[tuple[str, str, str]]) -> None:
    """Render headline metrics, two per row on narrow screens.

    Streamlit columns do not reflow, so four side by side become illegible below
    roughly 900 px. Two rows of two keep every card readable at any width.
    """
    for inicio in (0, 2):
        linha = cards[inicio: inicio + 2]
        if not linha:
            continue
        for coluna, (rotulo, valor, ajuda) in zip(st.columns(2), linha, strict=False):
            coluna.metric(rotulo, valor, help=ajuda)


def _model_performance(summary: dict[str, Any], dark: bool) -> None:
    if not _unavailable(summary, "Métricas do modelo"):
        st.subheader("Desempenho")
        _metric_cards(model_view.headline_cards(summary))
        resumo = model_view.confusion_summary(summary)
        if resumo:
            st.caption(resumo)

        st.markdown("###### Todas as métricas")
        st.dataframe(
            model_view.validation_frame(summary), hide_index=True, use_container_width=True
        )
        st.caption(
            f"A validação cruzada divide os dados em {summary.get('n_folds', 0)} partes e "
            "repete a avaliação em cada uma; o valor após o sinal ± mostra o quanto a "
            "métrica varia entre elas. A coluna do conjunto de teste usa dados que o "
            "modelo nunca viu."
        )
        # A matriz de confusão é leitura técnica e fica recolhida: o parágrafo acima já
        # entrega o compromisso que ela codifica. Em largura total, não corta na tela.
        with st.expander("Ver a matriz de confusão"):
            st.dataframe(model_view.confusion_frame(summary), use_container_width=True)
            st.caption(
                "Linhas: condição real. Colunas: decisão do modelo no limiar em serviço."
            )

    roc = _report("roc")
    st.subheader("Comparação com os demais modelos avaliados")
    if not _unavailable(roc, "Curvas ROC"):
        frame = model_view.roc_frame(roc)
        st.vega_lite_chart(
            frame, model_view.roc_spec(frame, dark), use_container_width=True, theme=None
        )
        st.caption(
            "Quanto mais a curva se afasta da diagonal, melhor o modelo separa quem tem "
            "de quem não tem hipertensão. O modelo em serviço aparece com traço mais "
            "espesso."
        )
    if roc.get("threshold_available"):
        st.subheader("Efeito do ponto de corte")
        st.vega_lite_chart(
            model_view.threshold_frame(roc),
            model_view.threshold_spec(float(summary.get("decision_threshold", 0.5)), dark),
            use_container_width=True,
            theme=None,
        )
        st.caption(
            "A linha tracejada marca o ponto de corte em uso. Baixá-lo sinaliza mais "
            "pessoas e encontra mais casos, ao custo de mais encaminhamentos "
            "desnecessários."
        )


def _model_interpretation(dark: bool) -> None:
    importance = _report("importance")
    if not _unavailable(importance, "Interpretação do modelo"):
        st.subheader("Quais características mais pesam")
        frame = model_view.importance_frame(importance, top=15)
        st.vega_lite_chart(
            frame, model_view.importance_spec(dark), use_container_width=True, theme=None
        )
        st.caption(
            "Peso médio de cada característica no risco estimado, somando todas as suas "
            "categorias. Peso alto indica que a característica costuma mover a estimativa, "
            "não que ela cause hipertensão."
        )

        st.subheader("Em que sentido cada categoria pesa")
        st.vega_lite_chart(
            model_view.coefficients_frame(importance),
            model_view.coefficients_spec(dark),
            use_container_width=True,
            theme=None,
        )
        st.caption(
            "Barras à direita aumentam o risco estimado; à esquerda, reduzem. A comparação "
            "é sempre contra a categoria de referência da mesma característica."
        )

    contributions = _report("contributions")
    if not _unavailable(contributions, "Contribuições"):
        with st.expander("Como o efeito varia de pessoa para pessoa"):
            st.vega_lite_chart(
                model_view.contributions_frame(contributions),
                model_view.contributions_spec(dark),
                use_container_width=True,
                theme=None,
            )
            st.caption(model_view.RISK_DISCLAIMER)


def _model_equity(dark: bool) -> None:
    subgroups = _report("subgroups")
    st.subheader("Desempenho por subgrupo")
    if _unavailable(subgroups, "Métricas por subgrupo"):
        return
    frame = model_view.subgroups_frame(subgroups)
    for dimension in frame["dimensao"].unique():
        st.markdown(f"**{dimension}**")
        subset = frame[frame["dimensao"] == dimension]
        st.vega_lite_chart(
            subset, model_view.subgroups_spec(subset, dark), use_container_width=False, theme=None
        )
    st.caption(
        "Desempenho medido separadamente em cada grupo. Diferenças entre grupos são uma "
        "limitação conhecida deste modelo e devem ser levadas em conta na leitura do "
        "resultado individual."
    )


def page_model() -> None:
    dark = _is_dark()
    try:
        summary = _report("summary")
    except ApiUnavailableError as error:
        _api_error(error)
        return
    st.subheader("Sobre o modelo em uso")
    st.caption(
        "Estimativa de risco de hipertensão arterial a partir de características "
        "socioeconômicas, de estilo de vida e de saúde declarada. Apoio à triagem, "
        "não diagnóstico."
    )
    with st.expander("Ficha técnica"):
        st.markdown(
            f"- Modelo: **{summary.get('model_name') or 'hypertension-lr'}**, versão "
            f"**{summary.get('model_version') or '-'}**, em "
            f"**{summary.get('model_stage') or '-'}**\n"
            f"- Execução de origem: `{summary.get('model_run') or '-'}`\n"
            f"- Ponto de corte em uso: {summary.get('decision_threshold')}\n"
            f"- Avaliação: {int((summary.get('holdout') or {}).get('n', 0))} pessoas no "
            "conjunto de teste\n"
            f"- Base: Pesquisa Nacional de Saúde 2013 (IBGE), modelo de Dias (2024)\n"
            f"- Interface v{__version__}, servida pela API do modelo"
        )
        st.caption(model_view.METHOD_NOTE)
    performance, interpretation, equity = st.tabs(
        ["Desempenho", "O que o modelo usa", "Equidade"]
    )
    try:
        with performance:
            _model_performance(summary, dark)
        with interpretation:
            _model_interpretation(dark)
        with equity:
            _model_equity(dark)
    except ApiUnavailableError as error:
        _api_error(error)


# ----------------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(page_title="Risco de hipertensão", page_icon="🩺", layout="wide")
    st.title("Risco de hipertensão arterial")
    st.caption("Ferramenta de apoio à triagem. Não substitui avaliação clínica.")
    health = _health_banner()
    page = st.sidebar.radio(
        "Página",
        ["Predição individual", "Predição em lote", "Modelo", "Monitoramento"],
        key="page",
    )
    if page == "Predição individual":
        page_individual()
    elif page == "Predição em lote":
        page_batch()
    elif page == "Modelo":
        page_model()
    else:
        page_monitoring(health)
    st.sidebar.caption(_settings().api.clinical_disclaimer)


if __name__ == "__main__":
    main()

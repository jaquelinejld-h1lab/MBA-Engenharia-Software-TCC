"""Generate ``docs/dicionario_dados.md`` from ``configs/variable_mapping.yaml`` and ``config.yaml``.

The dictionary is derived, never hand-edited: regenerate it whenever the
mapping or the data block of the configuration changes::

    python scripts/make_data_dictionary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config
from src.data.mapping import RawVariable, get_mapping
from src.paths import PROJECT_ROOT

OUTPUT = PROJECT_ROOT / "docs" / "dicionario_dados.md"

BLOCKS = {
    "socio": "Sociodemográfico",
    "anthro": "Antropometria",
    "lifestyle": "Comportamento",
    "health": "Saúde",
    "lab": "Exames laboratoriais",
}


def _domain(variable: RawVariable) -> str:
    domain = variable.domain
    if domain.type == "categorical":
        labels = variable.value_labels_pt or {}
        return "; ".join(f"{code} = {labels.get(code, '?')}" for code in domain.values or [])
    unit = f" {variable.unit}" if variable.unit else ""
    return f"{domain.min:g} a {domain.max:g}{unit}"


def _observed(variable: RawVariable) -> str:
    observed = variable.observed
    if variable.domain.type == "categorical":
        return f"moda {observed.mode:g}" if observed.mode is not None else ""
    parts = []
    if observed.median is not None:
        parts.append(f"mediana {observed.median:g}")
    if observed.min is not None and observed.max is not None:
        parts.append(f"[{observed.min:g}, {observed.max:g}]")
    return ", ".join(parts)


def build() -> str:
    """Render the Markdown document."""
    cfg = get_config()
    mapping = get_mapping()
    data = cfg.data
    lines = [
        "# Dicionário de dados",
        "",
        "Gerado por `scripts/make_data_dictionary.py` a partir de `configs/variable_mapping.yaml`",
        f"(versão {mapping.version}) e de `configs/config.yaml`. Não editar à mão.",
        "",
        "## 1. Origem, licença e congelamento",
        "",
        "| Item | Valor |",
        "|---|---|",
        "| Fonte | Pesquisa Nacional de Saúde 2013 (PNS 2013), IBGE em parceria com o Ministério da Saúde; módulo de exames laboratoriais (subamostra com coleta de sangue e urina) |",
        "| Natureza | Dados secundários, públicos e anonimizados (microdados sem identificação do domicílio, do morador ou do município; ver `docs/checklist_privacidade.md`) |",
        "| Período de coleta | Entrevistas de agosto de 2013 a fevereiro de 2014; exames laboratoriais coletados na subamostra no mesmo período. `[INCERTEZA]` conferir as datas exatas na documentação da PNS 2013 e do módulo laboratorial (IBGE, 2014; Szwarcwald et al., 2019) |",
        "| Licença | Microdados de uso público, disponibilizados pelo IBGE para livre acesso com citação da fonte. `[INCERTEZA]` transcrever o termo de uso vigente do portal do IBGE na versão final do TCC |",
        "| Atribuição | IBGE, Pesquisa Nacional de Saúde 2013, microdados do módulo laboratorial |",
        f"| Arquivo | `{cfg.paths.raw_dataset}` (aba `{data.raw_sheet}`) |",
        f"| Data freeze | Extrato de {Path(str(cfg.paths.raw_dataset)).stem.split('_')[-1]} (data no nome do arquivo, formato DDMMAAAA), congelado no repositório; qualquer alteração falha o gate de qualidade pelo hash |",
        f"| SHA-256 | `{data.raw_sha256}` |",
        f"| Dimensão bruta | {data.expected_raw_rows} linhas × {data.expected_raw_columns} colunas |",
        f"| Conjunto de desenvolvimento | {data.expected_rows_after_filters} linhas após os seis filtros populacionais (tabela abaixo); prevalência do alvo 17,04 % |",
        "| Peso amostral | `PESO_LAB` presente e não utilizado (ADR 0009); as métricas valem para a amostra analítica |",
        "",
        "### Filtros populacionais e alvo",
        "",
        "| Ordem | Regra | Coluna PNS | Removidos |",
        "|---|---|---|---:|",
    ]
    filters = [
        ("Consentiu armazenar exames (`Z051 == 1`)", "Z051"),
        ("Não grávida nem sem resposta (`P005 not in {1, 3}`)", "P005"),
        ("Sem hipertensão exclusiva da gravidez (`Q002 != 2`)", "Q002"),
        ("Sem medicação para hipertensão nas duas semanas (`Q006 != 1`)", "Q006"),
        ("Região informada", "REGIAO"),
        ("Questionário alimentar respondido (`P006` não nulo)", "P006"),
    ]
    for order, ((rule, column), removed) in enumerate(
        zip(filters, cfg.population_filters.expected_removed, strict=True), start=1
    ):
        lines.append(f"| {order} | {rule} | {column} | {removed} |")
    lines += [
        "",
        f"Alvo: `target = 1` se PAS (`W00407`) ≥ {cfg.target.systolic_threshold_mmhg:g} mmHg "
        f"ou PAD (`W00408`) ≥ {cfg.target.diastolic_threshold_mmhg:g} mmHg; registros sem "
        "pressão arterial são excluídos antes do alvo (nenhum sobrevive aos filtros).",
        "",
        "## 2. Variáveis auxiliares (filtro e alvo, fora do caminho de inferência)",
        "",
        "| PNS | Nome | Papel | Domínio |",
        "|---|---|---|---|",
    ]
    for aux in mapping.auxiliary_variables:
        domain = aux.domain
        dom = (
            ", ".join(map(str, domain.values or []))
            if domain.type == "categorical"
            else f"{domain.min:g} a {domain.max:g}"
        )
        lines.append(f"| {aux.pns} | `{aux.en}` ({aux.pt}) | {aux.role} | {dom} |")
    lines += [
        "",
        f"## 3. As {len(mapping.raw_variables)} variáveis brutas do caminho de inferência",
        "",
        "Nome em inglês é o único usado no código; `pt` é o nome do script original; "
        "`nulos` é a fração de ausentes observada no desenvolvimento e o máximo aceito "
        "pelo gate de qualidade.",
        "",
    ]
    for block, title in BLOCKS.items():
        variables = [v for v in mapping.raw_variables if v.block == block]
        lines += [
            f"### 3.{list(BLOCKS).index(block) + 1} {title} ({len(variables)})",
            "",
            "| PNS | Nome (en) | Nome original (pt) | Rótulo | Tipo | Domínio | Nulos obs. / máx. | Observado no treino |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for v in variables:
            lines.append(
                f"| {v.pns} | `{v.en}` | `{v.pt}` | {v.label_pt} | {v.dtype}"
                f"{' (opcional)' if v.nullable else ''} | {_domain(v)} | "
                f"{v.null_share_observed:.1%} / {v.null_share_max:.1%} | {_observed(v)} |"
            )
        lines.append("")
    lines += [
        f"## 4. As {len(mapping.model_variables)} variáveis do modelo "
        f"({len(mapping.dummy_names)} dummies com `drop='first'`)",
        "",
        "Todas categóricas em string; a primeira categoria é a referência (dummy omitida). "
        "As regras de derivação estão em `src/features/derivations.py` com os parâmetros "
        "em `configs/config.yaml::features`.",
        "",
        "| Nome (en) | Nome original (pt) | Rótulo | Categorias (referência primeiro) | Variáveis brutas de origem |",
        "|---|---|---|---|---|",
    ]
    for mv in mapping.model_variables:
        lines.append(
            f"| `{mv.en}` | `{mv.pt}` | {mv.label_pt} | {', '.join(mv.categories)} | "
            f"{', '.join(f'`{s}`' for s in mv.source_raw)} |"
        )
    lines += [
        "",
        "### Variáveis intermediárias (base da imputação KNN, não entram no modelo)",
        "",
        "| Nome (en) | Nome original (pt) | Categorias | Nota |",
        "|---|---|---|---|",
    ]
    for iv in mapping.intermediate_variables:
        lines.append(f"| `{iv.en}` | `{iv.pt}` | {', '.join(iv.categories)} | {iv.note} |")
    lab = cfg.features.laboratory
    lines += [
        "",
        "## 5. Regras com efeito semântico (herdadas do modelo original)",
        "",
        "| Regra | Efeito | Referência |",
        "|---|---|---|",
        "| Renda ausente conta como zero (D7) | Parte da faixa de renda 1 é ausência de resposta, não renda baixa | inventário 6.2 |",
        "| `fumante_hist` compara string com inteiro (D4) | Fumantes diários sem resposta em P051/P052 ficam sem histórico | ADR 0001 e 0007 |",
        f"| Exames ausentes imputados por KNN (k = {lab.knn_imputer_neighbors}) sobre {len(lab.imputation_base_variables)} variáveis | Predição possível sem exames, com aviso (`imputed_fields`) | ADR 0011 |",
        "| Códigos 4 em tipo de refrigerante e de leite significam ausência de resposta (D10) | Só surgem de valor nulo; códigos fora de 1 a 3 são rejeitados | `derivations.py` |",
        "| Peso amostral ignorado (D6) | Métricas não ponderadas | ADR 0009 |",
        "",
        "## 6. Retenção e descarte",
        "",
        "Ver `docs/checklist_privacidade.md`: o extrato congelado permanece no repositório "
        "enquanto o modelo for mantido; dados enviados à API não são persistidos em disco.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"written {OUTPUT}")

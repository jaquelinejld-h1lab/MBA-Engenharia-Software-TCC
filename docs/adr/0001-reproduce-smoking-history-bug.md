# ADR 0001: reproduzir o bug de `fumante_hist` no transformador de produção

| Campo | Valor |
|---|---|
| Status | Aceito (decisão P1 da Fase 0) |
| Data | 2026-09-14 |
| Referências | `docs/00_inventario_artefatos_originais.md`, item D4; `src/features/derivations.py::smoking_history`; `configs/config.yaml` chave `features.smoking.reproduce_string_int_comparison_bug` |

## Contexto

Em `funcoes_mecai_v3.py`, a variável intermediária `fumante` recebe as strings `'1'`
e `'2'` via `Series.replace`, e a variável do modelo `fumante_hist` é construída com a
expressão `row['fumante'] == 1 or row['fumo_rotina_hist'] == 1 or row['fumo_hist'] == 1`.
A comparação `'1' == 1` é sempre falsa em Python. Consequência: os 619 fumantes
diários do conjunto de treino (12,5 %) cujas variáveis de histórico (P051 e P052)
estão ausentes são classificados como "sem histórico de tabagismo". A variável do
modelo campeão, portanto, não mede o que o nome indica.

## Decisão

Reproduzir o comportamento defeituoso no pipeline refatorado, sob a chave de
configuração `reproduce_string_int_comparison_bug: true`, com teste unitário que
documenta o caso (fumante diário sem histórico registrado → `"2"`).

A versão corrigida existe no mesmo módulo, ativada por `false`, e será treinada como
run de comparação no MLflow na etapa 3, sem promoção ao registry.

## Alternativas consideradas

| Alternativa | Motivo do descarte |
|---|---|
| Corrigir o bug no transformador promovido | Viola a regra de fidelidade do projeto: a matriz de desenho deixaria de ser idêntica à da dissertação e as métricas de referência perderiam a comparabilidade. O objeto do TCC é a produtização, não a remodelagem |
| Remover a variável do modelo | Mesma violação, com perda adicional de rastreabilidade com a Tabela 4 da dissertação |

## Consequências

Positivas: matriz codificada idêntica à do notebook, verificada por hash em
`tests/integration/test_notebook_fidelity.py`; a correção fica a uma chave de
configuração de distância e é medida como run separada.

Negativas: o model card precisa declarar que `smoking_history` subestima a
prevalência de histórico de tabagismo; a interface deve rotular a variável de
forma compatível com o que ela de fato mede.

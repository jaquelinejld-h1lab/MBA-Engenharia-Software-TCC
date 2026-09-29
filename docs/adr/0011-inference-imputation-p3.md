# ADR 0011: decisão P3, imputação de exames ausentes em inferência

| Campo | Valor |
|---|---|
| Status | Aceito |
| Data | 2026-09-15 |
| Referências | `src/features/transformer.py::_complete`, `PredictionResponse.imputed_fields`, inventário item D3 |

## Contexto

O modelo depende de três exames laboratoriais (eGFR, colesterol, glicose) que o
notebook imputa por KNN ajustado no próprio conjunto. Em inferência individual não há
vizinhos, e a interface exige que a predição sempre possa rodar.

## Decisão

Os três exames são opcionais na API. O `KNNImputer` (k = 5) ajustado no treino, sobre a
base one-hot de doze variáveis declaradas em `features.laboratory.imputation_base_variables`,
é persistido com o transformador e aplicado à observação. A resposta carrega
`imputed_fields` e a interface exibe aviso visível de que a predição dependeu de
imputação.

## Alternativas consideradas

| Alternativa | Motivo do descarte |
|---|---|
| Exames obrigatórios | Recusaria o caso de uso mais frequente (triagem sem exames) e diverge do comportamento do modelo original |
| Imputação por mediana | Diverge do modelo original; a categorização dos exames (normal/alterado) mudaria para os imputados |

## Consequências

A imputação sem exames usa apenas variáveis sociodemográficas e de diagnóstico, portanto
a predição de um paciente sem exames é menos informativa; o aviso na interface e no
model card explicita isso. Custo por requisição com os três exames ausentes é cerca de
vinte milissegundos maior (três buscas de vizinhos).

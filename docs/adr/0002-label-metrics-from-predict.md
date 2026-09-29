# ADR 0002: métricas de rótulo calculadas a partir de `predict`, não de `predict_proba >= 0,5`

| Campo | Valor |
|---|---|
| Status | Aceito |
| Data | 2026-09-15 |
| Referências | `src/models/evaluate.py::compute_metrics`, `src/models/cross_validation.py`, Tabela 4 da dissertação |

## Contexto

O notebook original calcula AUC a partir de `predict_proba` e acurácia, precisão,
sensibilidade e F1 a partir de `predict`. Para a Regressão Logística as duas regras
coincidem (`predict` é `proba >= 0,5`). Para o `SVC(class_weight='balanced',
probability=True)` não coincidem: `predict` usa a função de decisão, ponderada pela
classe, enquanto `predict_proba` usa a calibração de Platt, ajustada ao prior
desbalanceado. Na primeira execução da etapa 3, com a regra `proba >= 0,5`, o SVM
saiu com sensibilidade 0,062; com `predict`, 0,598, coerente com os 0,60 da Tabela 4.

## Decisão

`compute_metrics` aceita `y_pred` explícito; o runner passa `estimator.predict` para as
métricas de rótulo e mantém `predict_proba` para a AUC e para as probabilidades
servidas pela API. O threshold da configuração continua governando a API, onde só a
Regressão Logística é servida e as regras coincidem.

## Alternativas consideradas

| Alternativa | Motivo do descarte |
|---|---|
| Manter `proba >= threshold` para todos | Tabela comparativa deixaria de reproduzir a dissertação para o SVM; a comparação entre candidatos perderia sentido |
| Recalibrar o SVM | Melhoria de modelagem, fora de escopo |

## Consequências

A tabela de experimentos é comparável à Tabela 4 (DIAS, 2024) em todos os candidatos.
Se um dia outro algoritmo além da LR for promovido, a API precisará escolher entre
`predict` e threshold sobre probabilidade, e essa escolha deverá ser registrada em novo ADR.

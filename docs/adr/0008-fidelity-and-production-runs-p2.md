# ADR 0008: decisão P2, duas runs (fidelidade e produção) e qual é promovida

| Campo | Valor |
|---|---|
| Status | Aceito |
| Data | 2026-09-15 |
| Referências | `HypertensionTransformer(mode=...)`, `evidencias/fidelity_vs_production.md`, inventário item D2 |

## Contexto

O notebook ajusta imputação KNN e one-hot em cada conjunto que transforma, inclusive
nos folds de teste da validação cruzada (vazamento leve). Um transformador correto faz
`fit` só no treino. As duas exigências colidem: "divergência é bug" e "vazamento é
errado".

## Decisão

Duas runs registradas. `lr_fidelity` reproduz o ajuste do notebook (`mode=fidelity`) e é
a origem dos golden values; `lr_production` ajusta o transformador só no treino
(`mode=production`) e é a run promovida a Production. A diferença entre elas é reportada
com a tolerância: AUC -0,000359 e sensibilidade -0,001896 na CV, holdout praticamente
idêntico.

## Alternativas consideradas

| Alternativa | Motivo do descarte |
|---|---|
| Só a run de fidelidade | Serviria um transformador com vazamento e impossível de aplicar a uma observação isolada |
| Só a run de produção | Não haveria como demonstrar que a refatoração reproduz o notebook antes de corrigi-lo |

## Consequências

O modelo servido tem métricas ligeiramente diferentes das da dissertação (diferença
documentada e dentro da tolerância). O modo `fidelity` continua disponível apenas para o
teste de regressão.

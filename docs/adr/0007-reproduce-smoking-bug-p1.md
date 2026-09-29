# ADR 0007: decisão P1, reproduzir o bug de `fumante_hist` no modelo servido

| Campo | Valor |
|---|---|
| Status | Aceito (complementa o ADR 0001, que descreve o mecanismo) |
| Data | 2026-09-15 |
| Referências | ADR 0001, `configs/config.yaml::features.smoking`, runs `lr_production` e `lr_smoking_fixed` |

## Contexto

O inventário (item D4) mostrou que `fumante_hist` compara string com inteiro e por isso
classifica fumantes diários com P051 e P052 ausentes como sem histórico. Corrigir muda a
variável dentro do modelo campeão e, portanto, as métricas de referência.

## Decisão

Reproduzir o comportamento (`smoking.reproduce_string_int_comparison_bug: true`) no
modelo promovido, porque a regra de fidelidade é inegociável: o objeto de estudo é a
produtização do modelo da dissertação, não a sua melhoria. A variante corrigida roda como
`lr_smoking_fixed` (papel `comparison`) e entra na tabela de experimentos.

## Alternativas consideradas

| Alternativa | Motivo do descarte |
|---|---|
| Corrigir e regerar os golden values | O modelo servido deixaria de ser o da dissertação; qualquer diferença de métrica passaria a ser indistinguível de bug de refatoração |

## Consequências

Resultado medido: a correção altera AUC em -0,0001 e sensibilidade em -0,0018 na CV
(dentro da tolerância). A variável permanece clinicamente incorreta no modelo servido;
o model card registra a limitação e a correção fica como oportunidade futura.

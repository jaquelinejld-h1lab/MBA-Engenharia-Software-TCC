# ADR 0016: sete algoritmos no pipeline, não os nove do notebook

Status: aceita. Data: 2026-09-23.

## Contexto

O notebook original (`experimento_mecai24_v4.ipynb`, função `evaluate_models_cv`) treina
nove algoritmos e a Tabela 4 da dissertação vem dessa comparação. O pipeline produtizado
declara sete em `configs/config.yaml::model.runs`. Ausentes: Naive Bayes e KNN.

Na comparação do estudo original, avaliada por validação cruzada com k=10, esses dois
ficaram nas duas últimas posições:

| Algoritmo | AUC-ROC (CV k=10) |
|---|---|
| Regressão Logística | 0,75 |
| CatBoost | 0,73 |
| LightGBM, SVM | 0,72 |
| Random Forest | 0,71 |
| XGBoost | 0,70 |
| **Naive Bayes** | **0,69** |
| Rede neural (MLP) | 0,64 |
| **KNN** | **0,58** |

O propósito das sete runs no pipeline não é reproduzir a Tabela 4, que já existe na
dissertação e está reproduzida em `docs/00_inventario_artefatos_originais.md`. É
sustentar a promoção do campeão: mostrar que a Regressão Logística foi escolhida entre
alternativas reais, incluindo as três famílias de gradient boosting, uma de ensemble por
bagging, uma de margem e uma de rede neural.

## Decisão

Manter sete algoritmos e registrar a ausência dos dois, em vez de acrescentá-los.

Critério: uma run existe no pipeline para disputar a promoção ou para documentar uma
decisão de projeto (é o caso de `lr_fidelity`, `lr_smoking_fixed` e
`lr_baseline_unweighted`). Naive Bayes e KNN não fazem nem uma coisa nem outra. O KNN,
em particular, é o pior classificador da comparação original e teria o custo adicional de
guardar o conjunto de treino inteiro no artefato servido.

O piso de comparação que o pipeline precisava não é um modelo fraco, e sim um sem
aprendizado: o baseline por regra clínica
(`src/models/rule_baseline.py`, AUC 0,676 no holdout) cumpre esse papel melhor do que o
KNN, porque responde à pergunta "o modelo vale a complexidade?" em vez de "existe
algoritmo pior?".

## Alternativas consideradas

| Alternativa | Motivo do descarte |
|---|---|
| Acrescentar os dois e igualar os nove | Duas runs a mais no MLflow e duas curvas a mais na ROC comparativa, sem mudar a promoção nem nenhuma conclusão do TCC |
| Reduzir a menos de sete | Enfraqueceria a demonstração de que a escolha do campeão foi comparativa, que é o que o OE2 pede |
| Acrescentar só o Naive Bayes | AUC 0,69 o coloca acima do MLP, que já está no pipeline; incluir um e não o outro exigiria um critério que não existe |

## Consequências

- A comparação de nove algoritmos permanece disponível como resultado do estudo original,
  citável no TCC a partir da dissertação, e não é refeita aqui.
- A ROC comparativa da página Modelo mostra as sete runs do pipeline, que é o que o
  sistema produziu, com a AUC de cada uma medida no holdout.
- Se a comparação de nove for exigida na defesa, acrescentar os dois é barato: duas
  entradas em `model.runs`, sem mudança de código, já que `NaiveBayes` e `KNN` são
  estimadores do scikit-learn como os demais. O que este ADR registra é que a ausência é
  escolha, não esquecimento.

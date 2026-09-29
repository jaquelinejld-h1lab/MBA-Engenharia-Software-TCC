# ADR 0003: explicabilidade local por contribuições lineares, não por `shap.TreeExplainer`

| Campo | Valor |
|---|---|
| Status | Aceito |
| Data | 2026-09-15 |
| Referências | `src/api/service.py::_contributions`, endpoint `/predict?explain=true`, inventário item `[INCERTEZA]` da etapa 1 |

## Contexto

O notebook original calcula valores SHAP com `shap.TreeExplainer` para o CatBoost. O
modelo promovido é a Regressão Logística, para a qual `TreeExplainer` não se aplica.
A interface (etapa 6) precisa das cinco variáveis mais influentes em cada predição
individual.

## Decisão

Para um modelo linear em log-odds, `logit(p) = b0 + sum(coef_j * x_j)`; com entradas
one-hot (`x_j` em {0, 1}), `coef_j * x_j` é uma atribuição aditiva exata de cada dummy
ativa, idêntica ao valor SHAP de um modelo linear com linha de base zero (o
`LinearExplainer` do pacote `shap` com `feature_perturbation="interventional"` e
referência nula devolve exatamente `coef_j * (x_j - 0)`). A API calcula isso
diretamente dos coeficientes, sem dependência do pacote `shap` na imagem da API, e
devolve as `api.top_contributions` maiores em módulo, em log-odds.

## Alternativas consideradas

| Alternativa | Motivo do descarte |
|---|---|
| `shap.LinearExplainer` com referência na média do treino | Resultado equivalente a menos de uma constante por variável; acrescentaria `shap` e `numba` à imagem da API sem ganho de informação |
| `shap.KernelExplainer` | Estocástico e lento por requisição; incompatível com o SLO de p95 |
| Explicabilidade no Streamlit importando o modelo | Viola a regra de que a interface é apenas cliente HTTP |

## Consequências

A contribuição é reportada em log-odds; a interface converte para linguagem de
apoio (aumenta ou reduz o risco). Se um modelo não linear for promovido, a lista
volta vazia e este ADR precisa ser revisto.

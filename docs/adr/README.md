# ADRs

Um arquivo por decisao, numerado: `NNNN-titulo-curto.md`. Formato: contexto, decisao,
alternativas consideradas, consequencias.

| ADR | Decisão |
|---|---|
| [0001](0001-reproduce-smoking-history-bug.md) | Mecanismo do bug de `fumante_hist` e sua reprodução no código |
| [0002](0002-label-metrics-from-predict.md) | Métricas de rótulo via `predict` (SVM com Platt) |
| [0003](0003-explainability-linear-contributions.md) | Explicabilidade local por contribuições lineares em vez de SHAP |
| [0004](0004-mlflow-tracking-and-registry.md) | MLflow local (SQLite) contra alternativas; fallback JSON |
| [0005](0005-no-feature-store.md) | Ausência de feature store |
| [0006](0006-own-psi-ks.md) | PSI e KS próprios contra biblioteca |
| [0007](0007-reproduce-smoking-bug-p1.md) | P1: bug de `fumante_hist` reproduzido no modelo servido |
| [0008](0008-fidelity-and-production-runs-p2.md) | P2: runs de fidelidade e produção |
| [0009](0009-sample-weights-not-used-d6.md) | D6: peso amostral não utilizado |
| [0010](0010-no-canary-no-autoscaling.md) | Canary release e autoscaling inviáveis sem nuvem |
| [0011](0011-inference-imputation-p3.md) | P3: imputação de exames ausentes em inferência |
| [0012](0012-pydantic-quality-gate-not-great-expectations.md) | Gate de qualidade em Pydantic, sem Great Expectations |
| [0013](0013-docker-first-execution.md) | Imagem de treino e execução do pipeline em container como caminho principal |
| [0014](0014-fidelity-reference-is-environment-bound.md) | Referência de fidelidade congelada no ambiente fixado, por script versionado |
| [0015](0015-model-page-served-by-the-api.md) | Página de modelo na interface, alimentada por rotas `/model` da API |
| [0016](0016-seven-algorithms-not-nine.md) | Sete algoritmos no pipeline, com o piso de comparação sendo a regra clínica |

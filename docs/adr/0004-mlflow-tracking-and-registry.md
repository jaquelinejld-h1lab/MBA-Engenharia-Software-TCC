# ADR 0004: MLflow local (SQLite) como tracking e registry, com fallback JSON

| Campo | Valor |
|---|---|
| Status | Aceito |
| Data | 2026-09-15 |
| Referências | `src/models/registry.py`, `configs/config.yaml::tracking`, OE2 |

## Contexto

O OE2 exige 100 % dos datasets com hash, parâmetros e modelos rastreados, no mínimo
dez execuções registradas e um campeão promovido por estágio (Staging/Production) com
versão. Não há nuvem: tudo roda em máquina Windows local, em containers e em runners
efêmeros do GitHub Actions.

## Decisão

MLflow 2.17 com backend SQLite (`mlruns/mlflow.db`) e artefatos em `mlruns/artifacts`.
O código acessa o tracking por um `Protocol` (`ExperimentTracker`) com duas
implementações: `MlflowTracker` (padrão) e `JsonTracker` (`evidencias/runs/*.json` e
`registry.json`), que reproduz o mesmo contrato sem a biblioteca. O registry usa
estágio (`transition_model_version_stage`) e alias (`champion`), e a API carrega o modelo
pelo alias ou por um diretório local de run.

## Alternativas consideradas

| Alternativa | Motivo do descarte |
|---|---|
| DVC experiments | Bom para dados, fraco para registry por estágio; exigiria dois sistemas |
| Weights & Biases, Neptune | Serviço em nuvem; viola a restrição de ambiente |
| MLflow com PostgreSQL e MinIO | Correto para produção, mas acrescenta dois serviços a uma pilha local já com cinco; SQLite atende dez runs e um registry de uma dezena de versões |
| Planilha ou JSON manual apenas | Sem registry, sem UI, sem comparação de runs; ficou como fallback e como camada de teste |

## Consequências

O `JsonTracker` permite executar e testar o pipeline sem MLflow (sessão de
desenvolvimento e testes unitários), mas o OE2 só é evidenciado com o backend MLflow no
CI. Os artefatos são registrados com URI absoluta do ambiente de treino; por isso
todos os containers do compose montam a raiz do repositório em `/app` (ADR 0013): uma
run treinada pelo serviço `train` tem URI `file:///app/mlruns/...`, legível pela UI do
MLflow e pela API com `api.model_source=mlflow` dentro do compose. Uma run treinada
com Python no host tem URI do host e só é servida pela API executada no host.

# Runbook de incidentes

Serviço: API de risco de hipertensão e interface Streamlit, em `docker compose`
(`docker/docker-compose.yml`). Toda resposta da API é apoio à decisão e não substitui
julgamento clínico; nenhum incidente aqui envolve decisão clínica automática.

## Alertas (Prometheus, `docker/prometheus/alerts.yml`)

| Alerta | Condição | Severidade | Seção |
|---|---|---|---|
| ApiDown | `up{job="hypertension-api"} == 0` por 2 min | crítica | [API fora do ar](#api-fora-do-ar) |
| ApiHighErrorRate | 5xx acima de 5 % das requisições por 5 min | crítica | [Erros](#erros) |
| ApiHighLatencyP95 | p95 de `/predict` acima de 300 ms (SLO) por 5 min | aviso | [Latência](#latencia) |
| DataDriftDetected | `data_drift_alerts > 0` por 10 min | aviso | [Drift](#drift) |

Onde olhar primeiro: painel do Grafana (`http://localhost:3000`, dashboard
"Hypertension API: RED e drift"), painel de monitoramento do Streamlit
(`http://localhost:8501`) e `docker compose -f docker/docker-compose.yml logs api`.
Todo log da API é JSON com `correlation_id`; o mesmo id aparece na resposta
(`X-Correlation-ID`) e na interface.

## API fora do ar

1. `docker compose -f docker/docker-compose.yml ps` e `logs api --tail 200`.
2. Falha no arranque com `ModelNotFoundError`: o volume `artifacts/` não contém
   `lr_production/pipeline.joblib` ou o pickle é de um esquema de configuração antigo
   (`incompatible configuration schema`). Ação: `python -m src.models.train --run lr_production`
   na máquina host e `docker compose restart api`.
3. Falha de porta (8000 ocupada): `HTN_API__PORT` não altera o mapeamento do compose;
   edite `ports` ou libere a porta.
4. Healthcheck vermelho com processo vivo: `curl localhost:8000/health`; `status:
   degraded` significa modelo não carregado.

## Erros

1. Filtrar o log por `"level": "ERROR"` e agrupar por `error`.
2. `DataContractError`/`TransformError` (422) não são incidentes do serviço: são
   payloads fora do contrato. Volume alto indica cliente ou integração enviando dados
   fora do domínio; conferir `api_prediction_errors_total` por tipo.
3. `InternalError` (500): o traceback está no log com o `correlation_id`; nunca no
   corpo da resposta. Reproduzir com o payload do log em `tests/contract`.
4. Se o erro surgiu após promoção de versão: [rollback](#rollback-de-versao-do-modelo).

## Latência

1. Confirmar carga: `sum(rate(api_requests_total[5m]))` no Prometheus. Um único worker
   atende cerca de 30 req/s (medido em 2 vCPU); acima disso a fila cresce.
2. Verificar CPU e memória do container: `docker stats hypertension-mlops-api-1`.
3. Requisições com os três exames ausentes custam cerca de 20 ms a mais (três buscas
   KNN); lotes grandes devem usar `/predict/batch`, não `/predict` em série.
4. Mitigação: `UVICORN_WORKERS=2 docker compose up -d api`. Consequência: janela de
   drift e métricas ficam por processo (ADR 0010).
5. Referência: `evidencias/load_test_*.md`; reexecutar com
   `python scripts/load_test.py --url http://localhost:8000 --container hypertension-mlops-api-1`.

## Drift

1. Painel de monitoramento (Streamlit) ou `topk(10, data_drift_psi)` no Prometheus:
   quais variáveis estão em alerta e se é PSI, KS ou ambos.
2. Classificar a causa:
   - integração enviando códigos fora do dicionário ou unidade errada (por exemplo
     altura em metros): PSI alto em poucas variáveis e `missing_share` inalterado;
   - mudança real de população (idade, região, prevalência de diagnósticos): PSI alto
     e KS significativo em várias variáveis correlacionadas;
   - exames deixaram de ser coletados: `missing_share_current` próximo de 1 nos exames.
3. Erro de integração: corrigir a origem; a janela é deslizante
   (`monitoring.window_size`) e o alerta cede sozinho.
4. Mudança real de população: abrir retreino (abaixo). Até lá, o modelo continua
   servindo; a interface avisa que a saída é apoio à decisão.
5. Falso positivo por janela pequena: `data_drift_window_rows` abaixo de algumas
   centenas produz PSI ruidoso; `monitoring.min_rows_for_drift` controla o mínimo.

## Retreino

### Quando retreinar: a flag

A decisão não é do operador nem do alerta isolado. `scripts/retrain.py` compara a janela
de observações com a referência do treino e grava
`evidencias/retraining_flag.json` mais o relatório em Markdown ao lado.

| Comando | O que faz |
|---|---|
| `make retrain-check` | Decide e relata. Nunca treina. É o que um job agendado ou o CI executam |
| `make retrain` | Decide e, se a flag recomendar, executa `python -m src.models.train --all` |
| `python scripts/retrain.py --force` | Treina mesmo com a flag negativa, para retreino planejado |
| `python scripts/retrain.py --window recent.csv` | Decide sobre um CSV de observações recentes |

Critério, declarado em `configs/config.yaml::monitoring.retraining`: a flag recomenda a
partir de `min_variables_in_alert` variáveis em alerta (hoje 3). Uma variável em alerta é
ruído de amostra; várias ao mesmo tempo indicam que a população mudou. Uma janela com
menos de `monitoring.min_rows_for_drift` observações **não decide**, e o relatório diz
isso, em vez de concluir que o modelo está bem.

A mesma decisão aparece no `/metrics` da API como o gauge `retraining_recommended`, e o
gauge e o arquivo usam o mesmo critério, de propósito: um painel que discorda do runbook
é pior do que nenhum painel.

### Como retreinar

1. Coletar a nova base no mesmo esquema (`configs/variable_mapping.yaml`) e registrar
   origem, data e SHA-256 em `configs/config.yaml::data`.
2. Rodar o gate de qualidade: `python -m src.data.quality --report evidencias/data_quality_real.json`.
3. Treinar: `python -m src.models.train --all` (dez runs no MLflow, promoção do
   `lr_production`). O teste de regressão contra o golden **vai falhar** se a base mudou;
   é esperado: congelar um novo golden com `--freeze-golden` e registrar a nova
   referência de fidelidade em ADR.
4. Validar a versão candidata em Staging (compose com `HTN_API__MODEL_SOURCE=mlflow`
   na máquina host) antes de promover a Production.

## Rollback de versão do modelo

Origem local (padrão do compose):

1. `artifacts/` mantém um diretório por run. Apontar a API para o diretório anterior:
   `HTN_API__LOCAL_RUN_DIR=artifacts/<run_anterior> docker compose up -d api`.
2. Confirmar em `/health` (`model_version`) e no painel.

Origem MLflow (API no host, ou no compose quando a run foi treinada pelo serviço `train`):

1. `mlflow.tracking.MlflowClient().set_registered_model_alias("hypertension-lr", "champion", "<versao_anterior>")`
   e `transition_model_version_stage(..., stage="Production", archive_existing_versions=True)`.
2. Reiniciar a API; a versão carregada aparece em `/health` e no log `model loaded`.
3. Registrar o rollback (versão, motivo, correlation ids do incidente) em `docs/adr/`
   ou no issue do GitHub.

## Backup e retenção

`mlruns/mlflow.db` e `artifacts/` são o estado do sistema; copiar antes de qualquer
retreino. A janela de observações da API vive só em memória e é descartada no reinício
(política de retenção em `docs/checklist_privacidade.md`).

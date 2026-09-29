# Predição de risco de hipertensão arterial: um estudo de produtização

[![ci](https://github.com/jaquelinejld-h1lab/MBA-Engenharia-Software-TCC/actions/workflows/ci.yml/badge.svg)](https://github.com/jaquelinejld-h1lab/MBA-Engenharia-Software-TCC/actions/workflows/ci.yml)
![coverage](https://img.shields.io/badge/cobertura-gate%2080%25-blue)
![python](https://img.shields.io/badge/python-3.11-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Produtização do modelo preditivo de risco de hipertensão arterial desenvolvido em
Dias (2024), como estudo de caso do TCC do MBA em Engenharia de Software (USP/Esalq).

O repositório leva um modelo que existia como notebook de pesquisa até um sistema em
operação: pipeline reprodutível, rastreamento de experimentos, testes automatizados,
API, interface e observabilidade, tudo executável localmente por comando único.

> **Aviso clínico.** Este sistema é apoio à decisão e **não substitui** avaliação e
> julgamento clínico de profissional de saúde. As saídas são probabilidades estimadas a
> partir de dados de inquérito populacional e não constituem diagnóstico.

## O que está implementado

| Etapa do pipeline | Implementação |
|---|---|
| Coleta e pré-processamento | Ingestão verificada por SHA-256, contrato de esquema e gate de qualidade bloqueante |
| Treinamento e validação | Dez configurações, baseline em Regressão Logística, validação cruzada estratificada k=10 |
| Versionamento e rastreamento | MLflow local, dados versionados por hash, campeão promovido no Model Registry |
| Testes automatizados | Unitários, de propriedade, de integração e de contrato, com gate de cobertura de 80 % |
| Implantação e containerização | FastAPI em imagem multi-stage e pilha completa em `docker compose` |
| Monitoramento e retreinamento | PSI e KS próprios, métricas Prometheus, dashboard Grafana e flag de retreino |

## Execução rápida

Pré-requisitos: Git e Docker Desktop. Nenhuma credencial, conta ou serviço externo é
necessário, e o extrato de dados acompanha o repositório.

```bash
git clone https://github.com/jaquelinejld-h1lab/MBA-Engenharia-Software-TCC.git
cd MBA-Engenharia-Software-TCC

# Constroi a imagem de treino e executa as seis etapas do pipeline
docker compose -f docker/docker-compose.yml --profile tools build train
docker compose -f docker/docker-compose.yml --profile tools run --rm train

# Sobe a pilha de servicos
docker compose -f docker/docker-compose.yml up --build -d
```

Duração: cerca de 15 minutos no primeiro build e mais 10 a 20 minutos no pipeline.

| Serviço | Endereço |
|---|---|
| Interface Streamlit | http://localhost:8501 |
| API (documentação OpenAPI) | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Para encerrar: `docker compose -f docker/docker-compose.yml down`.

O procedimento completo, incluindo o caminho alternativo com Python local, os comandos
do dia a dia e a tabela de erros comuns, está em
**[docs/guia_execucao.md](docs/guia_execucao.md)**.

## Stack

| Camada | Ferramentas |
|---|---|
| Modelagem | Python 3.11, pandas, scikit-learn, XGBoost, LightGBM, CatBoost |
| Rastreamento | MLflow com backend SQLite e Model Registry |
| API | FastAPI, Pydantic v2, Uvicorn |
| Interface | Streamlit, exclusivamente como cliente HTTP da API |
| Observabilidade | prometheus-client, Prometheus, Grafana |
| Testes e qualidade | pytest, httpx, schemathesis, ruff, black, mypy, bandit, pip-audit |
| Empacotamento | Docker multi-stage, `docker compose`, GitHub Actions |

## Estrutura

```
configs/    parametros do projeto (config.yaml, variable_mapping.yaml)
data/       raw (versionado por hash), interim e processed (gerados)
src/        data, features, models, monitoring, api, app
tests/      unit, integration, contract, fixtures
docker/     Dockerfiles, docker-compose.yml, Prometheus e Grafana
scripts/    pipeline completo, cenarios de drift, teste de carga, relatorio
docs/       guia de execucao, ADRs, model card, dicionario de dados, runbook
notebooks/  artefatos originais e relatorio de resultados
evidencias/ saidas consumidas pelo texto do TCC
```

## Dados

`data/raw/EXAMES-PNS-2013-FINAL_05052023.xlsx` deriva dos microdados públicos da
**Pesquisa Nacional de Saúde 2013**, do Instituto Brasileiro de Geografia e Estatística
(IBGE), módulo de exames laboratoriais. São dados secundários, públicos e anonimizados,
sem identificador de domicílio, pessoa ou município.

SHA-256: `80222fa535524829d531504109ac4b49d44e4e6d0c04c3c281ff972bd332d1f7`

Atribuição: IBGE, Pesquisa Nacional de Saúde 2013. Os dados pertencem ao IBGE e são
redistribuídos aqui apenas para fins de reprodutibilidade acadêmica. Origem, termos de
uso, data de coleta e data de congelamento estão em
[`docs/dicionario_dados.md`](docs/dicionario_dados.md).

## Documentação

| Documento | Conteúdo |
|---|---|
| [`docs/guia_execucao.md`](docs/guia_execucao.md) | Reprodução completa em máquina limpa e erros comuns |
| [`docs/arquitetura.md`](docs/arquitetura.md) | Diagramas do pipeline, dos componentes e do fluxo de requisição |
| [`docs/adr/`](docs/adr/) | Dezesseis registros de decisão de arquitetura |
| [`docs/model_card.md`](docs/model_card.md) | Uso pretendido, população, métricas por subgrupo e limitações |
| [`docs/dicionario_dados.md`](docs/dicionario_dados.md) | Variáveis brutas e do modelo, origem, licença e data freeze |
| [`docs/runbook.md`](docs/runbook.md) | Alertas, incidentes, retreinamento e rollback |
| [`docs/slo.md`](docs/slo.md) | Indicadores, objetivos e orçamento de erro |
| [`docs/checklist_privacidade.md`](docs/checklist_privacidade.md) | Risco de reidentificação, retenção e descarte |

## Referência

DIAS, Jaqueline Lopes. **Aprendizado de máquina aplicado à predição de doenças
crônicas**: um estudo de caso de hipertensão arterial. 2024. Dissertação (Mestrado
Profissional) - ICMC, Universidade de São Paulo, São Carlos, 2024.

## Licença

Código sob licença MIT ([`LICENSE`](LICENSE)). Os dados seguem os termos do IBGE.

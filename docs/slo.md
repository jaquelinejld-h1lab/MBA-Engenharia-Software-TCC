# SLO da API de risco de hipertensão

Documento de objetivo de nível de serviço da API que serve o modelo campeão. Define o que
se promete, como se mede, quanto se pode falhar antes de agir e o que fazer quando o
orçamento acaba.

Escopo: a API (`src/api`) em execução local por `docker compose`. Não cobre a interface
Streamlit, que é cliente, nem o MLflow, que é ferramenta de desenvolvimento.

## Indicadores e objetivos

| SLI | Como é medido | SLO | Onde aparece |
|---|---|---|---|
| Latência de `/predict` | p95 do histograma `api_request_latency_seconds`, janela de 5 min | **p95 ≤ 300 ms** | Grafana, painel de monitoramento, alerta `ApiHighLatencyP95` |
| Disponibilidade | `up{job="hypertension-api"}` e taxa de respostas não 5xx | **99 % das requisições sem 5xx**, janela de 30 dias | Alertas `ApiDown` e `ApiHighErrorRate` |
| Correção do serviço | Paridade entre a predição da API e a do pipeline offline | **100 %**, verificada em teste de contrato | `tests/contract/test_api_contract.py` |

O SLO de latência é declarado **para quatro clientes concorrentes sobre um único
worker** (`load_test.concurrency`, `UVICORN_WORKERS=1`). Fora dessa condição o número não
significa a mesma coisa: mais workers multiplicam o throughput, mas fragmentam a janela
de drift e as métricas entre processos.

## Gate interno mais rígido que o SLO

O critério de aceitação do projeto é `acceptance.max_p95_latency_ms = 200 ms`, abaixo do
SLO de 300 ms. A diferença é deliberada: o gate reprova a entrega antes que o usuário
perceba degradação, deixando 100 ms de folga entre "o teste de carga reprovou" e "o
serviço violou o que prometeu".

| Limiar | Valor | Efeito |
|---|---|---|
| Gate do teste de carga | 200 ms | `scripts/load_test.py` termina com código diferente de zero |
| SLO declarado | 300 ms | Alerta `ApiHighLatencyP95` no Prometheus |

## Orçamento de erro

Com 99 % de disponibilidade em 30 dias, o orçamento é de **1 % das requisições**, ou
cerca de 7 horas e 12 minutos de indisponibilidade equivalente por mês.

| Consumo do orçamento | Postura |
|---|---|
| Abaixo de 50 % | Operação normal; mudanças seguem o fluxo habitual |
| Entre 50 % e 100 % | Congelar mudanças de modelo; só correção de confiabilidade |
| Acima de 100 % | Incidente: rollback da versão do modelo (`docs/runbook.md`) antes de qualquer nova entrega |

Esta é a política declarada para o estudo. Em operação real ela seria negociada com quem
usa o serviço, e o número de 99 % viria dessa negociação, não do autor do sistema.

## Medição

| Fonte | Papel |
|---|---|
| `/metrics` da API | Histograma de latência, contador por endpoint e status, erros por tipo |
| Prometheus | Raspagem a cada 15 s e avaliação das regras de alerta |
| Grafana | Painel versionado em `docker/grafana/dashboards/hypertension_api.json` |
| `scripts/load_test.py` | Medição sob carga controlada, com p50, p95, throughput, CPU, memória e custo estimado por mil requisições |

A latência medida pelo teste de carga inclui a rede entre host e contêiner, que é o que
um cliente real enfrenta. Medir de dentro do contêiner subestima o número.

## O que este SLO não cobre

- **Qualidade da predição.** Latência e disponibilidade não dizem nada sobre AUC. O
  desempenho do modelo tem critérios próprios (`acceptance.min_auc` e
  `acceptance.min_sensitivity`) e monitoramento próprio (drift e flag de retreinamento).
- **Ambiente de nuvem.** Sem autoscaling nem canary (ADR 0010), o comportamento sob
  carga muito acima da medida é desconhecido e não é prometido.
- **Disponibilidade multi-instância.** Uma única instância local não tem alta
  disponibilidade; o objetivo de 99 % descreve o comportamento observado, não uma
  garantia de arquitetura.

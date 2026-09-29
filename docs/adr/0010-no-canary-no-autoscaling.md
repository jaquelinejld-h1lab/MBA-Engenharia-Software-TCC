# ADR 0010: canary release e autoscaling inviáveis sem nuvem

| Campo | Valor |
|---|---|
| Status | Aceito |
| Data | 2026-09-15 |
| Referências | `docker/docker-compose.yml`, `docs/runbook.md`, `docker/Dockerfile.api::UVICORN_WORKERS` |

## Contexto

O Projeto de Pesquisa lista canary release, blue/green e autoscaling como estratégias de
implantação e operação. O ambiente é uma máquina Windows local com Docker Compose, sem
orquestrador nem balanceador.

## Decisão

Não implementar. A promoção de versão é feita pelo registry (estágio e alias) e o
rollback é a troca do alias ou do diretório de run seguida de reinício do container
(procedimento no runbook). A escala horizontal disponível é `UVICORN_WORKERS` no mesmo
host, com a ressalva de que cada worker mantém a própria janela de drift e o próprio
registro de métricas.

## Alternativas consideradas

| Alternativa | Motivo do descarte |
|---|---|
| Dois serviços `api-stable` e `api-canary` atrás de um Nginx com split de tráfego | Simula canary, mas sem métricas comparadas automaticamente e sem rollback automático o ganho é cosmético; acrescenta um serviço sem critério de promoção |
| Kubernetes local (kind, minikube) com HPA | Exige cluster local, pesa na máquina e muda o objeto do TCC de "produtização em Docker" para "operação em Kubernetes" |

## Consequências

Em nuvem, o caminho natural é o registry apontar duas versões e o balanceador dividir
tráfego por peso, com as métricas RED e de drift por versão que a API já expõe (label
de versão a acrescentar). Registrado como trabalho futuro.

# ADR 0006: PSI e KS implementados no próprio código

| Campo | Valor |
|---|---|
| Status | Aceito |
| Data | 2026-09-15 |
| Referências | `src/monitoring/drift.py`, `tests/unit/test_drift.py`, `configs/config.yaml::monitoring` |

## Contexto

O monitoramento exige detecção de drift de dados por PSI e KS contra a janela de
referência do treino, com limiares em configuração, cenário simulado com alerta e cenário
baseline sem alerta.

## Decisão

Implementar as duas estatísticas diretamente (cerca de 200 linhas com testes que fixam
os valores): PSI com bins por quantis da referência para variáveis numéricas, códigos
para categóricas e **um bin próprio para valores ausentes**; KS de duas amostras via
`scipy.stats.ks_2samp` sobre os valores não nulos, com mínimo de observações. A
referência é persistida como JSON junto ao modelo e carregada pela API, que expõe as
estatísticas como gauges Prometheus a cada raspagem.

## Alternativas consideradas

| Alternativa | Motivo do descarte |
|---|---|
| Evidently | Biblioteca extensa (relatórios HTML, dependências pesadas) para duas estatísticas; a versão e o comportamento dos bins mudam entre releases e o TCC precisa explicar a fórmula |
| NannyML, whylogs | Mesma razão; whylogs ainda orienta ao serviço em nuvem |
| Alibi Detect | Focado em detectores multivariados; excede o escopo |

## Consequências

O bin de ausência foi decisivo: sem ele, variáveis respondidas por uma minoria (fontes
de renda, tabagismo passado) produziam alertas falsos no próprio holdout. Custo: manter
o código e os testes; benefício: rastreabilidade completa da fórmula na dissertação.

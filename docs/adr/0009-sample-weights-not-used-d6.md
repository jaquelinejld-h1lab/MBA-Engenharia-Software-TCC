# ADR 0009: peso amostral `PESO_LAB` não utilizado (item D6)

| Campo | Valor |
|---|---|
| Status | Aceito como limitação registrada |
| Data | 2026-09-15 |
| Referências | inventário itens D6 e 3.1, `docs/model_card.md` |

## Contexto

A PNS 2013 tem desenho amostral complexo em três estágios e fornece o peso do módulo
laboratorial (`PESO_LAB`). O notebook original nunca usa o peso: as métricas são não
ponderadas e descrevem a amostra, não a população brasileira.

## Decisão

Não usar o peso no treino nem na avaliação, para preservar a fidelidade ao modelo da
dissertação. A coluna não entra no caminho de inferência (não faz sentido em uma
predição individual) e o dicionário de dados registra a limitação.

## Alternativas consideradas

| Alternativa | Motivo do descarte |
|---|---|
| `sample_weight=PESO_LAB` no `fit` e nas métricas | Mudança de modelagem; altera coeficientes e métricas fora da tolerância; fora do escopo do TCC |
| Reportar métricas ponderadas como avaliação complementar | Útil, mas exige tratar os seis registros com peso 99,0 (código de ausência) e discussão metodológica que pertence a trabalho futuro |

## Consequências

As métricas reportadas (AUC 0,7488, sensibilidade 0,6813) valem para a amostra
analítica de 6.203 adultos com exames, não para a população. Oportunidade futura
registrada no model card.

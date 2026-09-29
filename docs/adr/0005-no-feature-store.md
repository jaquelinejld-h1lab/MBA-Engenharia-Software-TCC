# ADR 0005: ausência de feature store

| Campo | Valor |
|---|---|
| Status | Aceito |
| Data | 2026-09-15 |
| Referências | `src/features/transformer.py`, `configs/variable_mapping.yaml` |

## Contexto

Arquiteturas de referência de MLOps (Kreuzberger, Kühl e Hirschl, 2023) incluem um
feature store para compartilhar atributos entre treino e inferência e evitar
divergência (training/serving skew).

## Decisão

Não adotar feature store. Todas as derivações são funções puras em
`src/features/derivations.py`, parametrizadas por `configs/config.yaml`, e o
`HypertensionTransformer` (derivações, imputação KNN, one-hot com categorias
explícitas) é serializado com joblib e embutido no `Pipeline` servido pela API. O mesmo
objeto transforma o treino e cada requisição, o que elimina o skew por construção.

## Alternativas consideradas

| Alternativa | Motivo do descarte |
|---|---|
| Feast (modo local) | Infraestrutura adicional (registry, offline e online store) para um único modelo, um único dataset congelado e sem atributos calculados em tempo real |
| Tabela de atributos materializada em Parquet | Resolveria só o caminho offline; a inferência individual continuaria precisando do código de derivação |

## Consequências

O contrato de esquema (`variable_mapping.yaml`) substitui o catálogo do feature store.
Se um segundo modelo ou uma fonte de dados contínua entrarem, a decisão deve ser
revista.

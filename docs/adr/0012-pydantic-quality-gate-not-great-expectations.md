# ADR 0012: gate de qualidade de dados em Pydantic, sem Great Expectations

| Campo | Valor |
|---|---|
| Status | Aceito |
| Data | 2026-09-15 |
| Referências | `src/data/quality.py`, `src/data/contract.py`, `configs/variable_mapping.yaml`, job `tests` do CI |

## Contexto

A stack de referência do projeto admite "Pydantic e/ou Great Expectations" para o gate
de qualidade de dados, que precisa ser bloqueante no CI. A etapa 1 fixou as duas
bibliotecas no `requirements-train.lock` antes de qualquer implementação.

Duas constatações posteriores:

1. O gate implementado (`python -m src.data.quality`) é inteiramente próprio: verifica
   SHA-256 e dimensões do arquivo, as contagens de cada filtro populacional, a fração
   de nulos e o domínio declarado de cada uma das 61 variáveis brutas, tudo a partir de
   `configs/variable_mapping.yaml` e do contrato Pydantic (`RawRecord`, `ModelRecord`).
   Nenhum módulo importa `great_expectations`.
2. `great-expectations==1.2.4` exige `pandas<2.2` e torna o `requirements-train.lock`
   insolúvel junto com `pandas==2.2.3`, que é a versão do trio numérico de referência
   (pandas 2.2.3, numpy 1.26.4, scikit-learn 1.5.2). A instalação falha com
   `ResolutionImpossible` em máquina limpa.

## Decisão

Manter o gate em Pydantic e remover `great-expectations` das dependências. A mesma
varredura removeu `shap`, `imbalanced-learn`, `Boruta` e `seaborn`, que também não são
importados em `src/`, `tests/` nem `scripts/`.

## Alternativas consideradas

| Alternativa | Motivo do descarte |
|---|---|
| Rebaixar pandas para `<2.2` | Muda o ambiente numérico do campeão para acomodar uma biblioteca não usada; contraria a regra de fidelidade sem contrapartida |
| Atualizar Great Expectations para uma versão que aceite pandas 2.2 | Acrescentaria um segundo sistema de expectativas sobre as mesmas regras já expressas no mapping, com dezenas de dependências transitivas, para ganho nulo |
| Manter as duas e resolver o conflito com `--no-deps` | Instalação inconsistente e não reprodutível |

## Consequências

O contrato de dados tem uma única fonte, `configs/variable_mapping.yaml`, consumida
pelo gate, pela API e pela interface. Perde-se o formato de relatório e os "data docs"
do Great Expectations; em troca, o relatório do gate sai em JSON
(`--report evidencias/data_quality_real.json`) e entra nas evidências. Se o projeto
passar a validar várias fontes com regras versionadas por terceiros, esta decisão deve
ser revista.

Efeito colateral positivo: o ambiente de treino encolhe de forma relevante (Great
Expectations e SHAP somam centenas de megabytes de dependências transitivas), o que
reduz o tempo de instalação local e do CI.

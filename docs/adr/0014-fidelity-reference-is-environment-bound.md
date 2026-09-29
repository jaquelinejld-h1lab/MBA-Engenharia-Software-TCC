# ADR 0014: a referência de fidelidade é congelada no ambiente fixado

Status: aceita. Data: 2026-09-17.

## Contexto

O gate de saída da etapa 2 (`tests/integration/test_notebook_fidelity.py`) compara a
matriz de projeto do código refatorado com números congelados em
`evidencias/fase0_referencia_notebook_etapa2.json`, obtidos ao reexecutar as funções
originais de `notebooks/original/funcoes_mecai_v3.py`.

Esses números não são independentes de ambiente. A função `impute_knn` do código
original usa `sklearn.impute.KNNImputer` para completar `egfr_afro`, `colesterol` e
`glicose`. A escolha de vizinhos depende de distâncias em ponto flutuante e do
desempate entre elas, que mudam entre versões de scikit-learn e NumPy. O valor imputado
é em seguida categorizado por `pd.cut` com corte em 90 mL/min/1,73 m2, entre outros, de
modo que um punhado de registros próximos da borda muda de classe quando o ambiente
muda.

Observado na prática: a referência tinha sido congelada em um ambiente com pandas 3.0.2,
NumPy 2.4.4 e scikit-learn 1.8.0, enquanto o ambiente fixado do projeto é pandas 2.2.3,
NumPy 1.26.4 e scikit-learn 1.5.2. A divergência resultante em `egfr_class` foi de 13
registros em 6.203 no conjunto de desenvolvimento, 0,21%, suficiente para reprovar o
gate sem que houvesse qualquer erro de refatoração.

## Decisão

1. A referência passa a ser gerada por `scripts/freeze_notebook_reference.py`, versionado,
   e não mais de forma ad hoc. O script reexecuta as funções originais, sem alterar o
   arquivo em disco, e grava no JSON as versões de Python, pandas, NumPy e scikit-learn
   que o produziram.
2. A referência válida é a congelada **dentro da imagem de treino**, que é o ambiente
   fixado pelos `requirements-*.lock` e o mesmo do CI. Alvo: `make freeze-reference`.
3. O teste `test_reference_was_frozen_in_this_environment` compara as versões registradas
   com as do ambiente em execução e falha primeiro, com a instrução de recongelar,
   evitando que uma divergência de ambiente seja lida como erro de refatoração.
4. A única alteração de compatibilidade aplicada ao código original continua sendo
   `OneHotEncoder(sparse=)` para `sparse_output=`, agora aplicada no espaço de nomes do
   módulo em tempo de execução, e registrada no campo `compatibility_patches` do JSON.
   Bibliotecas que o arquivo original importa mas cujas funções usadas nunca chamam
   (SMOTE, Boruta, plotagem) são substituídas por marcadores, listados em
   `stubbed_libraries`; o script verifica que nenhuma das três funções utilizadas
   referencia um marcador e aborta se referenciar.

## Alternativas consideradas

| Alternativa | Motivo do descarte |
|---|---|
| Tolerância numérica no gate (aceitar diferença de N registros) | Descaracteriza o gate: qualquer erro real de refatoração de mesma magnitude passaria despercebido |
| Fixar a referência do ambiente antigo e alinhar os locks a ele | O ambiente antigo (pandas 3.0.2) não é o do projeto nem o do CI, e pandas 3 quebra outras dependências fixadas |
| Substituir `KNNImputer` por imputação determinística | Alteraria a fidelidade ao trabalho original, que é justamente o que o gate protege |

## Consequências

- A referência precisa ser recongelada, e o commit revisado, sempre que pandas, NumPy ou
  scikit-learn mudarem de versão nos locks. O teste de ambiente torna essa necessidade
  explícita em vez de silenciosa.
- O JSON passa a documentar o ambiente, os marcadores e o patch de compatibilidade,
  o que torna a evidência auditável.
- `[REPRODUCIBILIDADE]` registrado: evidência gerada fora do ambiente fixado não é
  evidência. O mesmo critério vale para `evidencias/golden_metrics.json`, que deve ser
  regenerado no container antes do congelamento final da v1.0.

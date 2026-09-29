# Política de versionamento

Três coisas mudam de forma independente neste sistema, e cada uma tem seu próprio
mecanismo de versão. A regra que as une: **qualquer resultado precisa ser rastreável até
o dado, o código e o modelo exatos que o produziram**.

## Dados

| Aspecto | Decisão |
|---|---|
| Identidade | SHA-256 do arquivo bruto, declarado em `configs/config.yaml::data.raw_sha256` |
| Verificação | Conferida a cada execução por `src/data/ingest.py`, antes de qualquer processamento; divergência aborta o pipeline |
| Conjunto derivado | SHA-256 do frame de desenvolvimento, registrado como parâmetro em cada run do MLflow |
| Armazenamento | O extrato vive no repositório (21 MB). Sem armazenamento remoto, conforme a restrição de ambiente |
| DVC | Avaliado e descartado nesta etapa: sem remoto, ele acrescentaria uma ferramenta sem resolver problema que o hash já resolve |

Uma base nova é **outro dado**, não uma atualização: exige novo hash em config, novo gate
de qualidade e retreino, com a referência de fidelidade recongelada
(`scripts/freeze_notebook_reference.py`) e registrada em ADR.

## Código

| Aspecto | Decisão |
|---|---|
| Controle | Git, branch única `main`, commits pequenos com mensagem convencional (`feat`, `fix`, `docs`, `chore`) |
| Versão publicada | Tag semântica `vMAJOR.MINOR.PATCH` ao fechar cada objetivo específico |
| Rastreabilidade no experimento | Revisão do git e hash do código do transformador gravados como parâmetro de cada run |
| Decisões | Um ADR por decisão em `docs/adr/`, numerado e imutável; decisão revista gera ADR novo que supera o anterior, sem apagar o histórico |
| Dependências | Pin exato em `requirements-*.lock`; o grafo transitivo é congelado por `make lock-docker` dentro da imagem, que é o ambiente do CI |

Critério de incremento:

| Incremento | Quando |
|---|---|
| MAJOR | Mudança incompatível no contrato da API ou no esquema de entrada |
| MINOR | Capacidade nova compatível (endpoint, página da interface, etapa do pipeline) |
| PATCH | Correção que não altera contrato nem números |

## Modelo

| Aspecto | Decisão |
|---|---|
| Registro | MLflow Model Registry, modelo `hypertension-lr`, versão inteira atribuída pelo registry |
| Estágios | `Staging` para candidato, `Production` para o servido; alias `champion` aponta para o promovido |
| Promoção | Automática apenas quando os critérios de aceitação são atendidos (`acceptance.min_auc`, `acceptance.min_sensitivity`); caso contrário a versão fica em Staging |
| Artefatos | Transformador, imputadores, encoder, modelo, pipeline, métricas, ROC, interpretação e janela de referência de drift, todos no diretório do run |
| Rollback | Reapontar alias e estágio para a versão anterior e reiniciar a API (`docs/runbook.md`) |
| Retenção | Versões anteriores nunca são apagadas; `mlruns/` e `artifacts/` são copiados antes de qualquer retreino |

## O que amarra os três

Cada run do MLflow guarda, como parâmetro:

- `raw_dataset_sha256` e `development_frame_sha256`, que identificam o dado;
- `code_revision` e `transformer_source_sha256`, que identificam o código;
- `seed`, `n_splits`, `test_size` e `threshold`, que identificam o desenho de validação.

Com esses cinco campos, qualquer número do TCC pode ser refeito: eles dizem qual commit,
qual arquivo e qual configuração produziram aquela linha. É isso que o OE2 pede e o que
`evidencias/reproducibility_two_runs.md` demonstra, com duas execuções completas
produzindo hashes idênticos.

## Evidência também é versionada

Os arquivos de `evidencias/` são commitados junto do código que os gerou. Um número do
texto do TCC que não esteja em `evidencias/` e não tenha sido produzido por um comando
versionado não é evidência: é anotação. O caso concreto está registrado no ADR 0014, em
que uma referência congelada fora do ambiente fixado reprovou o gate e precisou ser
regerada.

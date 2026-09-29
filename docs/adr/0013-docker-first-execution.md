# ADR 0013: imagem de treino e execução do pipeline em container como caminho principal

| Campo | Valor |
|---|---|
| Status | Aceito |
| Data | 2026-09-15 |
| Referências | `docker/Dockerfile.train`, serviço `train` em `docker/docker-compose.yml`, README (caminho A), OE1 e OE6 |

## Contexto

Até a etapa 7 o repositório tinha duas imagens, ambas de runtime de serviço
(`Dockerfile.api` e `Dockerfile.app`), e o pipeline só era executável com Python
instalado no host. Três consequências apareceram na primeira execução em máquina limpa:

1. `docker compose up` **não** funciona em um clone novo: a API exige
   `artifacts/lr_production/pipeline.joblib`, que só existe depois de um treino, e o
   treino só rodava fora do Docker. O critério "sobe a pilha completa" (OE4) dependia,
   na prática, de o caminho local ter dado certo antes.
2. O ambiente local é a maior fonte de atrito do projeto. A execução de referência
   encontrou, em sequência: Python 3.12 onde o projeto exige 3.11, ausência do launcher
   `py` (que só acompanha o instalador do python.org), política de execução do
   PowerShell bloqueando a ativação do venv e um conflito real de dependências.
3. O treino no Windows e o treino do CI rodam em sistemas operacionais diferentes. Os
   valores congelados em `evidencias/golden_metrics.json` vieram de Linux; BLAS e
   bibliotecas de sistema distintas podem deslocar métricas na última casa decimal, o
   que polui a leitura do teste de regressão e da decisão P4.

## Decisão

Acrescentar `docker/Dockerfile.train`, uma imagem de ferramentas com Python 3.11 e os
locks de treino e de desenvolvimento, exposta como o serviço `train` do Compose sob o
perfil `tools` (portanto fora do `docker compose up`). O README passa a apresentar dois
caminhos, com o Docker como principal: `build train`, `run --rm train` e `up -d`. O
caminho com Python local permanece documentado, para desenvolvimento interativo com IDE
e depurador.

Três escolhas de desenho merecem registro:

| Escolha | Motivo |
|---|---|
| A imagem não copia o código; o repositório é montado em `/app` | Editar no host vale imediatamente, sem rebuild. O código é importado via `PYTHONPATH=/app`, sem `pip install -e .`, o que elimina um passo e uma fonte de divergência entre o que está instalado e o que está montado |
| O serviço roda como root por padrão | É uma ferramenta local e efêmera, não um serviço exposto. Em Linux, `--user "$(id -u):$(id -g)"` resolve a propriedade dos arquivos gerados. As imagens de serviço, que são as expostas em rede, continuam como usuário não root |
| `git` e `libgomp1` instalados na imagem | `libgomp1` é o runtime OpenMP exigido por LightGBM, XGBoost e CatBoost; `git` permite que cada run registre a revisão real do código (OE2), com `safe.directory` configurado para que o bind mount não seja recusado por propriedade divergente |

## Alternativas consideradas

| Alternativa | Motivo do descarte |
|---|---|
| Manter só as imagens de serviço e documentar melhor o ambiente local | Documentação não remove a dependência de um Python 3.11 correto no host, e não resolve o fato de o compose não subir em clone novo |
| Reaproveitar `Dockerfile.api` para treinar | A imagem da API não pode conter bibliotecas de treino: é uma restrição explícita do projeto, verificada por smoke test no CI |
| Dev Container (`devcontainer.json`) | Resolve bem o caso do VS Code, mas amarra o projeto a um editor; o serviço do Compose atende qualquer terminal. Continua compatível: quem usa Dev Containers pode apontá-lo para esta mesma imagem |
| Copiar o código para dentro da imagem, como nas de serviço | Exigiria rebuild a cada edição, o que inviabiliza o uso como ambiente de desenvolvimento |

## Consequências

O único pré-requisito para reproduzir o trabalho do zero passa a ser Git e Docker, o que
fortalece o OE6 (README replicável em máquina limpa) e torna o OE1 verificável sem
qualquer instalação prévia. O treino passa a rodar no mesmo Linux do CI, o que remove a
divergência de plataforma da leitura do teste de regressão.

A imagem também é o lugar certo para resolver o grafo completo de dependências
(`make lock-docker`): os `.lock` iniciais fixavam só as diretas, e as duas primeiras
execuções em máquina limpa quebraram por transitivas mais novas do que as bibliotecas
esperavam (`anyio` contra `starlette`, `pyparsing` contra `matplotlib`). O filtro de
avisos do pytest passou a ignorar depreciações atribuídas a terceiros e a mantê-las
como erro quando atribuídas a `src/` ou `tests/` (`tests/unit/test_warning_policy.py`).

Custos: mais uma imagem para construir e manter (cerca de 3 GB, dominados por CatBoost e
pelo trio de gradient boosting), e o ciclo de desenvolvimento interativo dentro do
container é mais lento no Windows quando o repositório está em pasta sincronizada. O job
`docker` do CI constrói a imagem de treino e executa dentro dela o gate de qualidade e os
testes unitários, de modo que uma quebra desse caminho aparece no CI e não na máquina de
quem for reproduzir.

# hypertension-mlops

![ci](https://github.com/OWNER/hypertension-mlops/actions/workflows/ci.yml/badge.svg)
![coverage](https://img.shields.io/badge/coverage-CI%20gate%2080%25-blue)
![release](https://img.shields.io/github/v/release/OWNER/hypertension-mlops?include_prereleases)

> Os três badges e os comandos `git clone` deste arquivo apontam para `OWNER`.
> Troque por seu usuário do GitHub ao publicar o repositório; antes da publicação eles
> não resolvem, e isso é esperado.

Produtização do modelo preditivo de risco de hipertensão arterial desenvolvido em
Dias (2024), como estudo de caso do TCC do MBA em Engenharia de Software (USP/Esalq).
O repositório implementa o pipeline de seis etapas (coleta e pré-processamento,
treinamento e validação, versionamento e rastreamento, testes automatizados,
implantação e containerização, monitoramento e retreinamento), uma API FastAPI, uma
interface Streamlit e observabilidade local com Prometheus e Grafana.

> **Aviso clínico.** Este sistema é apoio à decisão e **não substitui** avaliação e
> julgamento clínico de profissional de saúde. As saídas são probabilidades estimadas a
> partir de dados de inquérito populacional e não constituem diagnóstico.

## Estado atual

| Etapa | Entrega | Status |
|---|---|---|
| 0 | Inventário dos artefatos originais (`docs/00_inventario_artefatos_originais.md`) | concluída |
| 1 | Esqueleto: config tipada, logging, exceções, CI mínimo | concluída |
| 2 | Ingestão, contrato de esquema, transformador (fidelidade verificada por hash) | concluída |
| 3 | Treino, avaliação, MLflow (fidelidade dentro da tolerância) | concluída |
| 4 | Testes unitários, de propriedade, de domínio, integração e reprodutibilidade; gate de cobertura 80 % | concluída |
| 5 | API FastAPI (`/predict`, `/predict/batch`, `/health`, `/metrics`), Dockerfile multi-stage, testes de contrato e paridade | concluída |
| 6 | Interface Streamlit (cliente HTTP), drift PSI/KS, cenários reprodutíveis, pilha `docker compose` (API, interface, MLflow, Prometheus, Grafana), teste de carga | concluída |
| 7 | ADRs, model card, dicionário de dados, runbook, arquitetura, checklist de privacidade, listagens, evidências consolidadas, release v1.0 | concluída |

## Regra de fidelidade

O pipeline reproduz o modelo campeão do notebook original (Regressão Logística,
`C=1.0`, `liblinear`, `class_weight='balanced'`, seed 42, validação cruzada
estratificada k=10). Tolerância declarada em `configs/config.yaml`:
`|ΔAUC| <= 0,005` e `|Δsensibilidade| <= 0,01`. Divergência acima disso é bug de
refatoração.

---

# Execução do zero em máquina limpa

Há dois caminhos, e o primeiro existe justamente para não depender do Python instalado
no Windows.

| | Caminho A: tudo em Docker | Caminho B: Python local |
|---|---|---|
| Pré-requisitos | Git e Docker Desktop | Git, Docker e **Python 3.11 exatamente** |
| Onde o código roda | container Linux, idêntico ao do CI | sistema operacional do host |
| Quando usar | executar o pipeline, reproduzir as evidências, validar o README, qualquer máquina sem Python 3.11 | desenvolver com IDE, depurador e ciclo de teste rápido |
| Custo | uma construção de imagem de alguns minutos e cerca de 3 GB em disco | instalação de dependências e a disciplina de manter o 3.11 |

Recomendação: **comece pelo caminho A**. Além de evitar a classe inteira de problemas de
ambiente do Windows (versão errada, launcher ausente, política de execução, PATH), ele
roda o treino no mesmo Linux do CI, onde os valores de referência de
`evidencias/golden_metrics.json` foram congelados. Rodar o treino no Windows introduz
uma fonte de divergência numérica na última casa decimal (BLAS e bibliotecas de sistema
diferentes) que o caminho A elimina, o que é exatamente o que o objetivo específico OE6
pede demonstrar.

Nenhum dos caminhos exige credencial, conta ou serviço externo: o extrato de dados está
no próprio repositório e é verificado por SHA-256 antes de qualquer processamento.

**Convenção dos blocos de comando.** Os comandos são digitados no terminal, a partir da
pasta do repositório, um por linha. No Windows o terminal recomendado é o **PowerShell**
(o que abre no Windows Terminal ou pelo menu Iniciar); onde o comando do `cmd.exe`
clássico for diferente, ele aparece na tabela.

---

# Caminho A: tudo em Docker

Quatro passos até a pilha no ar. O Python, as bibliotecas de treino e as ferramentas de
teste vivem dentro da imagem `hypertension-train`; nada é instalado no host.

## Passo A1. Conferir os pré-requisitos

```bash
git --version
docker --version
docker compose version
```

| Ferramenta | Esperado | Observação |
|---|---|---|
| Git | 2.30 ou superior | https://git-scm.com |
| Docker Desktop | 24 ou superior, com Compose v2 | `docker compose version` precisa responder; `docker-compose` com hífen é a v1, descontinuada |

No Windows, o Docker Desktop precisa estar aberto e com o ícone verde, e usar o backend
WSL2 (padrão da instalação atual). Reserve cerca de 6 GB de disco para as imagens.

## Passo A2. Clonar o repositório e entrar na pasta

```bash
git clone https://github.com/OWNER/hypertension-mlops.git
cd hypertension-mlops
```

Como conferir: `dir` (Windows) ou `ls` (Linux/macOS) lista `configs`, `src`, `tests`,
`docker` e `README.md`.

Se em vez de clonar você recebeu o repositório como `.zip`, descompacte-o e entre na
pasta resultante: o restante dos passos é idêntico. O extrato `EXAMES-PNS-2013-FINAL_05052023.xlsx`
precisa estar em `data/raw/`, e o `.zip` já o inclui.

## Passo A3. Construir a imagem de treino

Esta imagem carrega Python 3.11 com `requirements-train.lock` e `requirements-dev.lock`
já instalados. Ela **não** copia o código: o repositório é montado em `/app` na hora de
executar, então editar um arquivo no host vale imediatamente, sem reconstruir nada.

```bash
docker compose -f docker/docker-compose.yml --profile tools build train
```

Duração: 5 a 10 minutos na primeira vez, dominada por CatBoost, LightGBM e XGBoost.
Como conferir: `docker images` lista `hypertension-train:latest`.

## Passo A4. Congelar a referência de fidelidade no ambiente da imagem

Rode uma vez, depois de construir a imagem, e sempre que `pandas`, `numpy` ou
`scikit-learn` mudarem de versão nos `.lock`.

```bash
docker compose -f docker/docker-compose.yml --profile tools run --rm train python scripts/freeze_notebook_reference.py
```

Por que isso existe: o gate de saída da etapa 2 compara a matriz de projeto do código
refatorado com números obtidos ao reexecutar as funções originais do estudo. A função
`impute_knn` do código original usa `KNNImputer`, cuja escolha de vizinhos muda entre
versões de `scikit-learn` e `numpy`; registros próximos do corte de 90 mL/min/1,73 m2
mudam de classe e o gate acusaria um erro de refatoração inexistente. Por isso a
referência vale apenas para o ambiente que a produziu, e esse ambiente é o da imagem.
ADR 0014 registra a decisão.

Duração: 3 a 8 minutos, dominados pela leitura do Excel três vezes. Como conferir: o
script imprime o ambiente, que deve casar com os `.lock`, e `git status` mostra
`evidencias/fase0_referencia_notebook_etapa2.json` modificado. **Commite esse arquivo**:
ele é evidência do TCC. Se ele não mudar, a referência já era a do seu ambiente.

**Rode dentro do container, não no Python do host.** O ambiente que vale é o da imagem
(Python 3.11, pandas e scikit-learn dos `.lock`). Congelar a referência em outro
interpretador gera números que não valem para nenhum ambiente do projeto, e o teste
`test_reference_was_frozen_in_this_environment` reprova com as duas versões lado a lado.

Com `make` disponível, o atalho é `make freeze-reference`.

## Passo A5. Executar o pipeline completo dentro do container

```bash
docker compose -f docker/docker-compose.yml --profile tools run --rm train
```

O serviço `train` já vem com o comando `python scripts/run_pipeline.py --no-docker`, que
roda as seis etapas na ordem descrita na tabela do passo B7. `--rm` remove o container
ao final; o que o pipeline escreve (`artifacts/`, `mlruns/`, `evidencias/`) aparece na
sua pasta, porque `/app` é o próprio repositório montado.

Duração: 10 a 20 minutos. Como conferir: existem `artifacts/lr_production/pipeline.joblib`
e `evidencias/experiments.md`.

**Em Linux**, acrescente `--user "$(id -u):$(id -g)"` ao `run` para que os arquivos
gerados pertençam ao seu usuário e não ao root. No Windows e no macOS isso é
desnecessário: o Docker Desktop cuida da tradução de propriedade.

## Passo A6. Subir a pilha de serviços

Só faz sentido depois do passo A5: a API carrega o modelo promovido em `artifacts/`.

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

O `--build` reconstrói as imagens da API e da interface. Diferente do serviço `train`,
que monta o repositório em `/app`, essas duas **copiam** o código na construção: sem o
`--build`, uma alteração em `src/` não chega aos contêineres. Depois da primeira vez,
enquanto o código não mudar, `up -d` basta e sobe em segundos.

O serviço `train` **não** sobe aqui, porque está no perfil `tools`: `up` traz apenas os
cinco serviços de execução.

| Serviço | Abra em | Para quê |
|---|---|---|
| Interface Streamlit | http://localhost:8501 | predição individual, lote, página do modelo e painel de monitoramento |
| API (documentação OpenAPI) | http://localhost:8000/docs | testar os endpoints pelo navegador |
| MLflow | http://localhost:5000 | experimentos e model registry |
| Prometheus | http://localhost:9090 | métricas brutas e regras de alerta |
| Grafana | http://localhost:3000 | dashboard provisionado (usuário `admin`, senha em `GRAFANA_ADMIN_PASSWORD`, padrão `admin`) |

Como conferir: `docker compose -f docker/docker-compose.yml ps` mostra os cinco serviços
como `running` e a API responde em `http://localhost:8000/health` (no PowerShell, `curl`
é apelido de `Invoke-WebRequest`: use `curl.exe` ou abra a URL no navegador).

## Passo A7. Encerrar

```bash
docker compose -f docker/docker-compose.yml down
```

Acrescente `-v` para apagar também os volumes de dados do Prometheus e do Grafana.

## Comandos do dia a dia no caminho A

Qualquer comando pode substituir o padrão do serviço `train`. O prefixo é sempre o
mesmo; abaixo ele aparece abreviado como `run --rm train`.

| O que você quer | Comando completo |
|---|---|
| Suíte de testes | `docker compose -f docker/docker-compose.yml --profile tools run --rm train python -m pytest -q` |
| Um teste específico | `... run --rm train python -m pytest tests/unit/test_drift.py -q` |
| Lint, formatação e tipos | `... run --rm train ruff check .`, `... run --rm train black --check .` e `... run --rm train mypy` |
| Só o treino | `... run --rm train python -m src.models.train --all` |
| Gate de qualidade de dados | `... run --rm train python -m src.data.quality` |
| Cenários de drift | `... run --rm train python scripts/baseline_scenario.py` e `... run --rm train python scripts/simulate_drift.py` |
| Conferir a referência de fidelidade | `... run --rm train python scripts/freeze_notebook_reference.py --check` |
| Congelar a referência de fidelidade | `... run --rm train python scripts/freeze_notebook_reference.py` |
| Relatório global do modelo | com a pilha no ar, pelo host: `curl.exe http://localhost:8000/model/summary` (no PowerShell use `curl.exe`, não `curl`) |
| Flag de retreinamento (decide e relata) | `... run --rm train python scripts/retrain.py --check-only` |
| Retreinar se a flag recomendar | `... run --rm train python scripts/retrain.py` |
| Um shell dentro do container | `... run --rm train bash` |

Com `make` disponível, os atalhos são `make docker-train` (construir),
`make pipeline-docker` (rodar o pipeline) e `make shell-docker` (abrir o shell).

## Travar as dependências transitivas (recomendado uma vez)

Os arquivos `.lock` do repositório fixam as dependências **diretas**; as transitivas
(por exemplo `anyio`, `pyparsing`) são resolvidas pelo `pip` na hora da instalação e
mudam com o tempo. Para congelar o grafo inteiro, com hashes, resolvido no mesmo Linux
do container, rode uma vez e commite os quatro `.lock` regenerados:

```bash
docker compose -f docker/docker-compose.yml --profile tools run --rm train make lock
```

Depois disso, reconstrua a imagem (`build train`) para que ela passe a usar o grafo
travado. `make freeze-docker` grava em `evidencias/requirements_frozen_train.txt` a
fotografia exata do ambiente que executou o pipeline, útil como evidência de
reprodutibilidade no texto.

## Limitações e cuidados do caminho A

| Assunto | O que saber |
|---|---|
| Pasta sincronizada (Google Drive, OneDrive) | O bind mount de uma pasta sincronizada é lento no Windows e pode travar arquivos em uso. Prefira um caminho local curto, como `C:\dev\hypertension-mlops` |
| IDE e autocomplete | O editor no host não enxerga as bibliotecas instaladas no container. Para autocomplete e depurador, use o caminho B em paralelo, ou a extensão Dev Containers do VS Code |
| Build de imagens dentro do container | O serviço `train` roda com `--no-docker` porque não há daemon acessível de dentro dele. As imagens da API e da interface são construídas no host, pelo passo A6 ou por `make docker-api` e `make docker-app` |
| Usuário root | A imagem de treino roda como root por padrão, por ser uma ferramenta local e efêmera. As imagens de serviço (`Dockerfile.api` e `Dockerfile.app`), que são as expostas em rede, continuam rodando como usuário não root |
| Teste de carga | Meça a partir do host (`python scripts/load_test.py --url http://localhost:8000 --container hypertension-mlops-api-1`) para que a latência inclua a rede, como a de um cliente real |

---

# Caminho B: Python local

Necessário para desenvolvimento interativo com IDE e depurador. São oito passos, e o
primeiro cuida justamente da armadilha mais comum, a versão do Python.

## Passo B1. Conferir os pré-requisitos

Antes de clonar, verifique se as três ferramentas estão instaladas e em versão
compatível.

```bash
git --version
docker --version
docker compose version
```

| Ferramenta | Esperado | Onde obter | Necessária para |
|---|---|---|---|
| Git | 2.30 ou superior | https://git-scm.com | clonar o repositório |
| Python | **3.11.x, e apenas 3.11** | https://www.python.org/downloads/release/python-3119/ ou Miniconda | todo o pipeline |
| Docker Desktop | 24 ou superior, com Compose v2 (`docker compose`, sem hífen) | https://docker.com | API, interface, MLflow, Prometheus e Grafana em contêiner |

Neste caminho o Docker é necessário apenas nos passos B7 e B8.

### A versão do Python não é negociável

O projeto declara `requires-python = ">=3.11,<3.12"` e os arquivos `.lock` fixam o trio
numérico de referência (pandas 2.2.3, numpy 1.26.4, scikit-learn 1.5.2) resolvido para
3.11. Em Python 3.12 ou superior a instalação **falha** com
`Package 'hypertension-mlops' requires a different Python`. A restrição existe para que
as métricas sejam reproduzíveis: o modelo campeão é comparado com valores congelados em
`evidencias/golden_metrics.json`.

Descubra quais versões estão instaladas antes de criar o ambiente. Rode as quatro
linhas: as que falharem apenas indicam que aquele caminho não existe na sua máquina.

```powershell
conda --version
where.exe python
py --list
python --version
```

| Comando | O que mostra | Observação |
|---|---|---|
| `conda --version` | se o conda está disponível | se responder, este é o caminho mais curto: pule para a alternativa com conda no passo B5 |
| `where.exe python` | todos os `python.exe` do PATH, na ordem de precedência | no PowerShell precisa ser `where.exe`; `where` sozinho é apelido de `Where-Object` |
| `py --list` | todas as versões instaladas, com `-V:3.11` na lista se houver | **o `py` só existe se o Python veio do instalador do python.org**; a versão da Microsoft Store e o Anaconda não o incluem, e o erro "o termo 'py' não é reconhecido" significa apenas isso |
| `python --version` | qual versão o `python` genérico resolve | é a que um `python -m venv` usaria |

Em Linux ou macOS, o equivalente é `which -a python3` e `python3.11 --version`.

Se nenhuma das linhas apontar um Python 3.11, escolha um destes dois caminhos:

| Caminho | Como | Vantagem |
|---|---|---|
| **Conda** (recomendado se você já usa Anaconda ou Miniconda) | `conda env create -f environment.yml` no passo B5 | o conda baixa o Python 3.11.9 sozinho, sem instalar nada no sistema e sem tocar no seu Python atual |
| Instalador oficial | baixe o Python 3.11.9 em https://www.python.org/downloads/release/python-3119/, marque "Add python.exe to PATH" | instala também o launcher `py`, que passa a permitir `py -3.11` |

Instalar o 3.11 não remove nem substitui o 3.12: as duas versões convivem, e é por isso
que o ambiente precisa ser criado chamando a versão certa de forma explícita.

### Onde colocar o repositório (Windows)

Prefira um caminho local curto, como `C:\dev\hypertension-mlops`. Pastas sincronizadas
(Google Drive, OneDrive, Dropbox) funcionam, mas o `.venv` tem dezenas de milhares de
arquivos pequenos: a sincronização deixa o `pip install` e o `pytest` lentos e pode
travar arquivos em uso. Se mantiver o projeto no Drive, exclua `.venv`, `mlruns` e
`artifacts` da sincronização nas preferências do aplicativo.

## Passo B2. Clonar o repositório e entrar na pasta

`git clone` baixa o código e o histórico para uma pasta nova chamada
`hypertension-mlops`; `cd` entra nessa pasta, que é a raiz do projeto e o diretório de
trabalho de todos os comandos seguintes.

```bash
git clone https://github.com/OWNER/hypertension-mlops.git
cd hypertension-mlops
```

Como conferir: `dir` (Windows) ou `ls` (Linux/macOS) deve listar `configs`, `src`,
`tests`, `docker` e `README.md`.

## Passo B3. Criar o ambiente virtual isolado

Um ambiente virtual é uma cópia isolada do Python onde as bibliotecas deste projeto são
instaladas sem afetar o Python do sistema nem outros projetos. O comando abaixo cria a
pasta `.venv` dentro do repositório (ela já está no `.gitignore`, não é versionada).

**Chame o 3.11 explicitamente**, e não o `python` genérico: o ambiente herda para sempre
a versão de quem o criou.

| Situação | Comando |
|---|---|
| Windows, com o launcher `py` | `py -3.11 -m venv .venv` |
| Windows, sem o `py` | use o caminho completo do executável, por exemplo `& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv` (o `where.exe python` do passo B1 mostra os caminhos reais) |
| Linux, macOS | `python3.11 -m venv .venv` |
| Conda | não crie venv: o `conda env create` do passo B5 já cria o ambiente com o Python correto |

Se você já criou o `.venv` com a versão errada, não dá para consertar por dentro: apague
a pasta e recrie.

| Terminal | Comandos |
|---|---|
| PowerShell | `deactivate` (se estiver ativo), depois `Remove-Item -Recurse -Force .venv` e `py -3.11 -m venv .venv` |
| Linux, macOS, Git Bash | `deactivate`, depois `rm -rf .venv` e `python3.11 -m venv .venv` |

Escolha a alternativa conda se preferir; nesse caso pule para a seção "Alternativa com
conda" ao fim deste caminho.

## Passo B4. Ativar o ambiente virtual

Criar não basta: é preciso **ativar** o ambiente em cada terminal novo, para que
`python` e `pip` passem a apontar para dentro do `.venv`. O comando de ativação depende
do sistema e do terminal.

| Sistema e terminal | Comando de ativação |
|---|---|
| Windows, PowerShell (recomendado) | `.venv\Scripts\Activate.ps1` |
| Windows, `cmd.exe` | `.venv\Scripts\activate.bat` |
| Windows, Git Bash | `source .venv/Scripts/activate` |
| Linux ou macOS | `source .venv/bin/activate` |

Como conferir: o nome `(.venv)` passa a aparecer no início da linha do terminal, e o
comando abaixo deve responder `3.11.x` e um caminho terminado em `.venv`. Confira as
duas saídas antes de seguir; é aqui que se detecta um ambiente criado com a versão
errada.

```bash
python --version
python -c "import sys; print(sys.prefix)"
```

Se o PowerShell recusar a ativação com a mensagem "execução de scripts foi desabilitada
neste sistema", libere os scripts apenas para o seu usuário e ative de novo:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```

Toda vez que você fechar o terminal e voltar ao projeto, repita apenas este passo B4.

## Passo B5. Instalar as dependências travadas

O primeiro comando instala as bibliotecas de treino e de desenvolvimento com versão
exata, a partir dos arquivos `.lock`. O segundo instala o próprio projeto em modo
editável (`-e`), o que torna `import src...` possível de qualquer pasta; `--no-deps`
evita que o pip tente resolver dependências de novo e quebre as versões travadas.

```bash
pip install -r requirements-train.lock -r requirements-dev.lock
pip install -e . --no-deps
```

Duração: 2 a 5 minutos, conforme a conexão. Como conferir:

```bash
python -c "import src, sklearn, mlflow, streamlit; print('ambiente ok')"
```

Existem quatro arquivos de dependências, com propósitos diferentes:

| Arquivo | Conteúdo | Usado por |
|---|---|---|
| `requirements-api.lock` | runtime mínimo da API (sem bibliotecas de treino) | imagem `Dockerfile.api` |
| `requirements-app.lock` | runtime da interface (cliente HTTP, sem modelo) | imagem `Dockerfile.app` |
| `requirements-train.lock` | tudo da API mais treino (XGBoost, LightGBM, CatBoost, MLflow, Streamlit) | sua máquina e o CI |
| `requirements-dev.lock` | ferramentas de qualidade e teste (ruff, black, mypy, pytest, bandit) | sua máquina e o CI |

### Alternativa com conda (no lugar dos passos B3, B4 e B5)

```bash
conda env create -f environment.yml
conda activate hypertension-mlops
```

Duas linhas apenas: o `environment.yml` fixa Python 3.11.9, instala as mesmas
dependências a partir dos `.lock` e já inclui o projeto em modo editável, portanto o
`pip install` do passo B5 não é necessário. `conda activate` substitui o passo B4 e
precisa ser repetido a cada terminal novo.

## Passo B6. Congelar a referência de fidelidade neste ambiente

```bash
python scripts/freeze_notebook_reference.py
```

Mesmo motivo do passo A4 do caminho A: a referência da etapa 2 vale apenas para o
ambiente que a produziu, porque o `KNNImputer` do código original escolhe vizinhos
diferentes entre versões de `scikit-learn` e `numpy`. Commite o JSON alterado. ADR 0014
registra a decisão. Para apenas conferir, sem escrever, use `--check`.

Atenção: se você também usa o caminho A, escolha **um** ambiente para congelar a
referência e fique nele. Congelar aqui e rodar os testes no container (ou o contrário)
reprova o gate, porque as versões diferem. O ambiente da imagem é o do CI, então é o
mais seguro dos dois.

## Passo B7. Executar o pipeline completo com um comando

Este é o comando único exigido pelo objetivo específico OE1. Ele roda as seis etapas em
sequência, em subprocessos, e para na primeira falha com código de saída diferente de
zero.

```bash
python scripts/run_pipeline.py
```

O que acontece, na ordem:

| Ordem | Etapa | Comando equivalente | Resultado |
|---|---|---|---|
| 1 | Gera a fixture sintética | `python scripts/make_synthetic_fixture.py` | `tests/fixtures/synthetic_sample.parquet` |
| 2 | Gate de qualidade na fixture | `python -m src.data.quality --fixture` | falha se o gerador sair do domínio declarado |
| 3 | Gate de qualidade no extrato real | `python -m src.data.quality` | confere SHA-256, dimensões, contagens dos filtros, nulos e domínio das 61 variáveis |
| 4 | Treino, validação e rastreamento | `python -m src.models.train --all` | dez runs no MLflow (sete, se XGBoost, LightGBM e CatBoost não estiverem instalados), campeão promovido, `artifacts/` e `evidencias/` |
| 5 | Testes automatizados | `python -m pytest -q` | suíte completa: unitários, integração e contrato |
| 6 | Cenário de drift baseline | `python scripts/baseline_scenario.py` | `evidencias/drift_baseline.*`, sem alerta |
| 7 | Cenário de drift simulado | `python scripts/simulate_drift.py` | `evidencias/drift_simulated.*`, com alerta |
| 8 | Build das imagens Docker | `docker build -f docker/Dockerfile.api ...` e `Dockerfile.app` | `hypertension-api:latest` e `hypertension-app:latest` |

Duração total: 10 a 20 minutos em máquina de desenvolvimento, dominada pelo treino de
CatBoost e Random Forest. Opções úteis:

| Comando | Quando usar |
|---|---|
| `python scripts/run_pipeline.py --no-docker` | máquina sem Docker, ou para pular só a ação 8 da tabela acima |
| `python scripts/run_pipeline.py --backend json` | sem MLflow instalado: as runs saem em `evidencias/runs/*.json` |
| `make all` | atalho equivalente, se você tiver `make` (opcional; cada alvo do `Makefile` é uma linha Python pura, copiável no Windows) |

Como conferir: a última linha impressa é "Pipeline concluído" e existem os arquivos
`artifacts/lr_production/pipeline.joblib` e `evidencias/experiments.md`.

## Passo B8. Subir a pilha de serviços

Sobe cinco contêineres em rede própria: API, interface, MLflow, Prometheus e Grafana.
A flag `-d` (detached) devolve o terminal; sem ela os logs ficam em primeiro plano.

```bash
docker compose -f docker/docker-compose.yml up -d
```

Na primeira execução, se as imagens ainda não existirem, o Compose as constrói (leva
alguns minutos). Para forçar a reconstrução depois de alterar o código, acrescente
`--build`.

| Serviço | Abra em | Para quê |
|---|---|---|
| Interface Streamlit | http://localhost:8501 | predição individual, lote, página do modelo e painel de monitoramento |
| API (documentação OpenAPI) | http://localhost:8000/docs | testar os endpoints pelo navegador |
| MLflow | http://localhost:5000 | experimentos e model registry |
| Prometheus | http://localhost:9090 | métricas brutas e regras de alerta |
| Grafana | http://localhost:3000 | dashboard provisionado (usuário `admin`, senha em `GRAFANA_ADMIN_PASSWORD`, padrão `admin`) |

Como conferir: `docker compose -f docker/docker-compose.yml ps` mostra os cinco
serviços como `running`, e a API responde:

```bash
curl http://localhost:8000/health
```

No PowerShell, `curl` é um apelido de `Invoke-WebRequest`; use `curl.exe` ou abra a
URL no navegador.

## Passo B9. Encerrar

`down` para e remove os contêineres (os volumes nomeados do Prometheus e do Grafana
permanecem; acrescente `-v` para apagá-los também). `deactivate` devolve o terminal ao
Python do sistema.

```bash
docker compose -f docker/docker-compose.yml down
deactivate
```

## Erros comuns

| Sintoma | Causa provável | Solução |
|---|---|---|
| `AssertionError: egfr_class` em `test_encoded_matrix_is_identical_to_notebook` | a referência de fidelidade foi congelada em outro ambiente | rode `scripts/freeze_notebook_reference.py` neste ambiente (passo A4 ou B6) e commite o JSON; ADR 0014 |
| `reference frozen with {...}, running with {...}` | mesmo caso acima, detectado antes pelo teste de ambiente | idem |
| `no configuration file provided: not found` em um comando `docker compose` | faltou `-f docker/docker-compose.yml`: o arquivo não está na raiz | acrescente o `-f`, ou rode a partir de `docker/` |
| `Package 'hypertension-mlops' requires a different Python: 3.12.x not in '<3.12,>=3.11'` | o `.venv` foi criado com Python 3.12 ou superior | apague e recrie o ambiente com o 3.11 (tabela do passo B3); não há como converter o ambiente existente |
| `O termo 'py' não é reconhecido` | o launcher `py` não foi instalado: o Python veio da Microsoft Store ou do Anaconda | não é erro do projeto; use `where.exe python` e `conda --version` para localizar seus interpretadores (passo B1) e crie o ambiente pelo conda ou pelo caminho completo do executável |
| `where` devolve a ajuda do `Where-Object` | no PowerShell `where` é apelido de cmdlet | use `where.exe python` |
| `ERROR: ResolutionImpossible` ou `Cannot install -r requirements-train.lock ... conflicting dependencies` | `.lock` desatualizado, anterior à correção da etapa 7 | `git pull` (ou baixe o `requirements-train.lock` atual) e repita o passo B5; o conflito era `great-expectations` contra `pandas==2.2.3`, resolvido no ADR 0012 |
| `ModuleNotFoundError: No module named 'sklearn'` logo após um erro do `pip` | a instalação abortou no meio, nada foi instalado | corrija a causa do erro anterior e rode o passo B5 de novo; a mensagem é consequência, não causa |
| `Activate.ps1 ... não pode ser carregado` | política de execução do PowerShell | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` (passo B4) |
| `pip` instala fora do `.venv` | ambiente não ativado | repita o passo B4 e confira com `python -c "import sys; print(sys.prefix)"` |
| `ModuleNotFoundError: No module named 'src'` | faltou `pip install -e . --no-deps` | repita o passo B5 |
| `pip install` muito lento ou travado no Windows | repositório em pasta sincronizada (Drive, OneDrive) | mova o projeto para um caminho local ou exclua `.venv` da sincronização (passo B1) |
| `Cannot connect to the Docker daemon` | Docker Desktop fechado | abra o Docker Desktop e espere o ícone ficar verde |
| `port is already allocated` | porta 8000, 8501, 5000, 9090 ou 3000 em uso | encerre o processo que a ocupa ou altere `ports` em `docker/docker-compose.yml` |
| `ModelNotFoundError` ao subir a API | pipeline não executado, logo não há modelo em `artifacts/` | rode o passo B7 antes do passo B8 |
| Gate de qualidade falha no SHA-256 | o arquivo em `data/raw/` foi alterado | restaure com `git checkout -- data/raw` |

---

# Operação e verificação

## Verificação local (qualidade e testes)

Os mesmos comandos que o CI executa no job `quality`. Rode antes de cada commit.

| Comando | O que verifica | Falha quando |
|---|---|---|
| `ruff check .` | lint e ordenação de imports | há violação de regra em `pyproject.toml` |
| `black --check .` | formatação | algum arquivo está fora do formato |
| `mypy` | tipagem estática em modo estrito | há tipo faltante ou incompatível |
| `pytest --cov` | suíte completa e cobertura | algum teste quebra ou a cobertura cai abaixo de 80 % |

Atalho com `make`: `make check` roda lint, mypy e testes em sequência.

## Relatório de resultados

`notebooks/relatorio_resultados.ipynb` regenera, em uma execução, todas as tabelas e
figuras da seção de Resultados do TCC a partir dos artefatos que o pipeline já
persistiu. Ele **não** treina nem transforma nada: é camada de relatório sobre
`artifacts/` e `evidencias/`, o que garante que os números do texto e os do pipeline
sejam sempre os mesmos.

Pré-requisito: o pipeline precisa ter rodado ao menos uma vez
(`python scripts/run_pipeline.py --no-docker`), porque o relatório lê os artefatos.

### Modo 1: gerar sem abrir o navegador (recomendado)

```bash
# Docker
docker compose -f docker/docker-compose.yml --profile tools run --rm train python scripts/run_relatorio.py --html

# Python local, com o ambiente ativado
python scripts/run_relatorio.py --html
```

Não precisa de porta, de token nem de Jupyter instalado: o executor lê o `.ipynb` e
roda as células com as bibliotecas que já estão no ambiente de treino.

As saídas vão para `evidencias/relatorio_resultados/`:

| Arquivo | Conteúdo |
|---|---|
| `Relatorio.html` | documento único e autocontido, com as figuras embutidas |
| `Resultados_Regerados.md` | as tabelas e legendas na ordem da seção, para colar no TCC |
| `tabelas/*.csv` | precisão completa, sem arredondamento, para auditoria |
| `tabelas/*.md` | formato brasileiro, com vírgula decimal |
| `figuras/*.png` | 300 dpi, sem título e sem borda (item 15.1 do Manual) |
| `proveniencia.json` | ambiente, commit, versões e resumos SHA-256 da execução |
| `PENDENCIAS.md` | o que ficou faltando, por que, e o comando que resolve |

Opções: `--sem-codigo` omite os blocos de código do HTML e `--ate N` para na célula
N, para depurar uma seção isolada.

### Modo 2: abrir o `Relatorio.html` no navegador

Depois de gerar com `--html`, dê duplo clique em
`evidencias/relatorio_resultados/Relatorio.html`, ou abra pela linha de comando:

```bash
start evidencias\relatorio_resultados\Relatorio.html   # Windows
xdg-open evidencias/relatorio_resultados/Relatorio.html  # Linux
```

É um arquivo estático: abre offline, sem servidor e sem senha, e pode ser anexado
em e-mail.

### Modo 3: abrir o notebook no navegador, para editar célula a célula

O Jupyter **não** faz parte da imagem de treino nem do lock, porque é dependência de
relatório e não de treino. Instale-o no contêiner descartável:

```bash
docker compose -f docker/docker-compose.yml --profile tools run --rm -p 8888:8888 -e JUPYTER_TOKEN=tcc train sh -lc "pip install jupyterlab && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
```

Com o ambiente Python local, basta `pip install jupyterlab` uma vez e depois
`jupyter lab` na raiz do repositório.

Endereço:

```
http://localhost:8888/lab/tree/notebooks/relatorio_resultados.ipynb
```

O campo **"Password or token"** recebe `tcc`. Depois, menu **Run → Run All Cells**.

> A instalação acima não está travada por versão. Sirva-se dela para inspecionar e
> editar; a execução que vira entrega deve sair do modo 1, para que o
> `proveniencia.json` descreva um ambiente reconstruível.

### Erros comuns

| Mensagem | Causa | Correção |
|---|---|---|
| `no configuration file provided: not found` | `docker compose` executado fora da raiz, ou sem `-f docker/docker-compose.yml` | rode da raiz do repositório com o `-f` |
| `executable file not found in $PATH: jupyter` | Jupyter não está na imagem | use o modo 1, ou o comando do modo 3 com `pip install` |
| `Invalid credentials` na tela do Jupyter | o token informado não foi o gerado | use `-e JUPYTER_TOKEN=...`, ou copie a URL com `?token=` que o terminal imprime |
| A barra invertida `\` não continua a linha | PowerShell usa crase e o CMD usa `^` | escreva o comando em uma linha só |
| Seções saem como `[PENDENTE]` | o artefato correspondente não existe | veja `PENDENCIAS.md`: cada linha traz o comando que a resolve |

### O que o relatório não produz

As Figuras 11, 12 e 13 são capturas de tela de serviços no ar (registro de modelos,
interface e painel de observabilidade) e a linha do contêiner na Tabela 14 exige a
API em execução. Suba a pilha com `docker compose -f docker/docker-compose.yml up -d`
antes de capturá-las. Os comandos exatos saem em `PENDENCIAS.md`.

## Estrutura de pastas

```
configs/      config.yaml (todos os parâmetros) e variable_mapping.yaml
data/         raw (versionado), interim e processed (gerados)
src/          data, features, models, monitoring, api, app
tests/        unit, integration, contract, fixtures
docker/       Dockerfile.api, Dockerfile.app, Dockerfile.train, docker-compose.yml, prometheus/, grafana/
scripts/      pipeline completo, fixture sintética, cenários de drift, teste de carga, relatório de resultados
.github/      workflows de CI/CD
docs/         inventário, ADRs, model card, dicionário de dados, runbook, arquitetura
notebooks/    artefatos originais e relatorio_resultados.ipynb, fora do caminho de execução
evidencias/   saídas que alimentam Evidencias_Consolidadas.md e relatorio_resultados/ (seção de Resultados regerada)
```

## Pipeline de dados (etapa 2)

Cada comando roda isoladamente, quando você quiser reexecutar só uma parte.

| Comando | O que faz |
|---|---|
| `python -m src.data.quality` | ingestão do xlsx, verificação de SHA-256, filtros populacionais e gate de qualidade sobre o dado real |
| `python -m src.data.quality --fixture` | o mesmo sobre a fixture sintética (é o que roda no CI sem o dado real) |
| `python -m src.data.quality --report evidencias/data_quality_real.json` | grava o relatório do gate em JSON |
| `python scripts/make_synthetic_fixture.py` | regenera `tests/fixtures/synthetic_sample.parquet` a partir do gerador versionado |

O transformador (`src/features/transformer.py`) reproduz o pré-processamento do
notebook original: no modo `fidelity` a matriz codificada é idêntica à do notebook,
célula a célula, nos conjuntos de desenvolvimento, treino e teste (verificado por
SHA-256 em `tests/integration/test_notebook_fidelity.py`). No modo `production` a
imputação é ajustada apenas no treino e aplicada às demais observações.

## Treino e rastreamento (etapa 3)

| Comando | O que faz |
|---|---|
| `python -m src.models.train --all` | executa as dez configurações, registra no MLflow (SQLite em `mlruns/`) e promove o campeão |
| `python -m src.models.train --all --backend json` | o mesmo sem MLflow: as runs saem em `evidencias/runs/` |
| `python -m src.models.train --run lr_fidelity` | executa uma única run (o nome vem de `configs/config.yaml::model.runs`) |
| `python -m src.models.train --run lr_fidelity --freeze-golden` | recongela `evidencias/golden_metrics.json`, a referência do teste de regressão (decisão P4) |
| `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` | abre a interface do MLflow sem subir o Compose |

Cada run registra parâmetros, métricas de validação cruzada (k=10) e de holdout sem
arredondamento, SHA-256 do dado bruto e do conjunto de desenvolvimento, revisão do
código e hash do código do transformador, além dos artefatos `joblib` (transformador,
imputadores KNN, encoder, modelo e pipeline) e da janela de referência de drift. O run
`lr_production` é registrado como `hypertension-lr` e promovido ao estágio `Production`
com o alias `champion`.

## API (etapa 5)

| Comando | O que faz |
|---|---|
| `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` | sobe a API fora do Docker, servindo o modelo local (`artifacts/lr_production`) |
| `docker build -f docker/Dockerfile.api -t hypertension-api:latest .` | constrói a imagem (multi-stage, usuário não root) a partir da raiz do repositório |
| `docker run --rm -p 8000:8000 -v "${PWD}/artifacts:/app/artifacts:ro" hypertension-api:latest` | roda a imagem montando os artefatos em somente leitura (no bash use `"$PWD/..."`) |
| `curl http://localhost:8000/health` | confere se o modelo carregou, com versão e origem |

Para servir o modelo a partir do registry do MLflow em vez do diretório local, defina a
variável de ambiente antes de subir a API:

| Terminal | Comando |
|---|---|
| PowerShell | `$env:HTN_API__MODEL_SOURCE="mlflow"; uvicorn src.api.main:app` |
| Linux, macOS, Git Bash | `HTN_API__MODEL_SOURCE=mlflow uvicorn src.api.main:app` |

| Endpoint | Método | Conteúdo |
|---|---|---|
| `/predict?explain=true` | POST | probabilidade, faixa de risco, classe, threshold, nome/versão/estágio do modelo, campos imputados (P3), cinco maiores contribuições em log-odds (ADR 0003), disclaimer, correlation id |
| `/predict/batch` | POST | lista de predições na ordem recebida, contagem de linhas com imputação |
| `/health` | GET | liveness e readiness (modelo carregado, versão, origem) |
| `/metrics` | GET | exposição Prometheus: histograma de latência, contador por endpoint e status, predições por faixa de risco, erros por tipo, gauges de drift |
| `/model/summary` | GET | métricas do modelo em serviço: validação cruzada k=10, holdout, matriz de confusão, versão e estágio |
| `/model/roc` | GET | curva ROC de cada run treinado junto do campeão, no holdout, mais sensibilidade e especificidade por limiar |
| `/model/importance` | GET | importância por variável do modelo e coeficientes por categoria, com sinal |
| `/model/contributions` | GET | amostra de contribuições lineares no holdout (ADR 0003: não é SHAP) |
| `/model/subgroups` | GET | desempenho fora das dobras por sexo e faixa etária |

As cinco rotas `/model` são somente leitura, não executam inferência e leem apenas os
artefatos do run em serviço. Cada resposta traz `available` e, quando falso, `reason`:
um modelo treinado por uma versão anterior simplesmente não tem o artefato, e isso vira
mensagem na interface em vez de erro (ADR 0015).

**Autenticação opcional.** Sem a variável `HTN_API__AUTH_TOKEN` a API fica aberta, que é
o padrão local deste estudo, e a API registra um aviso na subida. Com a variável
definida, `/predict`, `/predict/batch` e as rotas `/model` exigem
`Authorization: Bearer <token>`; `/health` e `/metrics` continuam abertos, porque o
Prometheus os raspa e uma sonda de saúde que precisa de credencial falha pelo motivo
errado. O token nunca vai para o `config.yaml` nem para o código: só variável de ambiente
ou GitHub Secret. A interface envia o mesmo token por `HTN_APP__API_TOKEN`.

Toda resposta carrega `X-Correlation-ID` (propagado do cliente ou gerado). Erros de
domínio viram envelopes JSON `{error, detail, correlation_id}` sem stack trace. A
documentação OpenAPI fica em `/docs`.

## Interface, monitoramento e drift (etapa 6)

| Comando | O que faz |
|---|---|
| `streamlit run src/app/main.py` | sobe a interface fora do Docker; ela consome a API em `app.api_url` |
| `docker compose -f docker/docker-compose.yml up --build -d` | sobe a pilha inteira reconstruindo as imagens |
| `python scripts/baseline_scenario.py` | cenário sem alteração: espera-se nenhuma variável em alerta |
| `python scripts/simulate_drift.py` | cenário com deslocamentos determinísticos: espera-se alerta em idade, peso, glicose e região |
| `python scripts/simulate_drift.py --api-url http://localhost:8000` | o mesmo cenário enviado pela API, lendo o resultado de volta em `/metrics` |
| `python scripts/load_test.py --url http://localhost:8000 --container hypertension-mlops-api-1` | teste de carga: p95, throughput, CPU e memória do contêiner |

Os dois cenários encerram com código de saída 0 quando o resultado é o esperado, o que
os torna utilizáveis como gate no CI. Com `--fixture` eles rodam sobre dados sintéticos
e gravam em `evidencias/drift_*_fixture.*`, sem sobrescrever a evidência do dado real.

| Serviço | Porta | Função |
|---|---|---|
| API | 8000 | `/predict`, `/predict/batch`, `/health`, `/metrics` (RED, predições por faixa, gauges de drift) |
| Interface | 8501 | predição individual em quatro etapas com revisão e explicação local; lote via CSV com validação linha a linha; página do modelo (desempenho do campeão, ROC por run, importância, coeficientes, contribuições e subgrupos); painel de monitoramento (p95, volume, erros, gate de contrato, PSI/KS, modelo ativo) |
| MLflow | 5000 | UI de experimentos e registry sobre `mlruns/mlflow.db` |
| Prometheus | 9090 | raspagem de `/metrics` a cada 15 s e regras de alerta (`docker/prometheus/alerts.yml`) |
| Grafana | 3000 | dashboard provisionado por arquivo (`docker/grafana/dashboards/hypertension_api.json`) |

A interface é um cliente HTTP puro: nunca importa o modelo nem acessa o MLflow. A
barra lateral mostra a versão em serviço e as quatro métricas do holdout, e a página
Modelo separa o que é estático (desempenho do campeão, medido uma vez no treino) do que
é vivo (latência, drift e erros, no painel de monitoramento). O drift
compara a janela deslizante de observações recebidas pela API (em memória,
`monitoring.window_size`) com a referência do treino persistida junto ao modelo
(`artifacts/lr_production/drift_reference.json`): PSI por variável (com a ausência como
categoria própria) e KS nas numéricas, limiares em `monitoring.drift`. O SLO de latência
é declarado para quatro clientes concorrentes sobre um único worker
(`load_test.concurrency`); mais workers (`UVICORN_WORKERS`) multiplicam o throughput,
mas fragmentam a janela de drift e as métricas entre processos.

## Configuração

Todos os parâmetros vivem em `configs/config.yaml` e são carregados por `src/config.py`
com validação estrita: chave faltante ou desconhecida interrompe a execução com a lista
dos campos problemáticos. Qualquer valor pode ser sobrescrito por variável de ambiente
com prefixo `HTN_` e separador `__` entre os níveis. Exemplos:

| Variável | Efeito |
|---|---|
| `HTN_LOGGING__LEVEL=DEBUG` | eleva o nível do log estruturado |
| `HTN_API__MODEL_SOURCE=mlflow` | serve o modelo do registry em vez do diretório local |
| `HTN_APP__API_URL=http://localhost:8000` | aponta a interface para outra API |

## Dados

`data/raw/EXAMES-PNS-2013-FINAL_05052023.xlsx` deriva dos microdados públicos da
**Pesquisa Nacional de Saúde 2013**, do Instituto Brasileiro de Geografia e Estatística
(IBGE), módulo de exames laboratoriais. Os dados são secundários, públicos e
anonimizados; não contêm identificador de domicílio, pessoa ou município. SHA-256 do
arquivo: `80222fa535524829d531504109ac4b49d44e4e6d0c04c3c281ff972bd332d1f7`. Origem,
termos de uso, data de coleta e data de congelamento são detalhados em
`docs/dicionario_dados.md`.

Atribuição: IBGE, Pesquisa Nacional de Saúde 2013. Os dados pertencem ao IBGE e são
redistribuídos aqui apenas para fins de reprodutibilidade acadêmica.

## Documentação

| Documento | Conteúdo |
|---|---|
| `docs/arquitetura.md` | Diagramas do pipeline, dos componentes em execução e do fluxo de uma requisição |
| `docs/adr/` | Dezesseis registros de decisão (MLflow, feature store, PSI/KS próprios, P1, P2, D6, canary/autoscaling, P3, ambiente da referência, página de modelo, algoritmos, entre outros) |
| `docs/model_card.md` | Uso pretendido, população de treino, métricas globais e por subgrupo, limitações |
| `docs/dicionario_dados.md` | 61 variáveis brutas e 33 do modelo, origem, licença, data freeze, SHA-256 (gerado por script) |
| `docs/runbook.md` | Alertas, incidentes, flag de retreinamento e rollback de versão |
| `docs/slo.md` | Indicadores, objetivos, orçamento de erro e o que o SLO não cobre |
| `docs/politica_versionamento.md` | Versionamento de dados por hash, de código por tag e de modelo por registry |
| `docs/glossario.md` | Termos de aprendizado de máquina e MLOps como usados no projeto |
| `docs/checklist_privacidade.md` | Risco de reidentificação, retenção e descarte |
| `docs/listagens_tcc.md` | Listagens curadas S07 a S12 e Apêndice C |
| `evidencias/Evidencias_Consolidadas.md` | Todas as tabelas, números e figuras consumidos pelo texto do TCC |

## Referência

DIAS, Jaqueline Lopes. **Aprendizado de máquina aplicado à predição de doenças
crônicas**: um estudo de caso de hipertensão arterial. 2024. Dissertação (Mestrado
Profissional) - ICMC, Universidade de São Paulo, São Carlos, 2024.

## Licença

Código sob licença MIT (`LICENSE`). Os dados seguem os termos do IBGE.

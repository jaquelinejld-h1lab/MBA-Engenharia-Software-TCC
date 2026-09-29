# Glossário de aprendizado de máquina e MLOps

Termos como são usados neste repositório e no TCC. Onde um termo tem mais de um sentido
corrente, a coluna de observação diz qual foi adotado e por quê.

## Aprendizado de máquina

| Termo | Definição | Neste projeto |
|---|---|---|
| **AUC-ROC** | Área sob a curva ROC. Probabilidade de o modelo dar escore maior a um positivo sorteado ao acaso do que a um negativo | Métrica principal de discriminação; critério de aceitação `min_auc = 0,73` |
| **Calibração** | Quão próximo o escore está da probabilidade real do evento | Não calibrado. A faixa de risco é ordenação, não probabilidade absoluta (limitação do model card) |
| **Conjunto de desenvolvimento** | Dados usados para treinar e validar, antes de separar o teste | 6.203 registros após os filtros populacionais |
| **Dummy** | Coluna binária que representa uma categoria de uma variável categórica | 85 dummies geradas das 33 variáveis do modelo, com `drop='first'` |
| **Especificidade** | Proporção de negativos corretamente classificados | Contraparte da sensibilidade no ponto de operação |
| **Holdout** | Partição separada antes do treino, usada uma vez ao final | 1.241 registros, 20 % estratificado |
| **Imputação** | Preenchimento de valores ausentes | KNN com k=5 nos três exames laboratoriais, ajustado apenas no treino no modo produção |
| **Limiar de decisão** | Corte do escore que separa "com risco" de "sem risco" | 0,5, declarado em config; a curva por limiar está na página Modelo |
| **Prevalência** | Proporção de positivos na população | 17,04 % no conjunto de desenvolvimento |
| **Sensibilidade** (recall) | Proporção de positivos corretamente identificados | Métrica privilegiada: em rastreio, deixar de sinalizar custa mais que um falso positivo. Aceitação `min_sensitivity = 0,65` |
| **SHAP** | Valores de Shapley aplicados a atributos, atribuindo a cada um sua contribuição à predição | Não usado. Para regressão logística a contribuição linear `coef × valor` é exata e dispensa a aproximação (ADR 0003) |
| **Validação cruzada estratificada** | Divisão em k partes preservando a proporção de classes; cada parte é teste uma vez | k = 10, semente 42, com o transformador reajustado a cada fold |
| **Vazamento (leakage)** | Informação do teste influenciando o treino | Evitado ajustando imputador e codificador só no treino; o modo fidelidade reproduz o vazamento do estudo original de propósito e está documentado |

## Engenharia e operação de modelos

| Termo | Definição | Neste projeto |
|---|---|---|
| **Artefato** | Arquivo produzido por uma execução e guardado para reuso ou auditoria | Transformador, imputadores, encoder, modelo, pipeline, métricas, ROC, interpretação e referência de drift |
| **Campeão** (champion) | Versão do modelo em serviço | `lr_production`, alias `champion` no registry |
| **Canary release** | Liberar uma versão nova para uma fração do tráfego antes de todo ele | Inviável sem nuvem; registrado como ausência justificada (ADR 0010) |
| **Contrato de dados** | Definição executável do esquema, domínios e obrigatoriedade de cada campo | `src/data/contract.py`, gerado do mapeamento; rejeita código fora do domínio |
| **Deriva de conceito** (concept drift) | Mudança na relação entre atributos e alvo | Não medida: exigiria rótulos novos, que um sistema de rastreio não recebe de imediato |
| **Deriva de dados** (data drift) | Mudança na distribuição dos atributos em relação ao treino | Medida por PSI e KS contra a janela de referência do treino |
| **Gate** | Verificação que interrompe o fluxo quando reprova | Qualidade de dados, cobertura de 80 %, p95 de latência, fidelidade da matriz |
| **KS** (Kolmogorov-Smirnov) | Teste da maior diferença entre duas distribuições acumuladas | Aplicado às variáveis numéricas; complementa o PSI |
| **Model card** | Documento que descreve uso pretendido, desempenho, limitações e riscos do modelo | `docs/model_card.md` |
| **Model Registry** | Catálogo de versões de modelo com estágios e aliases | MLflow local, SQLite |
| **MLOps** | Práticas que levam um modelo de experimento a serviço operado, com rastreabilidade, teste e monitoramento | Objeto do TCC: o pipeline de seis etapas |
| **Observabilidade** | Capacidade de entender o estado interno do sistema pelo que ele emite | Log JSON com correlation id, métricas RED, gauges de drift, painel provisionado |
| **PSI** (Population Stability Index) | Soma ponderada das diferenças entre proporções de duas distribuições em faixas | Implementado no projeto, com a ausência como categoria própria (ADR 0006) |
| **RED** | Rate, Errors, Duration: as três métricas básicas de um serviço | Taxa, erros e latência expostos em `/metrics` |
| **Rastreabilidade** | Poder reconstruir um resultado a partir do dado, do código e da configuração | Cinco parâmetros em cada run amarram os três (`docs/politica_versionamento.md`) |
| **Registro de decisão** (ADR) | Documento curto com contexto, decisão, alternativas e consequências | `docs/adr/`, um arquivo por decisão |
| **Reprodutibilidade** | Mesma entrada e mesmo ambiente produzindo o mesmo resultado | Semente fixa, dependências travadas, duas execuções com hashes idênticos |
| **Retreino** | Treinar de novo com dados mais recentes | Recomendado por flag a partir do drift; a flag recomenda, o runbook executa |
| **Runbook** | Procedimento operacional para incidentes | `docs/runbook.md` |
| **SLI, SLO** | Indicador e objetivo de nível de serviço | `docs/slo.md` |
| **Smoke test** | Verificação mínima de que algo sobe e responde | Imagens no CI e serviços do compose |
| **Teste de contrato** | Verifica que a interface HTTP respeita o esquema publicado | `tests/contract/`, com TestClient e geração a partir do OpenAPI |

## Sobre os dados deste estudo

| Termo | Definição |
|---|---|
| **PNS 2013** | Pesquisa Nacional de Saúde do IBGE; a fonte é o módulo de exames laboratoriais |
| **eGFR** | Taxa de filtração glomerular estimada, indicador de função renal |
| **IMC** | Índice de massa corporal, peso dividido pelo quadrado da altura |
| **Data freeze** | Data em que a base foi congelada para o estudo, registrada no dicionário de dados |
| **Microdados** | Registros individuais anonimizados, em oposição a tabelas agregadas |

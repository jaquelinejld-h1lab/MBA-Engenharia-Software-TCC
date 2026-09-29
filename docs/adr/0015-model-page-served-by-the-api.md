# ADR 0015: página de modelo na interface, servida pela API

Status: aceita. Data: 2026-09-23.

## Contexto

A especificação da interface no escopo do projeto prevê quatro capacidades: predição
individual, predição em lote, painel de monitoramento e explicabilidade local. Faltava
uma leitura global do modelo: qual o desempenho do campeão, como ele se compara aos
demais runs, o que ele usa para decidir e como se comporta por subgrupo.

`[ESCOPO]` Esta é uma ampliação consciente, não parte das seis etapas. A justificativa
para aceitá-la: o model card, o OE5 e a seção de equidade já exigem exatamente esses
números, que hoje só existem em arquivos de `evidencias/` e em prints do MLflow. Expondo
os mesmos números na interface, eles passam a ser verificáveis por quem usa o sistema e
reaproveitáveis como figura do TCC, sem produzir evidência nova.

Restrição inegociável: a interface é cliente HTTP puro. Ler `artifacts/` ou o banco do
MLflow a partir do Streamlit resolveria o problema em poucas linhas e quebraria a
separação que o projeto se propôs a demonstrar.

## Decisão

1. O treino passa a persistir, no diretório de cada run linear,
   `interpretation.json` (coeficientes por dummy, importância agregada por variável e
   uma amostra de contribuições do holdout) e, para o run promovido,
   `subgroup_metrics.json`. Antes, a importância global existia apenas em
   `evidencias/global_importance.json`, gerado fora do pipeline, e as métricas por
   subgrupo apenas em CSV.
2. A API ganha cinco rotas somente leitura sob `/model`: `summary`, `roc`,
   `importance`, `contributions` e `subgroups`, no módulo `src/api/model_routes.py`.
   Elas não executam inferência e leem apenas os artefatos do run em serviço.
3. Toda resposta carrega `available` e, quando falso, `reason`. Um artefato ausente,
   porque o modelo foi treinado por uma versão anterior, é mensagem na página e não
   erro HTTP: uma página de diagnóstico que cai quando falta um arquivo é pior do que
   inútil.
4. A interface ganha a página Modelo, com três abas (Desempenho, Interpretação,
   Equidade). Os gráficos são especificações Vega-Lite renderizadas por
   `st.vega_lite_chart`, que faz parte do Streamlit, construídas por funções puras em
   `src/app/model_view.py` e cobertas por teste unitário.

## Alternativas consideradas

| Alternativa | Motivo do descarte |
|---|---|
| Streamlit lendo `artifacts/` diretamente | Quebra a separação entre interface e modelo, que é o ponto do OE4 e aparece no diagrama de arquitetura |
| Incorporar prints do MLflow por iframe | Acopla a interface ao servidor de tracking, que é ferramenta de desenvolvimento e não componente de produção |
| Adicionar `altair`, `plotly` ou `matplotlib` à imagem da interface | Dependência nova na imagem cujo propósito é ser mínima. `st.vega_lite_chart` dá o mesmo controle com especificação em dicionário, que ainda por cima é testável sem subir o Streamlit |
| Beeswarm com SHAP | O ADR 0003 já havia escolhido contribuições lineares, exatas para regressão logística, e o ADR 0012 removeu a biblioteca `shap` do ambiente. O gráfico tem a forma de um beeswarm de SHAP e o rótulo diz explicitamente que não é SHAP |
| Calcular a interpretação na API, sob demanda | Exigiria o conjunto de treino dentro do contêiner da API, que hoje carrega apenas o pipeline. Calcular uma vez no treino e persistir mantém a API sem dados |

## Consequências

- Modelos treinados antes desta versão não têm `interpretation.json` nem
  `subgroup_metrics.json`. A página mostra o motivo e pede um novo treino; nada quebra.
- Estimadores sem `coef_` (random forest, MLP, SVM) não geram interpretação. Se um deles
  for promovido a campeão, as abas Desempenho e Equidade continuam completas e a aba
  Interpretação informa a ausência. Uma leitura global para modelos não lineares exigiria
  importância por permutação, que fica fora deste escopo.
- A amostra de contribuições é limitada por `model.interpretation.max_contribution_rows`
  e arredondada em quatro casas: ela alimenta um gráfico, não uma auditoria. A
  contribuição exata de uma predição individual continua vindo de `/predict?explain=true`.
- `evidencias/global_importance.json` passa a ser derivável do artefato do run, o que
  fecha mais uma lacuna de reprodutibilidade do mesmo tipo tratado no ADR 0014.

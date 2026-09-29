# Fase 0: Inventário dos artefatos originais

| Item | Valor |
|---|---|
| Artefatos analisados | `funcoes_mecai_v3.py` (1.304 linhas), `experimento_mecai24_v4.ipynb` (95 células, kernel Python 3.11.7), `EXAMES-PNS-2013-FINAL_05052023.xlsx` |
| Data da análise | 2026-09-14 |
| Status | Aguardando aprovação antes de qualquer código em `src/` |
| Método | Leitura integral do `.py`, extração de código e saídas do `.ipynb`, inspeção do `.xlsx`, e reexecução de verificação do caminho do modelo campeão (seção 5.3) |

Legenda das flags: `[REPRODUCIBILIDADE]`, `[DADOS]`, `[VERSAO]`, `[ESCOPO]`, `[INCERTEZA]`, `[PERGUNTA]` (decisão que precisa da sua resposta antes de seguir).

---

## 1. Inventário de funções de `funcoes_mecai_v3.py`

Coluna "Caminho": **INF** entra na inferência (vira parte da API), **TRN** só no treino, **EDA** só na análise exploratória, **N/U** não usada pelo notebook.

### 1.1 Carga, filtro populacional e split

| Função | Assinatura real | O que faz | Dependências | Efeitos colaterais | Caminho |
|---|---|---|---|---|---|
| `carregar_dividir_dados` | `(path, teste) -> (df6, df_train, df_test)` | Lê o xlsx, sobe as colunas para maiúsculas, seleciona 105 colunas, aplica 4 filtros populacionais, constrói `target`, renomeia colunas, troca `'NaN'`/`'#NULL!'` por `np.nan`, remove linhas sem `regiao` e sem `feijao_dias`, faz `train_test_split(test_size=teste, stratify=y, random_state=42)` | pandas, sklearn | `print` extenso; `SettingWithCopy` nas atribuições em `df4`; `reset_index` | TRN (filtros e alvo). A lista de 105 colunas e o mapa de renomeação são reaproveitados na INF |

### 1.2 Engenharia de atributos (funções puras por linha)

| Função | Assinatura real | O que faz | Observações | Caminho |
|---|---|---|---|---|
| `consumo_saudavel` | `(frequencia) -> str` | 7 dias → '1'; 5 a 6 → '2'; 3 a 4 → '3'; senão '4' | Usada em feijão, salada, verduras, frutas | INF |
| `consumo_carne_frango` | `(frequencia, tipo) -> str \| None` | Cruza frequência e tipo (com/sem gordura) | Retorna `None` quando `tipo` é NaN e frequência 3 a 7. `carne_consumo` e `frango_consumo` **não** entram no modelo final | N/U (criadas, mas fora de `variaveis`) |
| `consumo_peixe` | `(row) -> str \| None` | >= 3 → '1'; 2 → '2'; 1 → '3'; 0 → '4' | Retorna `None` para NaN (não ocorre no dado) | INF |
| `consumo_suco_frutas` | `(frequencia) -> str` | 0 a 1 → '1'; 2 a 3 → '2'; 4 a 5 → '3'; 6 a 7 → '4'; senão `'Frequência Inválida'` | Lógica inversa à de `consumo_saudavel` (mais suco = pior) | INF |
| `consumo_doce` | `(frequencia) -> str` | 0 → '1'; 1 → '2'; 2 a 3 → '3'; >= 4 → '4' | | INF |
| `classificar_substituir_ref` | `(frequencia) -> str` | 0 → '1'; 1 → '2'; 2 → '3'; >= 3 → '4' | | INF |
| `calcular_tempo_total_atividade_fisica` | `(row) -> float` | Soma minutos semanais de 5 blocos: exercício (30 min/dia assumido), deslocamento a pé/bike, trabalho com esforço, ir ao trabalho a pé/bike, atividade doméstica | Depende de 17 colunas brutas; guarda de NaN só implícita (condições `== 1` falham em NaN) | INF |
| `classificar_nivel_atividade_fisica` | `(row) -> str` | 0 min → '1'; < 150 → '2'; depois cruza tipo de exercício (conjuntos moderado/vigoroso) com minutos → '3' ou '4'; fallback '2' | Tipos 1,2,7,8,11,16 moderados; 3,4,5,6,9,10,12,13,14,15 vigorosos; tipo 17 cai no fallback | INF |
| `impute_knn` | `(df, numeric_var, categorical_vars) -> np.ndarray` | One-hot (`drop='first'`) das categóricas + `KNNImputer(n_neighbors=5)`; devolve a coluna numérica imputada | **Estado ajustado no próprio DataFrame de entrada** (fit_transform), sem persistência do encoder nem do imputer | INF (ver [DADOS] D3) |
| `classificar_imc` | `(imc) -> str` | >= 40 → '4'; >= 30 → '3'; >= 25 → '2'; >= 18,5 → '1'; senão '5' | Ordem dos códigos não é monotônica ('5' = baixo peso) | INF |
| `add_variaveis_ajustadas` | `(df) -> df` | Orquestra todas as derivadas: faixa etária, moradores, estado civil, etnia negra, renda, diagnósticos recodificados, consumo alimentar, atividade física, tabagismo, IMC, cintura, percepção de saúde, angina, e três imputações KNN + categorização de eGFR, colesterol e glicose | **Muta o DataFrame recebido in place** e o devolve; 30+ `print`; contém a constante `salario_minimo = 678.00` | INF (é o núcleo do transformador) |

### 1.3 Seleção de variáveis

| Função | Assinatura real | O que faz | Caminho |
|---|---|---|---|
| `cramers_v`, `cramers_corrected_stat` | `(x, y) -> float`, `(confusion_matrix) -> float` | V de Cramér com correção de viés | TRN (seleção) |
| `calc_iv` | `(df, feature, target, eps=1e-10) -> float` | Information Value por categoria | TRN |
| `classify_cramers_v`, `classify_iv` | `(value) -> str` | Rótulos textuais dos cortes | TRN |
| `summarize_categorical_features` | `(df, variaveis, target) -> DataFrame` | Nulos, V de Cramér e IV de cada variável contra o alvo; converte float para `Int64` | TRN |
| `get_significant_variables` | `(summary_df) -> list` | Remove quem é "muito fraca" **e** "não útil" simultaneamente | TRN. Sua saída (17 variáveis) **não é usada** para definir o modelo final |
| `calculate_cramers_v` | `(df, variaveis) -> (list, list)` | Pares de variáveis com V > 0,5 | TRN. Executada sobre `df_fe` (desenvolvimento completo) |
| `variable_selection_cv` | `(df, target) -> DataFrame` | SMOTE + Boruta com `XGBClassifier` como estimador base; faz monkeypatch de `np.int`, `np.float`, `np.bool` | TRN (caminho Boruta, descartado no resultado final) |

### 1.4 Balanceamento e codificação

| Função | Assinatura real | O que faz | Efeitos colaterais | Caminho |
|---|---|---|---|---|
| `balance_data_with_smote` | `(df, target) -> DataFrame` | One-hot + SMOTE + arredondamento | Encoder ajustado no próprio df | N/U no notebook (SMOTE só dentro de `variable_selection_cv`) |
| `encode_data` | `(df, target) -> DataFrame` | `astype('category')` + `OneHotEncoder(drop='first', sparse=False)` + `astype(int)` + concat com `y` | **Encoder ajustado separadamente em cada DataFrame** (treino, teste e desenvolvimento); nomes de coluna dependem do dtype de origem (`regiao_2.0` vs `regiao_2`) | INF (a codificação é parte do transformador; o encoder deve ser persistido) |
| `prepare_test_dataframe` | `(df, target, variaveis) -> DataFrame` | Variante de `encode_data` com renomeação manual de colunas para casar com o treino | Evidência do problema de nomes acima | N/U no notebook |

### 1.5 Treino e avaliação

| Função | Assinatura real | O que faz | Caminho |
|---|---|---|---|
| `evaluate_models` | `(train_df, test_df, target) -> DataFrame` | Treina 9 algoritmos com hiperparâmetros fixos, avalia no holdout, plota ROC; métricas arredondadas a 2 casas | TRN |
| `evaluate_models_cv` | `(df, target, cv=5) -> DataFrame` | Mesmos 9 algoritmos em `StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)`; média e desvio por fold; **a Tabela 4 da dissertação vem daqui com `cv=10`** | TRN (gera as métricas de referência) |
| `train_catboost_model` | `(X_train, y_train) -> CatBoostClassifier` | CatBoost com hiperparâmetros fixos (não os do random search) | TRN (interpretabilidade) |
| `plot_shap_values_catboost` | `(catboost_model, X_test) -> None` | `shap.TreeExplainer` + summary plot | EDA |
| `train_logistic_model` | `(X_train, y_train) -> (model, scaler)` | `StandardScaler` + `LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced', random_state=42)` | TRN (interpretabilidade). **Atenção**: o campeão avaliado em `evaluate_models_cv` **não** usa scaler; esta função só serve aos coeficientes da Tabela 5 |
| `interpret_logistic_model` | `(logistic_model, X_train) -> DataFrame` | Top 20 coeficientes e exp(coef) | EDA |
| `verificar_colunas` | `(df1, df2) -> None` | Compara conjuntos de colunas | TRN (verificação) |
| `analisar_variaveis_numericas` | `(dados, variaveis, target) -> DataFrame` | Descritivas por variável numérica | N/U |

Importações do módulo sem uso em nenhuma função: `MinMaxScaler`, `SMOTENC`, `NearMiss`, `RFECV`, `SelectKBest`, `f_classif`, `GridSearchCV`, `cross_val_score`, `ParameterSampler`, `optuna`, `TPESampler`, `uniform`, `locale`, `math`, `time`, `imblearn.pipeline.Pipeline`. O módulo executa `warnings.filterwarnings('ignore')` e `sns.set` na importação.

---

## 2. Fluxo do notebook

Sequência exata das células com efeito, da leitura à métrica final. Células vazias (53, 62, 67, 71 a 73, 81, 91 a 94) omitidas.

| # | Célula | Chamada | Entrada | Saída e números registrados |
|---|---|---|---|---|
| 1 | 2 | `from funcoes_mecai_v3 import *` | | autoreload ativo |
| 2 | 4 | `carregar_dividir_dados('dados/EXAMES-PNS-2013-FINAL_05052023.xlsx', 0.20)` | xlsx (8.952 × 509) | `df` (6.203 × 103), `df_train` (4.962 × 103), `df_test` (1.241 × 103). Alvo: 17,04 % positivos no desenvolvimento, 17,05 % no treino, 17,00 % no teste |
| 3 | 6 | `add_variaveis_ajustadas(df_train)` | df_train | `df_train_fe` (imputação KNN ajustada no treino) |
| 4 | 7 | `add_variaveis_ajustadas(df_test)` | df_test | `df_test_fe` (imputação KNN ajustada **no teste**) |
| 5 | 8 | `add_variaveis_ajustadas(df)` | df | `df_fe` (imputação KNN ajustada no desenvolvimento completo) |
| 6 | 12, 13 | Definição de `variaveis_inicial` (38) e `variaveis` (33) | | Lista final hardcoded, não derivada de código |
| 7 | 15 a 18 | `summarize_categorical_features(df_train_fe, variaveis_inicial, 'target')`, `get_significant_variables` | df_train_fe | 17 variáveis "significativas". **Resultado não usado adiante** |
| 8 | 20 a 25 | `calculate_cramers_v(df_fe, ...)` duas vezes | df_fe | Pares com V > 0,5: conj_comp × estado_civil_ag (0,579); trabalho × faixa_renda_sl (0,646); diag_diab_rec × diag_colest_rec (0,528); cintura_risco_aumentado × cintura_risco_muito_aumentado (0,623); ambos × class_imc (0,575 e 0,606). Justifica os 5 cortes de 38 para 33 |
| 9 | 27 a 32 | Projeção `[variaveis + ['target']]` | | `df_train_v1` (4.962 × 34), `df_test_v1` (1.241 × 34), `df_v1` (6.203 × 34) |
| 10 | 34 a 40 | `encode_data(...)` em cada um; `verificar_colunas` | | 86 colunas (85 dummies + target) em todos; colunas iguais |
| 11 | 43 a 52 | `variable_selection_cv(df_train_v2, 'target')` (Boruta) | df_train_v2 | 59 dummies selecionadas → `*_v3` (60 colunas) |
| 12 | 57, 58 | `evaluate_models(df_train_v3, df_test_v3)` | Boruta, holdout | LR AUC 0,69, recall 0,64 |
| 13 | 60, 61 | `evaluate_models_cv(df_v3, cv=10)` | Boruta, CV | LR AUC 0,68 ± 0,02, recall 0,61 ± 0,05 (Tabela 3 da dissertação) |
| 14 | 65, 66 | `evaluate_models(df_train_v2, df_test_v2)` | Filtro, holdout | LR AUC 0,75, acc 0,70, prec 0,32, recall 0,68, F1 0,44 |
| 15 | 69, 70 | `evaluate_models_cv(df_v2, cv=10)` | Filtro, CV | **LR AUC 0,75 ± 0,02, acc 0,69 ± 0,02, prec 0,31 ± 0,02, recall 0,68 ± 0,06, F1 0,43 ± 0,03 (Tabela 4, resultado final adotado)** |
| 16 | 75, 78 | `RandomizedSearchCV(CatBoostClassifier(...), n_iter=50, cv=10, scoring=make_scorer(roc_auc_score), random_state=42)` | train_v2 / test_v2 | Melhor: `depth=3, iterations=1000, learning_rate=0.01, l2_leaf_reg=9, scale_pos_weight=4.882`. AUC holdout 0,7526 |
| 17 | 80 | `print(best_params_rl, auc_rl)` | **variáveis nunca definidas no notebook salvo** | `solver='saga', penalty='none', class_weight='balanced', C=0.1`; AUC 0,7452 |
| 18 | 84 a 87 | `train_catboost_model` + `plot_shap_values_catboost` | train_v2 / test_v2 | Gráfico SHAP (sem números) |
| 19 | 89, 90 | `train_logistic_model` + `interpret_logistic_model` | train_v2 | Top 20 coeficientes (Tabela 5): sexo_2 = -0,3876; faixa_etaria_60-69 = +0,5991 |

Observações sobre o fluxo:

1. O modelo campeão é definido pela linha 15 (CV k=10 sobre o desenvolvimento completo). O holdout da linha 14 é secundário.
2. A célula 80 prova que houve um random search de Regressão Logística cujo código foi apagado ou nunca salvo. O resultado dele (AUC 0,7452) é **inferior** ao da LR com hiperparâmetros padrão (0,7467 no mesmo holdout, ver 5.3), e não há evidência de que tenha sido adotado. `[INCERTEZA]` I1.
3. O scorer `make_scorer(roc_auc_score)` sem `needs_threshold`/`response_method` calcula AUC sobre rótulos preditos (0/1), não sobre probabilidades. Só afeta o random search do CatBoost, que não é o campeão.
4. A CV de referência (linha 15) roda sobre `df_v2`, cuja imputação KNN foi ajustada nas 6.203 linhas antes da divisão em folds. Ver `[DADOS]` D2.

---

## 3. Dicionário de variáveis

### 3.1 Arquivo bruto

| Item | Valor |
|---|---|
| Arquivo | `EXAMES-PNS-2013-FINAL_05052023.xlsx`, aba única `EXAMES-PNS-2013-FINAL_com-ponde` |
| Tamanho | 11.928.227 bytes (11,4 MB). Abaixo de 50 MB: commit direto, sem Git LFS |
| SHA-256 | `80222fa535524829d531504109ac4b49d44e4e6d0c04c3c281ff972bd332d1f7` |
| Dimensão | 8.952 linhas × 509 colunas; todas numéricas após leitura (nenhuma string; os literais `'NaN'`/`'#NULL!'` tratados no código não ocorrem neste arquivo) |
| Tempo de leitura | ~20 s com `pd.read_excel` (relevante para o pipeline: converter para Parquet no passo de ingestão) |
| Colunas que o notebook usa | 105 de 509 |
| Identificadores | Nenhuma coluna de UPA, domicílio, ordem do morador ou município. Há `REGIAO` (5 níveis) e `C008` (não usada). Combinação de região × idade × sexo × etnia não é rara o suficiente para reidentificar; risco documentado como baixo na checklist de privacidade |
| Peso amostral | `PESO_LAB` (peso do módulo de exames laboratoriais): presente, **nunca usado** pelo notebook. 6 registros com valor 99,0 (provável código de ausência); todos caem fora do filtro `Z051 == 1` |
| Licença e origem | PNS 2013, IBGE, microdados públicos. `[INCERTEZA]` I5: a data de coleta e a licença precisam ser transcritas da documentação do IBGE, não inferidas |

### 3.2 Filtros populacionais e alvo (reproduzidos no dado)

| Ordem | Regra | Coluna PNS | Removidos | Restantes |
|---|---|---|---|---|
| 0 | Conjunto original | | | 8.952 |
| 1 | `Z051 == 1` (consentiu armazenar exames; 2 = 119, 3 = 897) | Z051 | 1.016 | 7.936 |
| 2 | `P005 not in {1, 3}` (grávida ou não sabe) | P005 | 101 | 7.835 |
| 3 | `Q002 != 2` (hipertensão só na gravidez) | Q002 | 118 | 7.717 |
| 4 | `Q006 != 1` (tomou remédio para hipertensão nas 2 semanas) | Q006 | **1.501** | 6.216 |
| 5 | `regiao` nula | REGIAO | 6 | 6.210 |
| 6 | `feijao_dias` nulo (proxy de questionário vazio) | P006 | 7 | 6.203 |

Alvo: `target = 1` se `W00407 >= 140` ou `W00408 >= 90` (PA sistólica e diastólica finais, mmHg). Prevalência no desenvolvimento: 1.057 / 6.203 = 17,04 %. Threshold de decisão: 0,5 (padrão do `predict`, nunca alterado).

`[DADOS]` D1: 8 registros do bruto têm PA ausente; o código gera `target = 0` para eles em vez de excluí-los. Nenhum sobrevive aos filtros 4 a 6, então o resultado não muda, mas a regra precisa ser explícita no pipeline (excluir PA ausente antes de construir o alvo, com contagem registrada).

Divergência com o extrato da dissertação: o extrato registra 1.687 removidos no filtro 4; o dado e a aritmética do notebook dão 1.501. Ver seção 7.

### 3.3 Variáveis brutas do caminho de inferência

São as 61 colunas da PNS que o transformador precisa receber para produzir as 33 variáveis do modelo. Tipo: Q = questionário, A = antropometria/medida, L = exame laboratorial. Ausentes e domínio medidos no conjunto de desenvolvimento (6.203). Tratamento de outliers: **nenhum** em nenhuma variável.

| PNS | Nome no .py | Tipo | Domínio observado | Ausentes | Tratamento de ausentes no código | Derivada gerada |
|---|---|---|---|---|---|---|
| Z002 | idade | Q | 18 a 104 | 0 | nenhum | faixa_etaria (7 faixas, corte em 30/40/50/60/70/80) |
| Z001 | sexo | Q | 1 homem, 2 mulher | 0 | nenhum | sexo; cintura_risco (só para imputação) |
| REGIAO | regiao | Q | 1 N, 2 NE, 3 SE, 4 S, 5 CO | 0 (após filtro) | linha removida | regiao |
| Z003 | etnia | Q | 1, 2, 3, 4, 5, 9 | 0 | nenhum | etnia_negra ('1' se 2 preta ou 4 parda; '2' caso contrário, incl. 9 ignorado); etnia bruta entra na imputação KNN |
| C001 | num_moradores | Q | 1 a 13 | 0 | nenhum | faixa_num_mor ('1' a '5', '6+') |
| C010 | conj_comp | Q | 1 sim, 2 não | 0 | nenhum | conj_comp |
| E01602, E01604, E01802, E01804, F00102, F00702, F00802 | vl_* (7 rendimentos) | Q | 5 a 40.000 | 38 % a 100 % | `sum(skipna=True)`: ausente vale 0; 1.490 registros (24 %) sem nenhum rendimento viram renda 0 → faixa '1' | faixa_renda_sl (9 faixas em SM de R$ 678,00) |
| I001 | plano_saude | Q | 1, 2 | 0 | nenhum | plano_saude |
| Q124 | diag_renalc | Q | 1, 2 | 0 | nenhum | diag_renalc |
| Q030 | diag_diab | Q | 1, 2, 3 | 922 (14,9 %) | 3 → 2; NaN → 3 | diag_diab_rec |
| Q063 | diag_dcore | Q | 1, 2 | 0 | NaN → 2 | diag_dcore |
| Q068 | diag_avc | Q | 1, 2 | 0 | NaN → 2 | diag_avc |
| P006, P007, P009, P018 | feijao/salada/verd_legu/frutas_dias | Q | 0 a 7 | 0 | nenhum | *_consumo via `consumo_saudavel` |
| P015 | peixe_dias | Q | 0 a 7 | 0 | nenhum | peixe_consumo |
| P016 | suco_dias | Q | 0 a 7 | 0 | nenhum | suco_consumo |
| P021 | tipo_refri | Q | 1, 2, 3 | 1.766 (28,5 %) | NaN → 4 ("não consome") | tipo_refri_rec |
| P024 | tipo_leite | Q | 1, 2, 3 | 1.703 (27,5 %) | NaN → 4 | tipo_leite_rec |
| P025 | doces_dias | Q | 0 a 7 | 0 | nenhum | doces_consumo |
| P026 | subst_ref_dias | Q | 0 a 7 | 0 | nenhum | freq_subst_ref |
| P02601 | consumo_sal | Q | 1 a 5 | 0 | nenhum | consumo_sal_rec (1,2 → '3'; 3 → '2'; 4,5 → '1') |
| P027 | alcool | Q | 1, 2, 3 | 0 | nenhum | alcool (sem recodificação) |
| P034, P035, P036 | exercicio, exercicio_dias, tipo_exercicio | Q | 1/2; 0 a 7; 1 a 17 | 0; 70,5 %; 71,6 % | ausentes só quando `exercicio == 2`; 72 casos com `exercicio == 1` e tipo ausente caem no fallback '2' | nivel_atv_fisica |
| P042, P04301, P04302 | atv_pe_bk, _horas, _minutos | Q | 0 a 7; 0 a 8; 0 a 56 | 0; 62,5 %; 62,5 % | ausentes só quando dias = 0 | nivel_atv_fisica |
| P039, P03901, P03902, P03903 | trab_esforco, dias, horas, minutos | Q | 1/2; 1 a 7; 0 a 16; 0 a 59 | 37,4 %; 80,5 % ×3 | NaN em `trab_esforco` faz `== 1` falhar (conta 0 min) | nivel_atv_fisica |
| P040, P04101, P04102 | trab_pe_bk, horas, minutos | Q | 1, 2, 3; 0 a 7; 0 a 59 | 37,4 %; 72,8 % ×2 | idem | nivel_atv_fisica |
| P044, P04401, P04403, P04404 | atv_dom_esf, dias, horas, minutos | Q | 1/2; 1 a 7; 0 a 8; 0 a 58 | 0; 79,9 % ×3 | idem | nivel_atv_fisica |
| P050 | fuma_atual | Q | 1 diário, 2 menos que diário, 3 não | 0 | `'.'` → '2' (literal nunca ocorre) | fumante (intermediária) |
| P051 | fumo_rotina_hist | Q | 1, 2 | 6.045 (97,5 %) | NaN trata como "não" | fumante_hist |
| P052 | fumo_hist | Q | 1, 2, 3 | 941 (15,2 %) | NaN trata como "não" | fumante_hist |
| Q132 | med_dormir | Q | 1, 2 | 0 | nenhum | med_dormir |
| Z004, Z005 | peso (kg), altura (cm) | A | 30 a 179; 131 a 199,3 | 0 | nenhum | class_imc (IMC 13,1 a 61,3) |
| W00303 | circ_cintura (cm) | A | 50 a 141,5 | 0 | nenhum | cintura_risco_aumentado (só na imputação KNN) |
| N001 | pcp_saude | Q | 1 a 5 | 0 | nenhum | pcp_saude_rec (1,2,3 → '2' boa; 4,5 → '1' ruim) |
| N004, N005 | angina_rapd, angina_norm | Q | 1, 2, 3; 1, 2 | 60; 103 | `np.where` com NaN → 2 | angina ('1' se ambos = 1) |
| Z026 | egfr_afro (mL/min/1,73 m²) | L | 4,4 a 557,8 | 741 (11,9 %) | KNN k=5 | cat_egfr_afro (5 faixas: < 30, 30 a 45, 45 a 60, 60 a 90, >= 90) |
| Z031 | colesterol (mg/dL) | L | 68 a 433 | 243 (3,9 %) | KNN k=5 | colesterol_ideal ('1' < 190) |
| Z035 | glicose (mg/dL) | L | 56 a 411 | 267 (4,3 %) | KNN k=5 | cat_glicose (< 70 '4'; 70 a 100 '1'; 100 a 126 '2'; >= 126 '3') |
| Q060 | diag_colest | Q | 1, 2 | 1.166 (18,8 %) | NaN → 3 | diag_colest_rec (só na imputação KNN, cortada do modelo) |

Variáveis carregadas mas sem papel na inferência (só filtro, alvo, EDA ou derivadas descartadas): Z051, P005, Q002, Q006, W00407, W00408 (filtro e alvo); Q001, Q003, Q004, Q005, Q018xx, Q028, C011, E001, J002, J037, N010, N011, Z025, Z027, Z032, Z033, Z036, Z044, Z045, Z049, P002, P011, P012, P013, P014, P020, P023, P038, P045, P046, P053, P068.

### 3.4 Variáveis do modelo (33 categóricas → 85 dummies com `drop='first'`)

Ordem e nomes de coluna exatamente como o encoder gerou (nomes com `.0` vêm de colunas float):

`faixa_etaria` (7 níveis, ref 18-29), `sexo` (ref 1), `regiao` (ref 1.0), `etnia_negra` (ref 1), `faixa_num_mor` (ref 1), `conj_comp` (ref 1), `faixa_renda_sl` (9, ref 1), `plano_saude` (ref 1), `diag_renalc` (ref 1.0), `diag_diab_rec` (3, ref 1.0), `diag_dcore` (ref 1.0), `diag_avc` (ref 1.0), `feijao_consumo`, `salada_consumo`, `verd_legu_consumo`, `peixe_consumo`, `suco_consumo`, `frutas_consumo` (4 cada, ref 1), `tipo_refri_rec`, `tipo_leite_rec` (4, ref 1.0), `doces_consumo`, `freq_subst_ref` (4, ref 1), `consumo_sal_rec` (3, ref 1), `alcool` (3, ref 1.0), `nivel_atv_fisica` (4, ref 1), `fumante_hist` (ref 1), `med_dormir` (ref 1.0), `class_imc` (5, ref 1), `pcp_saude_rec` (ref 1), `angina` (ref 1), `cat_egfr_afro` (5, ref 1), `cat_glicose` (4, ref 1), `colesterol_ideal` (ref 1).

Nenhuma variável do modelo tem valor ausente após o transformador. O mapa PNS → português → inglês vai para `configs/variable_mapping.yaml` no passo 2.

---

## 4. Modelo campeão

| Item | Valor no notebook |
|---|---|
| Algoritmo | `sklearn.linear_model.LogisticRegression` |
| Hiperparâmetros | `C=1.0, solver='liblinear', class_weight='balanced', random_state=42`; demais no padrão (`penalty='l2'`, `max_iter=100`, `tol=1e-4`, `fit_intercept=True`) |
| Escalonamento | **Nenhum** na avaliação (entrada é 85 dummies inteiras 0/1). `StandardScaler` só em `train_logistic_model`, usado exclusivamente para os coeficientes da Tabela 5 |
| Conjunto de variáveis | Filtro (33 categóricas, 85 dummies). Boruta descartado |
| Tratamento de desbalanceamento | `class_weight='balanced'`. SMOTE descartado (AUC 0,69 contra 0,75, segundo a dissertação) |
| Estratégia de validação | `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` sobre o **desenvolvimento completo** (6.203), não sobre o treino. Holdout 80/20 estratificado (`random_state=42`) como avaliação complementar |
| k | 10 |
| Threshold | 0,5 (implícito no `predict`) |
| Seed | 42 em todos os pontos: split, folds, LR, SMOTE, Boruta, random search |
| Segundo e terceiro colocados | CatBoost (AUC 0,73, recall 0,59) e SVM (AUC 0,72, recall 0,60), com hiperparâmetros fixos de `evaluate_models_cv` |
| Baseline exigido pelo projeto | Coincide com o campeão (LR). O baseline do OE2 será a própria LR e os concorrentes do Projeto de Pesquisa (RF, XGBoost, LightGBM, MLP) mais CatBoost e SVM |

---

## 5. Métricas de referência

### 5.1 Como reportadas no notebook (arredondadas a 2 casas)

| Avaliação | AUC | Acurácia | Precisão | Recall (sensibilidade) | F1 |
|---|---|---|---|---|---|
| CV k=10, desenvolvimento (célula 70, Tabela 4) | 0,75 ± 0,02 | 0,69 ± 0,02 | 0,31 ± 0,02 | 0,68 ± 0,06 | 0,43 ± 0,03 |
| Holdout 20 % (célula 66) | 0,75 | 0,70 | 0,32 | 0,68 | 0,44 |

Especificidade e matriz de confusão **não são reportadas** no notebook nem no extrato da dissertação.

### 5.2 Problema de precisão dos golden values

`[REPRODUCIBILIDADE]` R1: a tolerância exigida (`|ΔAUC| <= 0,005`, `|Δsens| <= 0,01`) é mais fina do que a precisão das métricas registradas (0,01). Um golden value de 0,75 não permite decidir se 0,7466 está dentro ou fora da tolerância. Por isso o caminho do campeão foi reexecutado para obter os valores sem arredondamento (5.3).

### 5.3 Reexecução de verificação (valores sem arredondamento)

Reexecutei exatamente `carregar_dividir_dados` → `add_variaveis_ajustadas` (nos três conjuntos) → projeção nas 33 variáveis → `encode_data` → LR nos mesmos folds e no mesmo holdout, importando as funções do `.py` original com um único ajuste de compatibilidade (`OneHotEncoder(sparse=False)` → `sparse_output=False`).

| Métrica | CV k=10 (média ± dp) | Holdout |
|---|---|---|
| AUC-ROC | **0,748804 ± 0,021710** | **0,746689** |
| Sensibilidade (recall) | **0,681339 ± 0,059980** | **0,677725** |
| Especificidade | 0,688688 ± 0,016796 | 0,704854 |
| Acurácia | 0,687415 | 0,700242 |
| Precisão | 0,310076 | 0,319911 |
| F1 | 0,426033 | 0,434650 |
| Matriz de confusão (VN, FP / FN, VP) | agregada nos folds: 3.544, 1.602 / 337, 720 | 726, 304 / 68, 143 |

Resultado da comparação com o notebook: contagens de filtro, distribuições das derivadas, 86 colunas e **as 5 métricas da CV k=10 (Tabela 4) coincidem em todas as casas reportadas**. No holdout, AUC, acurácia, precisão e recall coincidem; o **F1 diverge na segunda casa** (0,4347 aqui contra 0,44 no notebook), o que significa que pelo menos uma predição do holdout mudou entre os ambientes. A CV, que é a avaliação de referência, não foi afetada. AUC por fold: 0,7512; 0,7271; 0,7079; 0,7629; 0,7872; 0,7340; 0,7627; 0,7454; 0,7408; 0,7689.

`[VERSAO]` V1: o notebook não registra versões de biblioteca (só Python 3.11.7). A reexecução rodou em scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4, scipy 1.17.1, porque o PyPI está bloqueado nesta sessão e não foi possível montar o ambiente da época. A coincidência na CV é evidência forte, não prova, de que os valores acima são os do notebook; a divergência do F1 no holdout mostra que diferenças de ambiente já produzem variação na terceira casa. Proposta: os golden values de `tests/integration/test_metric_regression.py` serão gerados pela **primeira execução do pipeline refatorado no ambiente travado do repositório** e congelados em `evidencias/golden_metrics.json`; os valores da tabela acima entram como verificação externa com tolerância documentada. `[PERGUNTA]` P4 na seção 8.

Arquivo com todos os valores: `evidencias/fase0_reexecucao_metricas.json` (anexo).

---

## 6. Lacunas e riscos

### 6.1 `[REPRODUCIBILIDADE]`

| ID | Achado | Onde | Impacto | Ação no pipeline |
|---|---|---|---|---|
| R1 | Métricas de referência com 2 casas decimais, tolerância com 3 | Notebook | Teste de regressão indecidível | Golden values sem arredondamento (5.3) |
| R2 | Caminho relativo `'dados/...'` hardcoded no notebook; nenhuma raiz de projeto | Célula 4 | Baixo | Caminhos derivados da raiz via config |
| R3 | `warnings.filterwarnings('ignore')` global e `%autoreload` | Módulo, célula 2 | Esconde avisos de convergência e de depreciação | Logging estruturado; warnings visíveis no CI |
| R4 | Código do random search da LR ausente (célula 80 usa `best_params_rl` indefinido) | Célula 80 | Resultado não reproduzível; não é o campeão | Documentar como não reproduzível; não implementar (`[ESCOPO]`) |
| R5 | `salario_minimo = 678.00`, cortes de IMC, glicose, eGFR, colesterol, minutos de atividade, k do KNN, threshold 0,5: todos magic numbers no corpo das funções | `add_variaveis_ajustadas` e auxiliares | Sem rastreabilidade | Tudo para `configs/config.yaml` |
| R6 | Monkeypatch `np.int = np.int32`, `np.float`, `np.bool` para o Boruta funcionar | `variable_selection_cv` | Quebra em numpy >= 1.24 sem o patch; altera estado global | Boruta fica fora do caminho de produção; se reproduzido, isolado com ADR |
| R7 | Seed 42 aplicada objeto a objeto, sem `random.seed`/`np.random.seed` globais; `n_jobs=-1` no random search | Vários | LR liblinear é determinística; risco só nos concorrentes | Seed única em config aplicada a Python, NumPy e cada biblioteca |
| R8 | `evaluate_models*` chama `plt.show()` e devolve métricas arredondadas | `.py` | Efeito colateral de UI no caminho de avaliação | Separar cálculo de métricas de plotagem; nunca arredondar antes de persistir |
| R9 | Leitura do xlsx leva ~20 s e depende de `openpyxl` | Ingestão | Lentidão em testes e CI | Ingestão converte para Parquet com hash; testes usam fixture sintética |

### 6.2 `[DADOS]`

| ID | Achado | Onde | Impacto | Ação no pipeline |
|---|---|---|---|---|
| D1 | PA ausente gera `target = 0` em vez de exclusão | `carregar_dividir_dados` | Nulo no dado atual (0 casos sobrevivem aos filtros), mas a regra está errada | Exclusão explícita com contagem logada; teste unitário |
| D2 | Imputação KNN e one-hot ajustados no próprio conjunto que está sendo transformado: treino, teste e desenvolvimento cada um com seu próprio ajuste. As métricas de referência (CV sobre `df_v2`) usam imputação ajustada em todas as 6.203 linhas, incluindo os folds de teste | `impute_knn`, `encode_data`, células 6 a 8 e 34 a 38 | Vazamento leve (imputação de 3 variáveis laboratoriais usando vizinhos do fold de teste). O extrato da dissertação afirma que "as transformações foram ajustadas no treino e aplicadas ao teste"; o código **não** faz isso | Transformador com `fit` no treino e `transform` no resto, persistido. Ver `[PERGUNTA]` P2 sobre o impacto na fidelidade |
| D3 | `impute_knn` é impossível de reproduzir para uma única observação em inferência (precisa de vizinhos) | `impute_knn` | Bloqueia a API se um exame laboratorial vier ausente | Persistir `KNNImputer` ajustado no treino e aplicá-lo à observação; ou exigir os 3 exames na API. `[PERGUNTA]` P3 |
| D4 | `fumante_hist` compara string com inteiro: `fumante` recebe `'1'`/`'2'` (strings) e a lambda testa `row['fumante'] == 1`, que é sempre falso. Consequência: os 619 fumantes diários do treino (12,5 %) cujo P051 e P052 estão ausentes são classificados como **sem histórico de tabagismo**. Confirmado: a distribuição de `fumante_hist = 1` (14,3 %) é menor que a de `fumante = 1` (15,0 %), o que seria impossível com o OR funcionando | `add_variaveis_ajustadas` | Variável clinicamente errada dentro do modelo campeão. Corrigir muda as métricas | Regra de fidelidade manda reproduzir. `[PERGUNTA]` P1 |
| D5 | Nomes das dummies dependem do dtype de origem (`regiao_2.0` vs `regiao_2`); `prepare_test_dataframe` tem renomeação manual para contornar; a igualdade de colunas entre treino e teste só se sustenta porque todas as categorias aparecem nos dois | `encode_data` | Contrato de esquema frágil | Encoder persistido com `categories` explícitas do `variable_mapping.yaml`; nomes de coluna definidos, não inferidos |
| D6 | Peso amostral `PESO_LAB` ignorado; PNS tem desenho complexo em 3 estágios | Todo o notebook | As métricas são não ponderadas e valem para a amostra, não para a população. Trocar isso é mudança de modelagem | Registrar em ADR como limitação e oportunidade futura, não implementar (`[ESCOPO]`) |
| D7 | Renda: ausente vira zero; 24 % dos registros ficam na faixa '1' por ausência, não por renda baixa | `add_variaveis_ajustadas` | Semântica da faixa de renda contaminada | Reproduzir; documentar no dicionário de dados |
| D8 | Sem tratamento de outliers: eGFR até 557,8, IMC de 13,1 a 61,3, renda até 40.200 | Todo | Nenhum para fidelidade | Faixas plausíveis só na validação de entrada da API/UI, com limites derivados do treino |
| D9 | Sem validação de esquema em nenhum ponto; colunas selecionadas por lista após `str.upper()` | `carregar_dividir_dados` | Falha silenciosa se o IBGE mudar nomes | Contrato Pydantic/Great Expectations como gate no CI |
| D10 | Categorias com fallback silencioso: `consumo_carne_frango` devolve `None`; `consumo_suco_frutas` devolve o literal `'Frequência Inválida'`; tipo de exercício 17 e casos sem tipo caem em `'2'` | Funções de derivação | Não ocorrem no dado de treino; ocorreriam em produção | Validação de domínio na entrada; erro de domínio explícito em vez de fallback |

### 6.3 `[VERSAO]`

| ID | Achado | Ação |
|---|---|---|
| V1 | Nenhuma versão de biblioteca registrada; `OneHotEncoder(sparse=False)` indica scikit-learn < 1.4 (parâmetro removido na 1.4, renomeado para `sparse_output` desde a 1.2). `KNNImputer`, `liblinear` e `StratifiedKFold` são estáveis entre versões, mas não há garantia | `requirements.lock` com pin exato; golden values gerados no ambiente travado |
| V2 | Dependências de treino pesadas e fora do caminho de inferência (catboost, lightgbm, xgboost, boruta, optuna, shap, imbalanced-learn, seaborn) importadas no mesmo módulo das transformações | Separar `requirements-api` de `requirements-train`; a imagem da API não carrega bibliotecas de treino |

### 6.4 `[ESCOPO]` (oportunidades futuras, não implementar; vão para `docs/adr/`)

Correção do bug D4; uso do peso amostral D6; calibração de probabilidade; escolha de threshold por curva de sensibilidade; imputação sem vazamento como modelo alternativo; random search da LR reconstruído.

---

## 7. Divergências entre `.py`, `.ipynb` e extrato da dissertação

| # | Item | `.py` | `.ipynb` | Extrato da dissertação | Versão adotada e motivo |
|---|---|---|---|---|---|
| 1 | Removidos por medicação para hipertensão | `print` comentado (linha 158) | Não imprime; aritmética dá 1.501 | 1.687 | **1.501**, confirmado no dado. O extrato precisa de errata |
| 2 | Variáveis após o filtro por V de Cramér | | `variaveis` tem 33 | "restando 38" e "cinco removidas" | **33** (38 iniciais menos 5). O extrato provavelmente cita as 38 iniciais como finais |
| 3 | Estimador base do Boruta | `XGBClassifier(random_state=42)` | Idem | "Random Forest" | **XGBoost**, conforme o código. Só afeta o caminho descartado |
| 4 | Ausentes nos exames | | | 596 / 201 / 201 (eGFR, colesterol, glicose) | No desenvolvimento: 741 / 243 / 267; no treino: 596 / 201 / **216**. O extrato reporta o treino; a glicose diverge (201 contra 216). Registrar a base e o valor medido |
| 5 | Ajuste das transformações | `fit_transform` no conjunto recebido | Chamado em treino, teste e desenvolvimento separadamente | "ajustadas no treino e aplicadas ao teste" | O **código** é a fonte da verdade para a fidelidade; o pipeline refatorado fará fit no treino (D2). Ver P2 |
| 6 | Hiperparâmetros da LR campeã | `evaluate_models_cv` usa padrão (`C=1.0, liblinear`) | Célula 80 mostra random search (`saga, penalty none, C=0.1`, AUC 0,7452) sem código | Não menciona tuning da LR | **Padrão**, porque a Tabela 4 (0,75) vem de `evaluate_models_cv` e o tuned teve AUC menor no holdout. Ver P4 |
| 7 | Scaler na LR | `train_logistic_model` escala | Métricas vêm de `evaluate_models_cv`, sem scaler | | **Sem scaler** no modelo servido; coeficientes escalados só para a Tabela 5 |
| 8 | `get_significant_variables` (17 variáveis) | Definida | Executada, saída ignorada | Não mencionada como etapa | Não entra no pipeline |
| 9 | Coeficientes da Tabela 5 | | Célula 90: sexo_2 = -0,3876, faixa_etaria_60-69 = +0,5991 | -0,39 e 0,60 | Coincidem |
| 10 | F1 da Tabela 3 | | SVM 0,34, LightGBM 0,34, XGBoost 0,33, CatBoost 0,33, RF 0,01 | 0,06 / 0,06 / 0,06 / 0,06 / 0,01, precisão RF 0,35, recall RF 0,40 | O notebook é a fonte: F1 0,34/0,34/0,33/0,33 e RF recall 0,01. Os 0,06 e o recall 0,40 do extrato são erro de transcrição |
| 11 | Especificidade e matriz de confusão | Não calculadas | Não calculadas | Não reportadas | Calculadas na reexecução (5.3); entram como métrica nova, sem valor de referência original |

---

## 8. Perguntas antes de assumir

| ID | Pergunta | Opções | Minha recomendação |
|---|---|---|---|
| P1 | O bug D4 em `fumante_hist` (fumantes diários sem histórico registrado) deve ser **reproduzido** no pipeline para manter a fidelidade, ou **corrigido**? | (a) Reproduzir, com teste que documenta o comportamento e ADR de oportunidade futura; (b) corrigir e regerar as métricas de referência, aceitando que o modelo servido difere do da dissertação | (a). A regra de fidelidade é inegociável e o TCC é sobre produtização. A correção vira ADR e, se você quiser, uma segunda run no MLflow para comparação (ajuda o mínimo de dez execuções) |
| P2 | O transformador refatorado fará `fit` só no treino (imputação KNN e encoder) e `transform` no teste e nos folds. Isso **muda levemente** a imputação em relação ao notebook (D2) e pode deslocar as métricas. Se a diferença ficar dentro da tolerância, sigo; se sair, qual é a regra? | (a) Tolerância vale e a divergência é bug a investigar; (b) reproduzir o vazamento do notebook para a run de fidelidade e registrar a versão sem vazamento como segunda run | (b) para a run de fidelidade estrita e (a) para o modelo que vai para Production. Preciso da sua confirmação porque é exatamente o caso em que "divergência é bug" e "vazamento é errado" colidem |
| P3 | Em inferência, como tratar exame laboratorial ausente (eGFR, colesterol, glicose)? | (a) Campos obrigatórios na API; (b) aplicar o `KNNImputer` ajustado no treino e persistido, sinalizando na resposta que houve imputação | (b), porque é o comportamento do modelo original e a interface pede que a predição sempre possa rodar; a resposta da API carrega a flag de imputação |
| P4 | Os golden values do teste de regressão: usar os valores da reexecução (5.3, ambiente diferente do original) ou congelar a primeira execução do pipeline refatorado no ambiente travado? | (a) Reexecução; (b) primeira run do repo, com os valores de 5.3 como verificação externa | (b), com os dois arquivos versionados e a diferença entre eles reportada em `Evidencias_Consolidadas.md` |
| P5 | Definição operacional do alvo: confirma `PAS >= 140 ou PAD >= 90` sem considerar diagnóstico autorreferido (Q002) nem uso de medicação (Q006), que estão comentados no código? | Sim / Não | Sim, é o que roda e o que a dissertação descreve |
| P6 | O CV k=10 de referência roda sobre o desenvolvimento completo (6.203), não sobre o treino. Manter isso como a avaliação de aceitação do OE5, com o holdout 80/20 como evidência complementar? | Sim / Não | Sim, é a única forma de comparar com a Tabela 4 |
| P7 | Data de coleta e licença dos microdados da PNS 2013 para o dicionário de dados: você tem a fonte documental do IBGE que usou na dissertação, ou busco na documentação pública e cito? | | Buscar e citar, sujeito à sua conferência |

Nenhuma linha de `src/` será escrita antes das respostas a P1, P2 e P3, que alteram o desenho do transformador.

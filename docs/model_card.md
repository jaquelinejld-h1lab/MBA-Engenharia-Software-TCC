# Model card: risco de hipertensão arterial (hypertension-lr)

| Campo | Valor |
|---|---|
| Modelo | Regressão logística (`LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced', random_state=42)`), sem escalonamento, 33 variáveis categóricas em 85 dummies (`drop='first'`) |
| Origem | Dias (2024), dissertação do MECAI/ICMC USP; reproduzido dentro da tolerância de fidelidade (`|ΔAUC| ≤ 0,005`, `|Δsens| ≤ 0,01`) |
| Versão servida | Run `lr_production` (transformador ajustado só no treino, ADR 0008), registrada como `hypertension-lr`, estágio Production, alias `champion` |
| Código | `src/features/transformer.py`, `src/models/train.py`; parâmetros em `configs/config.yaml` |
| Responsável | Jaqueline Dias (MBA em Engenharia de Software, USP/Esalq) |
| Disclaimer | A saída é apoio à decisão e não substitui avaliação e julgamento clínico de profissional de saúde. Está no README, na resposta da API e na interface |

## 1. Uso pretendido

Estimar a probabilidade de um adulto (18 anos ou mais) apresentar pressão arterial
elevada (PAS ≥ 140 mmHg ou PAD ≥ 90 mmHg) a partir de questionário sociodemográfico,
de hábitos e de saúde, antropometria e, quando disponíveis, três exames laboratoriais
(eGFR, colesterol total, glicemia). Uso previsto: triagem e priorização em contexto de
apoio à decisão, com interpretação por profissional de saúde. Saída: probabilidade,
faixa de risco (baixo < 0,30, moderado, alto ≥ 0,60), classe no limiar 0,5 e as cinco
contribuições mais influentes.

## 2. Usos fora do escopo

Diagnóstico de hipertensão; decisão terapêutica; pessoas com menos de 18 anos,
gestantes, pessoas em uso de medicação anti-hipertensiva ou já diagnosticadas
(todas excluídas da população de treino pelos filtros); populações fora do Brasil ou
fora do período da PNS 2013; qualquer uso que trate a faixa de risco como laudo.

## 3. População de treino

| Item | Valor |
|---|---|
| Fonte | PNS 2013 (IBGE), módulo laboratorial; `docs/dicionario_dados.md` |
| Conjunto de desenvolvimento | 6.203 adultos após seis filtros (consentimento, gestação, hipertensão gestacional, medicação, região, questionário alimentar) |
| Prevalência do alvo | 17,04 % (1.057 positivos) |
| Split | 80/20 estratificado (4.962 / 1.241), seed 42; validação cruzada estratificada k = 10 sobre o desenvolvimento completo (decisão P6) |
| Exames ausentes no treino | eGFR 596, colesterol 201, glicose 216 (imputados por KNN, k = 5) |
| Peso amostral | não utilizado (ADR 0009): as métricas descrevem a amostra, não a população brasileira |

## 4. Métricas globais

Run promovida `lr_production`; a run de fidelidade `lr_fidelity` reproduz o notebook e
difere em AUC -0,0004 e sensibilidade -0,0019 (`evidencias/fidelity_vs_production.md`).

| Métrica | CV k = 10 (média ± dp) | Holdout 20 % | Critério OE5 |
|---|---|---|---|
| AUC-ROC | 0,7484 ± 0,0215 | 0,7468 | ≥ 0,73: atendido |
| Sensibilidade | 0,6794 ± 0,0601 | 0,6777 | ≥ 0,65: atendido |
| Especificidade | 0,6885 ± 0,0158 | 0,7058 | |
| Precisão | 0,3093 ± 0,0223 | 0,3206 | |
| F1 | 0,4249 ± 0,0321 | 0,4353 | |
| Acurácia | 0,6869 ± 0,0183 | 0,7010 | |
| Matriz de confusão (VN, FP / FN, VP) | 3.543, 1.603 / 339, 718 (agregada nos folds) | 727, 303 / 68, 143 | |

A precisão baixa é consequência de `class_weight='balanced'` com prevalência de 17 %:
o modelo prioriza sensibilidade, o que é coerente com triagem, ao custo de cerca de
dois falsos positivos por verdadeiro positivo.

## 5. Desempenho por subgrupo (predições out-of-fold da CV k = 10, `evidencias/subgroup_metrics_lr_production.csv`)

Os mesmos números são servidos por `GET /model/subgroups` e aparecem na aba Equidade da
página Modelo da interface (ADR 0015), a partir de `subgroup_metrics.json` persistido no
diretório do run.

| Dimensão | Grupo | n | Prevalência | AUC | Sensibilidade | Especificidade |
|---|---|---:|---:|---:|---:|---:|
| Sexo | feminino | 3.421 | 12,9 % | 0,752 | 0,549 | 0,791 |
| Sexo | masculino | 2.782 | 22,1 % | 0,719 | 0,774 | 0,547 |
| Faixa etária | 18-29 | 1.160 | 5,8 % | 0,739 | 0,134 | 0,974 |
| Faixa etária | 30-39 | 1.581 | 9,9 % | 0,703 | 0,353 | 0,888 |
| Faixa etária | 40-49 | 1.438 | 17,2 % | 0,653 | 0,603 | 0,612 |
| Faixa etária | 50-59 | 1.003 | 21,8 % | 0,668 | 0,753 | 0,492 |
| Faixa etária | 60-69 | 624 | 33,3 % | 0,620 | 0,899 | 0,173 |
| Faixa etária | 70-79 | 280 | 39,3 % | 0,634 | 0,945 | 0,118 |
| Faixa etária | 80+ | 117 | 42,7 % | 0,650 | 0,980 | 0,075 |

Os n somam o conjunto de desenvolvimento (6.203), cada pessoa avaliada no fold em que
ficou fora do treino; grupos com menos de 100 pessoas seriam reportados sem métrica.

Leitura crítica: a discriminação (AUC) é semelhante entre sexos e cai nas faixas mais
velhas, mas a **sensibilidade varia de 0,13 (18-29) a 0,98 (80+)** e de 0,55
(mulheres) a 0,77 (homens). Isso não é um erro do modelo e sim consequência de um limiar
único (0,5) aplicado a um modelo em que a faixa etária é a variável de maior peso: em
grupos de baixa prevalência quase ninguém ultrapassa 0,5. Estratégia de mitigação
documentada, não implementada por fidelidade (ADR 0008 e 0009 mantêm o modelo original):

1. limiar por subgrupo (sexo e faixa etária) escolhido para sensibilidade alvo, com o
   custo em especificidade explicitado;
2. calibração das probabilidades (Platt ou isotônica) antes das faixas de risco;
3. modelo com termos de interação idade × sexo, avaliado com o mesmo protocolo.

Na interface, a faixa de risco e a probabilidade são exibidas juntas justamente para
que o profissional não dependa do corte em 0,5.

## 6. Limitações conhecidas do dado e do modelo

| Limitação | Efeito | Referência |
|---|---|---|
| `fumante_hist` com bug reproduzido (comparação string/inteiro) | Fumantes diários sem P051/P052 aparecem sem histórico de tabagismo | ADR 0001, 0007 |
| Renda ausente contada como zero | Faixa de renda 1 mistura baixa renda e não resposta | dicionário, item D7 |
| Imputação KNN dos exames sem os próprios exames como vizinhança | Predição sem exames depende de variáveis sociodemográficas; aviso na resposta | ADR 0011 |
| Sem peso amostral | Métricas da amostra, não da população | ADR 0009 |
| Dado de 2013 congelado | Deriva temporal de hábitos e prevalência não capturada; monitorada por PSI/KS em produção | `docs/runbook.md` |
| Sem calibração e limiar único | Probabilidades e faixas não calibradas por subgrupo | seção 5 |
| Sem tratamento de outliers no treino | A API rejeita apenas valores fora da faixa plausível (D8) | `configs/variable_mapping.yaml` |
| Diagnósticos autorreferidos | Variáveis `dx_*` refletem relato, não prontuário | dicionário |

## 7. Explicabilidade

Contribuições aditivas em log-odds (`coef_j * x_j`) das dummies ativas, exatas para o
modelo linear (ADR 0003). As variáveis brutas de maior importância global no treino
(média de `|coef * x|`, `evidencias/global_importance.json`): idade, doença do coração,
região, sexo, AVC, peso e altura (IMC) e medicação para dormir.

## 8. Monitoramento e manutenção

Drift de dados por PSI e KS por variável contra a referência do treino (limiares em
`monitoring.drift`); métricas RED em Prometheus e Grafana; retreino e rollback em
`docs/runbook.md`. Tag de referência: `v1.0`.

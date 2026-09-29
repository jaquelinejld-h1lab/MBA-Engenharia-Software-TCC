# Dicionário de dados

Gerado por `scripts/make_data_dictionary.py` a partir de `configs/variable_mapping.yaml`
(versão 2) e de `configs/config.yaml`. Não editar à mão.

## 1. Origem, licença e congelamento

| Item | Valor |
|---|---|
| Fonte | Pesquisa Nacional de Saúde 2013 (PNS 2013), IBGE em parceria com o Ministério da Saúde; módulo de exames laboratoriais (subamostra com coleta de sangue e urina) |
| Natureza | Dados secundários, públicos e anonimizados (microdados sem identificação do domicílio, do morador ou do município; ver `docs/checklist_privacidade.md`) |
| Período de coleta | Entrevistas de agosto de 2013 a fevereiro de 2014; exames laboratoriais coletados na subamostra no mesmo período. `[INCERTEZA]` conferir as datas exatas na documentação da PNS 2013 e do módulo laboratorial (IBGE, 2014; Szwarcwald et al., 2019) |
| Licença | Microdados de uso público, disponibilizados pelo IBGE para livre acesso com citação da fonte. `[INCERTEZA]` transcrever o termo de uso vigente do portal do IBGE na versão final do TCC |
| Atribuição | IBGE, Pesquisa Nacional de Saúde 2013, microdados do módulo laboratorial |
| Arquivo | `data/raw/EXAMES-PNS-2013-FINAL_05052023.xlsx` (aba `EXAMES-PNS-2013-FINAL_com-ponde`) |
| Data freeze | Extrato de 05052023 (data no nome do arquivo, formato DDMMAAAA), congelado no repositório; qualquer alteração falha o gate de qualidade pelo hash |
| SHA-256 | `80222fa535524829d531504109ac4b49d44e4e6d0c04c3c281ff972bd332d1f7` |
| Dimensão bruta | 8952 linhas × 509 colunas |
| Conjunto de desenvolvimento | 6203 linhas após os seis filtros populacionais (tabela abaixo); prevalência do alvo 17,04 % |
| Peso amostral | `PESO_LAB` presente e não utilizado (ADR 0009); as métricas valem para a amostra analítica |

### Filtros populacionais e alvo

| Ordem | Regra | Coluna PNS | Removidos |
|---|---|---|---:|
| 1 | Consentiu armazenar exames (`Z051 == 1`) | Z051 | 1016 |
| 2 | Não grávida nem sem resposta (`P005 not in {1, 3}`) | P005 | 101 |
| 3 | Sem hipertensão exclusiva da gravidez (`Q002 != 2`) | Q002 | 118 |
| 4 | Sem medicação para hipertensão nas duas semanas (`Q006 != 1`) | Q006 | 1501 |
| 5 | Região informada | REGIAO | 6 |
| 6 | Questionário alimentar respondido (`P006` não nulo) | P006 | 7 |

Alvo: `target = 1` se PAS (`W00407`) ≥ 140 mmHg ou PAD (`W00408`) ≥ 90 mmHg; registros sem pressão arterial são excluídos antes do alvo (nenhum sobrevive aos filtros).

## 2. Variáveis auxiliares (filtro e alvo, fora do caminho de inferência)

| PNS | Nome | Papel | Domínio |
|---|---|---|---|
| Z051 | `lab_consent` (questionario) | filter | 1, 2, 3 |
| P005 | `pregnant` (gravida) | filter | 1, 2, 3 |
| Q002 | `dx_hypertension` (diag_ha) | filter | 1, 2, 3 |
| Q006 | `hypertension_medication` (remed_ha) | filter | 1, 2 |
| W00407 | `systolic_bp` (pa_sistolica) | target | 50 a 300 |
| W00408 | `diastolic_bp` (pa_distolica) | target | 30 a 200 |

## 3. As 61 variáveis brutas do caminho de inferência

Nome em inglês é o único usado no código; `pt` é o nome do script original; `nulos` é a fração de ausentes observada no desenvolvimento e o máximo aceito pelo gate de qualidade.

### 3.1 Sociodemográfico (14)

| PNS | Nome (en) | Nome original (pt) | Rótulo | Tipo | Domínio | Nulos obs. / máx. | Observado no treino |
|---|---|---|---|---|---|---|---|
| Z002 | `age` | `idade` | Idade | int | 18 a 104 anos | 0.0% / 0.0% | mediana 42, [18, 104] |
| Z001 | `sex` | `sexo` | Sexo | int | 1 = Homem; 2 = Mulher | 0.0% / 0.0% | moda 2 |
| REGIAO | `region` | `regiao` | Região de residência | int | 1 = Norte; 2 = Nordeste; 3 = Sudeste; 4 = Sul; 5 = Centro-Oeste | 0.0% / 0.0% | moda 2 |
| Z003 | `race_color` | `etnia` | Cor ou raça | int | 1 = Branca; 2 = Preta; 3 = Amarela; 4 = Parda; 5 = Indígena; 9 = Ignorado | 0.0% / 0.0% | moda 4 |
| C001 | `household_size` | `num_moradores` | Número de moradores no domicílio | int | 1 a 30 pessoas | 0.0% / 0.0% | mediana 3, [1, 13] |
| C010 | `lives_with_spouse` | `conj_comp` | Vive com cônjuge | int | 1 = Sim; 2 = Não | 0.0% / 0.0% | moda 1 |
| E01602 | `income_main_job` | `vl_trab_mes` | Rendimento mensal do trabalho principal | float (opcional) | 0 a 200000 R$ | 38.5% / 48.0% | mediana 800, [5, 40000] |
| E01604 | `income_main_job_in_kind` | `vl_est_mercad_mes` | Rendimento em produtos do trabalho principal | float (opcional) | 0 a 200000 R$ | 99.5% / 100.0% | mediana 300, [40, 2400] |
| E01802 | `income_other_jobs` | `vl_trab_outros_mes` | Rendimento de outros trabalhos | float (opcional) | 0 a 200000 R$ | 96.7% / 100.0% | mediana 678, [20, 15000] |
| E01804 | `income_other_jobs_in_kind` | `vl_trab_est_mercad_mes` | Rendimento em produtos de outros trabalhos | float (opcional) | 0 a 200000 R$ | 100.0% / 100.0% | mediana 700, [700, 700] |
| F00102 | `income_pension` | `vl_aposent` | Aposentadoria ou pensão | float (opcional) | 0 a 200000 R$ | 84.4% / 94.0% | mediana 678, [70, 17000] |
| F00702 | `income_alimony` | `vl_pens_al` | Pensão alimentícia ou doação | float (opcional) | 0 a 200000 R$ | 95.9% / 100.0% | mediana 250, [30, 6000] |
| F00802 | `income_rent` | `vl_aluguel` | Aluguel ou arrendamento | float (opcional) | 0 a 200000 R$ | 96.6% / 100.0% | mediana 500, [50, 12000] |
| I001 | `health_insurance` | `plano_saude` | Possui plano de saúde | int | 1 = Sim; 2 = Não | 0.0% / 0.0% | moda 2 |

### 3.2 Antropometria (3)

| PNS | Nome (en) | Nome original (pt) | Rótulo | Tipo | Domínio | Nulos obs. / máx. | Observado no treino |
|---|---|---|---|---|---|---|---|
| Z004 | `weight_kg` | `peso` | Peso | float | 25 a 250 kg | 0.0% / 0.0% | mediana 67.3, [30, 179] |
| Z005 | `height_cm` | `altura` | Altura | float | 120 a 220 cm | 0.0% / 0.0% | mediana 162.4, [131, 199.3] |
| W00303 | `waist_cm` | `circ_cintura` | Circunferência da cintura | float | 40 a 200 cm | 0.0% / 0.0% | mediana 88.5, [50, 141.5] |

### 3.3 Comportamento (32)

| PNS | Nome (en) | Nome original (pt) | Rótulo | Tipo | Domínio | Nulos obs. / máx. | Observado no treino |
|---|---|---|---|---|---|---|---|
| P006 | `beans_days` | `feijao_dias` | Dias por semana que come feijão | int | 0 = 0 dia(s); 1 = 1 dia(s); 2 = 2 dia(s); 3 = 3 dia(s); 4 = 4 dia(s); 5 = 5 dia(s); 6 = 6 dia(s); 7 = 7 dia(s) | 0.0% / 0.0% | moda 7 |
| P007 | `salad_days` | `salada_dias` | Dias por semana que come salada crua | int | 0 = 0 dia(s); 1 = 1 dia(s); 2 = 2 dia(s); 3 = 3 dia(s); 4 = 4 dia(s); 5 = 5 dia(s); 6 = 6 dia(s); 7 = 7 dia(s) | 0.0% / 0.0% | moda 7 |
| P009 | `vegetables_days` | `verd_legu_dias` | Dias por semana que come verduras ou legumes cozidos | int | 0 = 0 dia(s); 1 = 1 dia(s); 2 = 2 dia(s); 3 = 3 dia(s); 4 = 4 dia(s); 5 = 5 dia(s); 6 = 6 dia(s); 7 = 7 dia(s) | 0.0% / 0.0% | moda 0 |
| P015 | `fish_days` | `peixe_dias` | Dias por semana que come peixe | int | 0 = 0 dia(s); 1 = 1 dia(s); 2 = 2 dia(s); 3 = 3 dia(s); 4 = 4 dia(s); 5 = 5 dia(s); 6 = 6 dia(s); 7 = 7 dia(s) | 0.0% / 0.0% | moda 0 |
| P016 | `juice_days` | `suco_dias` | Dias por semana que toma suco natural | int | 0 = 0 dia(s); 1 = 1 dia(s); 2 = 2 dia(s); 3 = 3 dia(s); 4 = 4 dia(s); 5 = 5 dia(s); 6 = 6 dia(s); 7 = 7 dia(s) | 0.0% / 0.0% | moda 0 |
| P018 | `fruit_days` | `frutas_dias` | Dias por semana que come frutas | int | 0 = 0 dia(s); 1 = 1 dia(s); 2 = 2 dia(s); 3 = 3 dia(s); 4 = 4 dia(s); 5 = 5 dia(s); 6 = 6 dia(s); 7 = 7 dia(s) | 0.0% / 0.0% | moda 7 |
| P021 | `soda_type` | `tipo_refri` | Tipo de refrigerante ou suco artificial | int (opcional) | 1 = Normal; 2 = Diet, light ou zero; 3 = Ambos | 28.5% / 38.0% | moda 1 |
| P024 | `milk_type` | `tipo_leite` | Tipo de leite | int (opcional) | 1 = Integral; 2 = Desnatado ou semidesnatado; 3 = Ambos | 27.5% / 37.0% | moda 1 |
| P025 | `sweets_days` | `doces_dias` | Dias por semana que come doces | int | 0 = 0 dia(s); 1 = 1 dia(s); 2 = 2 dia(s); 3 = 3 dia(s); 4 = 4 dia(s); 5 = 5 dia(s); 6 = 6 dia(s); 7 = 7 dia(s) | 0.0% / 0.0% | moda 0 |
| P026 | `meal_replacement_days` | `subst_ref_dias` | Dias por semana que substitui almoço ou jantar por lanche | int | 0 = 0 dia(s); 1 = 1 dia(s); 2 = 2 dia(s); 3 = 3 dia(s); 4 = 4 dia(s); 5 = 5 dia(s); 6 = 6 dia(s); 7 = 7 dia(s) | 0.0% / 0.0% | moda 0 |
| P02601 | `salt_intake` | `consumo_sal` | Consumo de sal (autoavaliação) | int | 1 = Muito alto; 2 = Alto; 3 = Adequado; 4 = Baixo; 5 = Muito baixo | 0.0% / 0.0% | moda 3 |
| P027 | `alcohol` | `alcool` | Frequência de consumo de bebida alcoólica | int | 1 = Nunca; 2 = Menos de uma vez por mês; 3 = Uma vez ou mais por mês | 0.0% / 0.0% | moda 1 |
| P034 | `exercise` | `exercicio` | Praticou exercício ou esporte nos últimos 3 meses | int | 1 = Sim; 2 = Não | 0.0% / 0.0% | moda 2 |
| P035 | `exercise_days` | `exercicio_dias` | Dias por semana de exercício | int (opcional) | 0 = 0 dia(s); 1 = 1 dia(s); 2 = 2 dia(s); 3 = 3 dia(s); 4 = 4 dia(s); 5 = 5 dia(s); 6 = 6 dia(s); 7 = 7 dia(s) | 70.5% / 80.0% | moda 1 |
| P036 | `exercise_type` | `tipo_exercicio` | Tipo principal de exercício | int (opcional) | 1 = Caminhada; 2 = Caminhada em esteira; 3 = Corrida; 4 = Corrida em esteira; 5 = Musculação; 6 = Ginástica aeróbica; 7 = Hidroginástica; 8 = Ginástica em geral; 9 = Natação; 10 = Artes marciais e luta; 11 = Bicicleta; 12 = Futebol; 13 = Basquetebol; 14 = Voleibol; 15 = Tênis; 16 = Dança; 17 = Outro | 71.6% / 82.0% | moda 1 |
| P042 | `walk_bike_days` | `atv_pe_bk` | Dias por semana de deslocamento a pé ou bicicleta para atividades | int | 0 = 0 dia(s); 1 = 1 dia(s); 2 = 2 dia(s); 3 = 3 dia(s); 4 = 4 dia(s); 5 = 5 dia(s); 6 = 6 dia(s); 7 = 7 dia(s) | 0.0% / 0.0% | moda 0 |
| P04301 | `walk_bike_hours` | `atv_pe_bk_horas` | Horas por dia desse deslocamento | int (opcional) | 0 a 24 horas | 62.5% / 73.0% | mediana 0, [0, 8] |
| P04302 | `walk_bike_minutes` | `atv_pe_bk_minutos` | Minutos por dia desse deslocamento | int (opcional) | 0 a 59 minutos | 62.5% / 73.0% | mediana 20, [0, 56] |
| P039 | `work_physical_effort` | `trab_esforco` | Trabalho exige esforço físico intenso | int (opcional) | 1 = Sim; 2 = Não | 37.4% / 47.0% | moda 2 |
| P03901 | `work_effort_days` | `trab_esf_dias` | Dias por semana de esforço no trabalho | int (opcional) | 1 = 1 dia(s); 2 = 2 dia(s); 3 = 3 dia(s); 4 = 4 dia(s); 5 = 5 dia(s); 6 = 6 dia(s); 7 = 7 dia(s) | 80.5% / 90.0% | moda 5 |
| P03902 | `work_effort_hours` | `trab_esf_horas` | Horas por dia de esforço no trabalho | int (opcional) | 0 a 24 horas | 80.5% / 90.0% | mediana 3, [0, 16] |
| P03903 | `work_effort_minutes` | `trab_esf_minutos` | Minutos por dia de esforço no trabalho | int (opcional) | 0 a 59 minutos | 80.5% / 90.0% | mediana 0, [0, 59] |
| P040 | `commute_walk_bike` | `trab_pe_bk` | Vai ao trabalho a pé ou de bicicleta | int (opcional) | 1 = Sim, todo o trajeto; 2 = Sim, parte do trajeto; 3 = Não | 37.4% / 47.0% | moda 3 |
| P04101 | `commute_hours` | `trab_pe_bk_horas` | Horas por dia no deslocamento ao trabalho | int (opcional) | 0 a 24 horas | 72.8% / 83.0% | mediana 0, [0, 7] |
| P04102 | `commute_minutes` | `trab_pe_bk_minutos` | Minutos por dia no deslocamento ao trabalho | int (opcional) | 0 a 59 minutos | 72.8% / 83.0% | mediana 15, [0, 59] |
| P044 | `chores_physical_effort` | `atv_dom_esf` | Faz faxina pesada ou atividade doméstica intensa | int | 1 = Sim; 2 = Não | 0.0% / 0.0% | moda 2 |
| P04401 | `chores_days` | `atv_dom_esf_dias` | Dias por semana de atividade doméstica intensa | int (opcional) | 1 = 1 dia(s); 2 = 2 dia(s); 3 = 3 dia(s); 4 = 4 dia(s); 5 = 5 dia(s); 6 = 6 dia(s); 7 = 7 dia(s) | 79.9% / 90.0% | moda 1 |
| P04403 | `chores_hours` | `atv_dom_esf_horas` | Horas por dia de atividade doméstica intensa | int (opcional) | 0 a 24 horas | 79.9% / 90.0% | mediana 3, [0, 8] |
| P04404 | `chores_minutes` | `atv_dom_esf_minutos` | Minutos por dia de atividade doméstica intensa | int (opcional) | 0 a 59 minutos | 79.9% / 90.0% | mediana 0, [0, 58] |
| P050 | `smokes_currently` | `fuma_atual` | Fuma atualmente | int | 1 = Sim, diariamente; 2 = Sim, menos que diariamente; 3 = Não | 0.0% / 0.0% | moda 3 |
| P051 | `smoked_daily_past` | `fumo_rotina_hist` | Já fumou diariamente no passado | int (opcional) | 1 = Sim; 2 = Não | 97.5% / 100.0% | moda 1 |
| P052 | `smoked_past` | `fumo_hist` | Já fumou no passado | int (opcional) | 1 = Sim, diariamente; 2 = Sim, menos que diariamente; 3 = Não | 15.2% / 25.0% | moda 3 |

### 3.4 Saúde (9)

| PNS | Nome (en) | Nome original (pt) | Rótulo | Tipo | Domínio | Nulos obs. / máx. | Observado no treino |
|---|---|---|---|---|---|---|---|
| Q124 | `dx_chronic_kidney` | `diag_renalc` | Diagnóstico de insuficiência renal crônica | int | 1 = Sim; 2 = Não | 0.0% / 0.0% | moda 2 |
| Q030 | `dx_diabetes` | `diag_diab` | Diagnóstico de diabetes | int (opcional) | 1 = Sim; 2 = Não; 3 = Apenas durante a gravidez | 14.9% / 25.0% | moda 3 |
| Q063 | `dx_heart_disease` | `diag_dcore` | Diagnóstico de doença do coração | int | 1 = Sim; 2 = Não | 0.0% / 0.0% | moda 2 |
| Q068 | `dx_stroke` | `diag_avc` | Diagnóstico de AVC | int | 1 = Sim; 2 = Não | 0.0% / 0.0% | moda 2 |
| Q060 | `dx_high_cholesterol` | `diag_colest` | Diagnóstico de colesterol alto | int (opcional) | 1 = Sim; 2 = Não | 18.8% / 29.0% | moda 2 |
| Q132 | `sleep_medication` | `med_dormir` | Usa medicamento para dormir | int | 1 = Sim; 2 = Não | 0.0% / 0.0% | moda 2 |
| N001 | `self_rated_health` | `pcp_saude` | Autopercepção do estado de saúde | int | 1 = Muito bom; 2 = Bom; 3 = Regular; 4 = Ruim; 5 = Muito ruim | 0.0% / 0.0% | moda 2 |
| N004 | `angina_walking_fast` | `angina_rapd` | Dor no peito ao andar rápido ou subir | int (opcional) | 1 = Sim; 2 = Não; 3 = Nunca anda rápido ou sobe ladeira | 1.0% / 11.0% | moda 2 |
| N005 | `angina_walking_normal` | `angina_norm` | Dor no peito ao andar em ritmo normal | int (opcional) | 1 = Sim; 2 = Não | 1.7% / 12.0% | moda 2 |

### 3.5 Exames laboratoriais (3)

| PNS | Nome (en) | Nome original (pt) | Rótulo | Tipo | Domínio | Nulos obs. / máx. | Observado no treino |
|---|---|---|---|---|---|---|---|
| Z026 | `egfr_afro` | `egfr_afro` | Filtração glomerular estimada (afrodescendente) | float (opcional) | 1 a 600 mL/min/1,73 m2 | 11.9% / 22.0% | mediana 106.433, [4.40461, 557.776] |
| Z031 | `cholesterol` | `colesterol` | Colesterol total | float (opcional) | 40 a 600 mg/dL | 3.9% / 14.0% | mediana 183, [68, 433] |
| Z035 | `glucose` | `glicose` | Glicose média estimada | float (opcional) | 30 a 600 mg/dL | 4.3% / 14.0% | mediana 105.41, [56.046, 411.352] |

## 4. As 33 variáveis do modelo (85 dummies com `drop='first'`)

Todas categóricas em string; a primeira categoria é a referência (dummy omitida). As regras de derivação estão em `src/features/derivations.py` com os parâmetros em `configs/config.yaml::features`.

| Nome (en) | Nome original (pt) | Rótulo | Categorias (referência primeiro) | Variáveis brutas de origem |
|---|---|---|---|---|
| `age_group` | `faixa_etaria` | Faixa etária | 18-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80+ | `age` |
| `sex` | `sexo` | Sexo | 1, 2 | `sex` |
| `region` | `regiao` | Região | 1, 2, 3, 4, 5 | `region` |
| `black_or_brown` | `etnia_negra` | Cor preta ou parda | 1, 2 | `race_color` |
| `household_size_group` | `faixa_num_mor` | Faixa de moradores | 1, 2, 3, 4, 5, 6+ | `household_size` |
| `lives_with_spouse` | `conj_comp` | Vive com cônjuge | 1, 2 | `lives_with_spouse` |
| `income_bracket` | `faixa_renda_sl` | Faixa de renda em salários mínimos | 1, 2, 3, 4, 5, 6, 7, 8, 9 | `income_main_job`, `income_main_job_in_kind`, `income_other_jobs`, `income_other_jobs_in_kind`, `income_pension`, `income_alimony`, `income_rent` |
| `health_insurance` | `plano_saude` | Plano de saúde | 1, 2 | `health_insurance` |
| `dx_chronic_kidney` | `diag_renalc` | Insuficiência renal crônica | 1, 2 | `dx_chronic_kidney` |
| `dx_diabetes_rec` | `diag_diab_rec` | Diabetes (recodificado) | 1, 2, 3 | `dx_diabetes` |
| `dx_heart_disease` | `diag_dcore` | Doença do coração | 1, 2 | `dx_heart_disease` |
| `dx_stroke` | `diag_avc` | AVC | 1, 2 | `dx_stroke` |
| `beans_intake` | `feijao_consumo` | Consumo de feijão | 1, 2, 3, 4 | `beans_days` |
| `salad_intake` | `salada_consumo` | Consumo de salada | 1, 2, 3, 4 | `salad_days` |
| `vegetables_intake` | `verd_legu_consumo` | Consumo de verduras e legumes | 1, 2, 3, 4 | `vegetables_days` |
| `fish_intake` | `peixe_consumo` | Consumo de peixe | 1, 2, 3, 4 | `fish_days` |
| `juice_intake` | `suco_consumo` | Consumo de suco natural | 1, 2, 3, 4 | `juice_days` |
| `fruit_intake` | `frutas_consumo` | Consumo de frutas | 1, 2, 3, 4 | `fruit_days` |
| `soda_type_rec` | `tipo_refri_rec` | Tipo de refrigerante (recodificado) | 1, 2, 3, 4 | `soda_type` |
| `milk_type_rec` | `tipo_leite_rec` | Tipo de leite (recodificado) | 1, 2, 3, 4 | `milk_type` |
| `sweets_intake` | `doces_consumo` | Consumo de doces | 1, 2, 3, 4 | `sweets_days` |
| `meal_replacement_freq` | `freq_subst_ref` | Substituição de refeição por lanche | 1, 2, 3, 4 | `meal_replacement_days` |
| `salt_intake_rec` | `consumo_sal_rec` | Consumo de sal (recodificado) | 1, 2, 3 | `salt_intake` |
| `alcohol` | `alcool` | Bebida alcoólica | 1, 2, 3 | `alcohol` |
| `physical_activity_level` | `nivel_atv_fisica` | Nível de atividade física | 1, 2, 3, 4 | `exercise`, `exercise_days`, `exercise_type`, `walk_bike_days`, `walk_bike_hours`, `walk_bike_minutes`, `work_physical_effort`, `work_effort_days`, `work_effort_hours`, `work_effort_minutes`, `commute_walk_bike`, `commute_hours`, `commute_minutes`, `chores_physical_effort`, `chores_days`, `chores_hours`, `chores_minutes` |
| `smoking_history` | `fumante_hist` | Histórico de tabagismo | 1, 2 | `smokes_currently`, `smoked_daily_past`, `smoked_past` |
| `sleep_medication` | `med_dormir` | Medicamento para dormir | 1, 2 | `sleep_medication` |
| `bmi_class` | `class_imc` | Classificação do IMC | 1, 2, 3, 4, 5 | `weight_kg`, `height_cm` |
| `self_rated_health_rec` | `pcp_saude_rec` | Autopercepção de saúde (recodificado) | 1, 2 | `self_rated_health` |
| `angina` | `angina` | Angina | 1, 2 | `angina_walking_fast`, `angina_walking_normal` |
| `egfr_class` | `cat_egfr_afro` | Classe de filtração glomerular | 1, 2, 3, 4, 5 | `egfr_afro` |
| `glucose_class` | `cat_glicose` | Classe de glicose | 1, 2, 3, 4 | `glucose` |
| `cholesterol_desirable` | `colesterol_ideal` | Colesterol desejável | 1, 2 | `cholesterol` |

### Variáveis intermediárias (base da imputação KNN, não entram no modelo)

| Nome (en) | Nome original (pt) | Categorias | Nota |
|---|---|---|---|
| `race_color` | `etnia` | 1, 2, 3, 4, 5, 9 | raw race/color code, used as imputation base |
| `waist_risk_increased` | `cintura_risco_aumentado` | 1, 2 | derived from waist_cm and sex, imputation base only |
| `dx_high_cholesterol_rec` | `diag_colest_rec` | 1, 2, 3 | dx_high_cholesterol with NaN -> 3, imputation base only |

## 5. Regras com efeito semântico (herdadas do modelo original)

| Regra | Efeito | Referência |
|---|---|---|
| Renda ausente conta como zero (D7) | Parte da faixa de renda 1 é ausência de resposta, não renda baixa | inventário 6.2 |
| `fumante_hist` compara string com inteiro (D4) | Fumantes diários sem resposta em P051/P052 ficam sem histórico | ADR 0001 e 0007 |
| Exames ausentes imputados por KNN (k = 5) sobre 12 variáveis | Predição possível sem exames, com aviso (`imputed_fields`) | ADR 0011 |
| Códigos 4 em tipo de refrigerante e de leite significam ausência de resposta (D10) | Só surgem de valor nulo; códigos fora de 1 a 3 são rejeitados | `derivations.py` |
| Peso amostral ignorado (D6) | Métricas não ponderadas | ADR 0009 |

## 6. Retenção e descarte

Ver `docs/checklist_privacidade.md`: o extrato congelado permanece no repositório enquanto o modelo for mantido; dados enviados à API não são persistidos em disco.

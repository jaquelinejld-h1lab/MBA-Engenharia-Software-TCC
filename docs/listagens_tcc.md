# Listagens de código curadas para o TCC

Um trecho por subtópico da seção de Implementação (S07 a S12, uma por etapa do
pipeline), com no máximo 25 linhas, numeração e comentário didático em cada passo.
Os trechos são recortes fiéis do código versionado (arquivo e função indicados);
docstrings e logging foram omitidos para caber no limite. O excedente está no
Apêndice C, ao final.

Premissa adotada: S07 a S12 correspondem, na ordem, às seis etapas do pipeline
(coleta e pré-processamento; treino e validação; versionamento e rastreamento; testes
automatizados; implantação e containerização; monitoramento e retreinamento).

## Listagem S07: filtros populacionais e alvo (`src/data/filters.py::apply_population_filters`)

```python
 1  def apply_population_filters(frame, filters, target):
 2      steps = []                                   # relatório: linhas removidas por filtro
 3      f = frame
 4      # 1. só quem consentiu armazenar os exames laboratoriais (Z051 == 1)
 5      f = _step(f, f[filters.consent_column] == filters.consent_keep_value, "lab_consent", steps)
 6      # 2. exclui gestantes e "não sabe" (P005 in {1, 3})
 7      f = _step(f, ~f[filters.pregnancy_column].isin(filters.pregnancy_drop_values), "pregnant", steps)
 8      # 3. exclui hipertensão diagnosticada só na gravidez (Q002 == 2)
 9      f = _step(f, f[filters.hypertension_pregnancy_only_column]
10                != filters.hypertension_pregnancy_only_drop_value, "hypertension_pregnancy_only", steps)
11      # 4. exclui quem tomou remédio para hipertensão nas duas semanas (Q006 == 1)
12      f = _step(f, f[filters.medication_column] != filters.medication_drop_value, "medication", steps)
13      # 5 e 6. região informada e questionário alimentar respondido (proxy P006)
14      f = _step(f, f[filters.region_column].notna(), "region_missing", steps)
15      f = _step(f, f[filters.empty_questionnaire_proxy_column].notna(), "empty_questionnaire", steps)
16      # D1: pressão arterial precisa existir antes de definir o alvo (exclusão contada)
17      sbp = pd.to_numeric(f[target.systolic_column], errors="coerce")
18      dbp = pd.to_numeric(f[target.diastolic_column], errors="coerce")
19      bp_missing = sbp.isna() | dbp.isna()
20      if target.drop_if_blood_pressure_missing:
21          f, sbp, dbp = f.loc[~bp_missing], sbp.loc[~bp_missing], dbp.loc[~bp_missing]
22      # alvo binário: PAS >= 140 ou PAD >= 90 mmHg (limiares em configs/config.yaml)
23      f = f.assign(target=((sbp >= target.systolic_threshold_mmhg)
24                           | (dbp >= target.diastolic_threshold_mmhg)).astype(int))
25      return f.reset_index(drop=True), FilterReport(len(frame), steps, int(bp_missing.sum()), len(f), f["target"].mean())
```

Cada filtro passa por `_step`, que registra o nome e a contagem de removidos; o CI
compara essas contagens com `population_filters.expected_removed` (1.016, 101, 118,
1.501, 6, 7) e falha se o dado mudar.

## Listagem S08: validação cruzada com transformador ajustado por fold (`src/models/cross_validation.py::cross_validate`)

```python
 1  def cross_validate(development, transformer, estimator, split, seed, threshold):
 2      features = development.drop(columns=[TARGET_COLUMN])      # 61 variáveis brutas
 3      y = development[TARGET_COLUMN].to_numpy()                 # alvo 0/1
 4      folds = StratifiedKFold(split.n_splits, shuffle=True, random_state=seed)  # k = 10, seed 42
 5      oof = np.full(len(y), np.nan)                             # probabilidade out-of-fold por linha
 6      fold_metrics = []
 7      encoded_all = None
 8      if transformer.mode == "fidelity":
 9          # protocolo do notebook: ajusta o transformador no desenvolvimento inteiro
10          encoded_all = clone(transformer).fit(features).transform(features)
11      for k, (train_idx, test_idx) in enumerate(folds.split(features, y)):
12          if encoded_all is not None:                           # modo fidelidade
13              x_train, x_test = encoded_all.iloc[train_idx], encoded_all.iloc[test_idx]
14          else:                                                 # modo produção: fit só no treino do fold
15              fold_transformer = clone(transformer).fit(features.iloc[train_idx])
16              x_train = fold_transformer.transform(features.iloc[train_idx])
17              x_test = fold_transformer.transform(features.iloc[test_idx])
18          model = clone(estimator).fit(x_train, y[train_idx])   # estimador novo a cada fold
19          prob = model.predict_proba(x_test)[:, 1]              # probabilidade da classe positiva
20          oof[test_idx] = prob
21          # métricas de rótulo pelo predict (semântica do notebook; ADR 0002)
22          fold_metrics.append(compute_metrics(y[test_idx], prob, threshold, y_pred=model.predict(x_test)))
23      return CrossValidationResult(fold_metrics=fold_metrics,
24                                   summary=summarize_folds(fold_metrics),   # média e desvio por métrica
25                                   oof_probabilities=oof)
```

O modo `fidelity` reproduz o vazamento leve do notebook (imputação ajustada em todos os
folds) e gera os golden values; o modo `production` é o promovido (ADR 0008).

## Listagem S09: rastreamento e registro no MLflow (`src/models/registry.py::MlflowTracker`)

```python
 1  class MlflowTracker:
 2      def __init__(self, tracking):
 3          db_path = resolve(tracking.sqlite_path)                 # mlruns/mlflow.db, relativo à raiz
 4          mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
 5          if mlflow.get_experiment_by_name(tracking.experiment_name) is None:
 6              mlflow.create_experiment(tracking.experiment_name,   # cria o experimento uma vez
 7                                       artifact_location=resolve(tracking.artifact_root).as_uri())
 8          mlflow.set_experiment(tracking.experiment_name)
 9
10      def start_run(self, name, tags):
11          active = mlflow.start_run(run_name=name, tags=tags)      # uma run por algoritmo/configuração
12          return _MlflowRun(mlflow, active.info.run_id)            # log_params, log_metrics, log_artifact
13
14      def register_model(self, run, model_dir, name):
15          pipeline = joblib.load(model_dir / "pipeline.joblib")   # transformador + modelo em um só objeto
16          mlflow.sklearn.log_model(pipeline, artifact_path="model")
17          for extra in model_dir.glob("*.json"):                   # métricas, ROC e referência de drift
18              mlflow.log_artifact(str(extra))
19          version = mlflow.register_model(f"runs:/{run.run_id}/model", name)
20          return str(version.version)                              # versão numérica no registry
21
22      def promote(self, name, version, stage, alias):
23          client = mlflow.tracking.MlflowClient()
24          client.transition_model_version_stage(name=name, version=version, stage=stage,
25                                                archive_existing_versions=True)   # Production; depois alias "champion"
```

Cada run recebe como parâmetros o SHA-256 do dado bruto e do desenvolvimento, a revisão
do código e o hash do transformador (`execute_run`), o que fecha o critério "100 % dos
datasets com hash" do OE2.

## Listagem S10: teste de contrato com paridade online/offline (`tests/contract/test_api_contract.py`)

```python
 1  @pytest.fixture(scope="module")
 2  def loaded(cfg):
 3      return train_fixture_pipeline(cfg)             # pipeline treinado na fixture sintética
 4
 5  @pytest.fixture(scope="module")
 6  def client(cfg, loaded):
 7      app = create_app(cfg, model=loaded, cache=ModelCache())   # injeta o modelo, sem MLflow
 8      with TestClient(app) as test_client:            # dispara o lifespan (fail fast se não carregar)
 9          yield test_client
10
11  def test_predict_validation_error_is_422(client, rows):
12      payload = payload_from_row(rows.iloc[0]) | {"age": 999}   # idade fora do domínio (18 a 104)
13      response = client.post("/predict", json=payload)
14      assert response.status_code == 422             # contrato Pydantic rejeita antes do modelo
15      assert "Traceback" not in response.text        # envelope JSON, nunca stack trace
16
17  def test_online_offline_parity_50_rows(client, rows, loaded):
18      offline = loaded.pipeline.predict_proba(rows)[:, 1]       # caminho de treino (batch)
19      online = []
20      for _, row in rows.iterrows():                 # caminho de produção (uma requisição por linha)
21          response = client.post("/predict", json=payload_from_row(row))
22          assert response.status_code == 200, response.text
23          online.append(response.json()["probability"])
24      # mesma probabilidade pelos dois caminhos: sem training/serving skew
25      assert float(np.max(np.abs(np.asarray(online) - offline))) < 1e-9
```

A suíte tem 261 testes (unitários, propriedade, domínio, integração, contrato e
schemathesis) com gate de cobertura de 80 % no CI.

## Listagem S11: carga do modelo e endpoint `/predict` (`src/api/model_loader.py` e `src/api/main.py`)

```python
 1  def _check_pipeline(obj, origin):
 2      if not isinstance(obj, Pipeline) or set(obj.named_steps) != {"transformer", "model"}:
 3          raise ModelNotFoundError(f"{origin}: expected a Pipeline with steps 'transformer' and 'model'")
 4      transformer = obj.named_steps["transformer"]
 5      try:                                            # guarda de esquema: artefato de config antiga
 6          FeaturesConfig.model_validate(transformer.features.model_dump())
 7      except (ValidationError, AttributeError) as error:
 8          raise ModelNotFoundError(f"{origin}: incompatible configuration schema; retrain") from error
 9      return obj
10
11  @asynccontextmanager
12  async def lifespan(_):
13      loaded = model_cache.get(settings)              # carrega uma vez; falha rápido sem modelo
14      monitor.reference = load_drift_reference(settings, loaded)   # janela de referência do treino
15      yield
16
17  @app.post("/predict", response_model=PredictionResponse)
18  async def predict(payload: PredictionRequest, request: Request, explain: bool = False,
19                    service: PredictionService = Depends(get_service)):
20      record = payload.model_dump()                   # 61 variáveis já validadas pelo contrato
21      prediction = service.predict_records([record], explain=explain)[0]   # transformador + LR
22      metrics.observe_prediction(prediction.risk_band)   # contador Prometheus por faixa de risco
23      monitor.observe([record])                       # alimenta a janela de drift
24      return to_response(prediction, service.model,   # probabilidade, faixa, versão, imputados,
25                         correlation_id_of(request))  # contribuições, disclaimer, correlation id
```

A imagem Docker (multi-stage, usuário não root, somente `requirements-api.lock`) executa
`uvicorn src.api.main:app`; o `docker compose` sobe API, interface, MLflow, Prometheus
e Grafana.

## Listagem S12: PSI com bin de ausência e KS por variável (`src/monitoring/drift.py`)

```python
 1  def psi(reference, current, epsilon):
 2      r = np.asarray(list(reference), dtype=float)      # proporções de referência por bin
 3      c = np.asarray(list(current), dtype=float)        # proporções da janela atual por bin
 4      if r.shape != c.shape:
 5          raise DriftError("PSI share vectors differ in length")
 6      r = np.where(r <= 0, epsilon, r)                  # evita log(0): bin vazio vira epsilon
 7      c = np.where(c <= 0, epsilon, c)
 8      return float(np.sum((c - r) * np.log(c / r)))    # PSI = soma (c - r) * ln(c / r)
 9
10  def _numeric_shares(values, edges):
11      numeric = pd.to_numeric(values, errors="coerce")
12      counts, _ = np.histogram(numeric.dropna(), bins=edges)   # bins por quantis da referência
13      n_missing = int(numeric.isna().sum())            # ausência é um bin próprio (última posição)
14      all_counts = np.append(counts, n_missing).astype(float)
15      return all_counts / all_counts.sum(), n_missing / all_counts.sum()
16
17  def ks_test(reference_sample, current_sample, min_rows):
18      if reference_sample.size < min_rows or current_sample.size < min_rows:
19          return math.nan, math.nan                     # amostra pequena: KS não é reportado
20      result = stats.ks_2samp(reference_sample, current_sample)   # KS de duas amostras
21      return float(result.statistic), float(result.pvalue)
22
23  def psi_level(value, drift):                          # limiares em monitoring.drift
24      return ("alert" if value >= drift.psi_alert
25              else "warning" if value >= drift.psi_warning else "ok")
```

A API recalcula PSI e KS sobre a janela deslizante a cada raspagem de `/metrics` e
publica `data_drift_psi{variable}`, `data_drift_ks_statistic{variable}` e
`data_drift_alert{variable}`; o Prometheus dispara `DataDriftDetected` e o Grafana exibe
o ranking.

## Apêndice C: excedentes

### C.1 Transformador (`src/features/transformer.py::HypertensionTransformer.fit` e `transform_detailed`)

```python
 1  def fit(self, X, y=None):
 2      self._check_mode()                                        # "fidelity" ou "production"
 3      derived = derive_stateless(X, self.features)              # 33 derivações puras + bases da imputação
 4      self.imputation_ = self._fit_imputation(derived)          # one-hot da base + KNNImputer por exame
 5      model_frame, _ = self._complete(derived, self.imputation_)
 6      self.encoder_ = OneHotEncoder(                            # categorias explícitas do mapping
 7          categories=[self.mapping.model_categories[c] for c in self.mapping.model_en_names],
 8          drop="first" if self.features.encoding.drop_first else None,
 9          sparse_output=False, handle_unknown="error", dtype=np.int64)
10      self.encoder_.fit(model_frame[self.mapping.model_en_names])
11      self.feature_names_out_ = list(self.mapping.dummy_names)  # 85 nomes fixos em inglês
12      return self
13
14  def transform_detailed(self, X):
15      self._check_fitted()
16      derived = derive_stateless(X, self.features)
17      # modo fidelidade reajusta a imputação no conjunto recebido, como o notebook
18      imputation = self._fit_imputation(derived) if self.mode == "fidelity" else self.imputation_
19      model_frame, imputed = self._complete(derived, imputation)  # imputa e categoriza os exames
20      try:
21          matrix = self.encoder_.transform(model_frame[self.mapping.model_en_names])
22      except ValueError as error:                               # categoria fora do mapping -> erro de domínio
23          raise TransformError(f"category outside the declared mapping: {error}") from error
24      encoded = pd.DataFrame(matrix, columns=self.feature_names_out_, index=X.index)
25      return TransformResult(encoded=encoded, model_frame=model_frame, imputed=imputed)
```

### C.2 Reprodução do bug de `fumante_hist` (`src/features/derivations.py::smoking_history`, ADR 0001)

```python
 1  def smoking_history(smokes_currently, smoked_daily_past, smoked_past, cfg):
 2      # no original, a flag de fumante atual era string ("1") comparada ao inteiro 1:
 3      # sempre falsa. Só P051 e P052 contam quando o bug é reproduzido (D4).
 4      ever = smoked_daily_past == 1 or smoked_past == 1
 5      if not cfg.smoking.reproduce_string_int_comparison_bug:   # variante corrigida (run lr_smoking_fixed)
 6          ever = ever or smokes_currently in (1, 2)             # fuma diariamente ou menos que diariamente
 7      return YES if ever else NO
```

### C.3 Contribuições lineares (`src/api/service.py::PredictionService._contributions`, ADR 0003)

```python
 1  def _contributions(self, encoded_row):
 2      weights = np.asarray(self.model.estimator.coef_).ravel()  # coeficientes da LR (85)
 3      values = encoded_row.to_numpy(dtype=float)                # dummies ativas 0/1
 4      contributions = weights * values                          # atribuição aditiva exata em log-odds
 5      active = [{"feature": name, "value": int(v), "contribution": float(c)}
 6                for name, v, c in zip(encoded_row.index, values, contributions) if v != 0]
 7      active.sort(key=lambda item: abs(item["contribution"]), reverse=True)
 8      return active[: self.cfg.api.top_contributions]           # cinco maiores em módulo
```

### C.4 Middleware de correlação e envelope de erro (`src/api/middleware.py`)

```python
 1  async def dispatch(self, request, call_next):
 2      correlation_id = request.headers.get(CORRELATION_HEADER) or str(uuid.uuid4())
 3      request.state.correlation_id = correlation_id             # propagado a handlers e logs
 4      started = time.perf_counter()
 5      status = 500
 6      try:
 7          response = await call_next(request)
 8          status = response.status_code
 9      finally:
10          elapsed = time.perf_counter() - started
11          self.metrics.observe_request(request.url.path, request.method, status, elapsed)
12          logger.info("request", extra={"correlation_id": correlation_id, "path": request.url.path,
13                                        "status": status, "duration_ms": round(elapsed * 1000, 3)})
14      response.headers[CORRELATION_HEADER] = correlation_id
15      return response
16
17  async def _domain_error(request, exc):                        # DataContractError/TransformError -> 422
18      status = next((code for kind, code in _STATUS_BY_ERROR.items() if isinstance(exc, kind)), 500)
19      return _envelope(request, status, type(exc).__name__, str(exc))   # sem stack trace
```

### C.5 Cenário de drift reprodutível (`src/monitoring/scenarios.py::apply_scenario`)

```python
 1  def apply_scenario(frame, scenario, seed):
 2      shifted = frame.copy()                                    # função pura: não altera a entrada
 3      for name, shift in scenario.numeric_shifts.items():       # ex.: age + 15, weight_kg * 1.2
 4          shifted[name] = shifted[name].astype(float) * shift.multiply + shift.add
 5      rng = np.random.default_rng(seed)                         # seleção de linhas determinística
 6      for name, override in scenario.categorical_overrides.items():   # ex.: 80 % das linhas -> região 3
 7          n_rows = round(override.share * len(shifted))
 8          chosen = rng.choice(len(shifted), size=n_rows, replace=False)
 9          shifted.iloc[chosen, shifted.columns.get_loc(name)] = float(override.code)
10      return shifted
```

### C.6 Teste de propriedade das derivações (`tests/unit/test_derivations_property.py`)

```python
 1  @given(days=st.integers(0, 7))                                # todo o domínio declarado no mapping
 2  def test_food_functions_stay_in_declared_categories(days, cfg):
 3      fc = cfg.features
 4      assert d.healthy_frequency(days, fc) in _CATEGORIES["beans_intake"]     # sempre categoria válida
 5      assert d.fish_frequency(days, fc) in _CATEGORIES["fish_intake"]
 6      assert d.sweets_frequency(days, fc) in _CATEGORIES["sweets_intake"]
 7
 8  @given(row=_raw_row_strategy())                                # linha aleatória dentro do domínio das 61 variáveis
 9  def test_full_row_derivation_satisfies_model_contract(row, cfg):
10      frame = pd.DataFrame([{k: (math.nan if v is None else v) for k, v in row.items()}])
11      derived = derive_stateless(frame, cfg.features)
12      labs = derive_laboratory_classes(derived.fillna({"egfr_afro": 100.0, "cholesterol": 100.0,
13                                                       "glucose": 100.0}), cfg.features)
14      record = {**derived.iloc[0].to_dict(), **labs.iloc[0].to_dict()}
15      ModelRecord.model_validate({name: record[name] for name in ModelRecord.model_fields})  # contrato das 33
```

def apply_population_filters(frame, filters, target):
      steps = []                                   # relatório: linhas removidas por filtro
      f = frame
      # 1. só quem consentiu armazenar os exames laboratoriais (Z051 == 1)
      f = _step(f, f[filters.consent_column] == filters.consent_keep_value, "lab_consent", steps)
      # 2. exclui gestantes e "não sabe" (P005 in {1, 3})
      f = _step(f, ~f[filters.pregnancy_column].isin(filters.pregnancy_drop_values), "pregnant", steps)
      # 3. exclui hipertensão diagnosticada só na gravidez (Q002 == 2)
      f = _step(f, f[filters.hypertension_pregnancy_only_column]
                != filters.hypertension_pregnancy_only_drop_value, "hypertension_pregnancy_only", steps)
      # 4. exclui quem tomou remédio para hipertensão nas duas semanas (Q006 == 1)
      f = _step(f, f[filters.medication_column] != filters.medication_drop_value, "medication", steps)
      # 5 e 6. região informada e questionário alimentar respondido (proxy P006)
      f = _step(f, f[filters.region_column].notna(), "region_missing", steps)
      f = _step(f, f[filters.empty_questionnaire_proxy_column].notna(), "empty_questionnaire", steps)
      # D1: pressão arterial precisa existir antes de definir o alvo (exclusão contada)
      sbp = pd.to_numeric(f[target.systolic_column], errors="coerce")
      dbp = pd.to_numeric(f[target.diastolic_column], errors="coerce")
      bp_missing = sbp.isna() | dbp.isna()
      if target.drop_if_blood_pressure_missing:
          f, sbp, dbp = f.loc[~bp_missing], sbp.loc[~bp_missing], dbp.loc[~bp_missing]
      # alvo binário: PAS >= 140 ou PAD >= 90 mmHg (limiares em configs/config.yaml)
      f = f.assign(target=((sbp >= target.systolic_threshold_mmhg)
                           | (dbp >= target.diastolic_threshold_mmhg)).astype(int))

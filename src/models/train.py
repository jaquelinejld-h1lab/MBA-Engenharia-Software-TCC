"""Experiment runner: train, evaluate, persist and track every configured run.

Run as a module::

    python -m src.models.train --all                 # every run in config
    python -m src.models.train --run lr_fidelity     # one run
    python -m src.models.train --all --backend json  # without MLflow
    python -m src.models.train --all --freeze-golden # rewrite golden metrics (P4)

Each run produces, under ``artifacts/<run>/``: the fitted transformer, its
KNN imputers and encoder, the estimator and a scikit-learn ``Pipeline`` of
both, all saved with joblib; plus ``metrics.json`` and ``roc.json``. The same
files are logged to the tracker together with the dataset hashes, the code
revision and the transformer source hash.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline

from src import __version__
from src.config import FeaturesConfig, RunConfig, Settings, get_config
from src.data.filters import TARGET_COLUMN, split_holdout
from src.data.mapping import get_mapping
from src.data.pipeline import load_development_frame
from src.exceptions import ModelNotFoundError
from src.features.transformer import HypertensionTransformer
from src.logging_setup import configure_logging, get_logger
from src.models.algorithms import build_estimator, library_available
from src.models.cross_validation import CrossValidationResult, cross_validate
from src.models.evaluate import (
    ClassificationMetrics,
    RocPoints,
    compute_metrics,
    roc_points,
    subgroup_metrics,
)
from src.models.interpretation import (
    INTERPRETATION_FILENAME,
    SUBGROUP_FILENAME,
    build_interpretation,
)
from src.models.plots import plot_confusion_matrix, plot_roc
from src.models.registry import (
    ExperimentTracker,
    frame_sha256,
    git_revision,
    make_tracker,
    source_sha256,
)
from src.models.rule_baseline import apply_rule, describe
from src.monitoring.drift import build_reference, save_reference
from src.paths import PROJECT_ROOT, resolve

logger = get_logger(__name__)

_FEATURES_DIR = PROJECT_ROOT / "src" / "features"


@dataclass(frozen=True)
class RunResult:
    """Everything one run produced."""

    name: str
    role: str
    cv: CrossValidationResult
    holdout: ClassificationMetrics
    artifact_dir: Path
    run_id: str


@dataclass(frozen=True)
class ExperimentContext:
    """Data and provenance shared by all runs."""

    development: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    raw_sha256: str
    development_sha256: str
    code_revision: str
    transformer_source_sha256: str


def features_for_run(cfg: Settings, run: RunConfig) -> FeaturesConfig:
    """Feature config of a run (only the smoking flag varies, decision P1)."""
    smoking = cfg.features.smoking.model_copy(
        update={"reproduce_string_int_comparison_bug": run.smoking_bug_reproduced}
    )
    return cfg.features.model_copy(update={"smoking": smoking})


def build_context(cfg: Settings, *, materialize: bool) -> ExperimentContext:
    """Load the development frame, split it and compute provenance hashes."""
    development = load_development_frame(cfg, materialize=materialize)
    train, test = split_holdout(development, cfg.split, cfg.project.seed)
    return ExperimentContext(
        development=development,
        train=train,
        test=test,
        raw_sha256=cfg.data.raw_sha256,
        development_sha256=frame_sha256(development),
        code_revision=git_revision(),
        transformer_source_sha256=source_sha256(
            _FEATURES_DIR / "transformer.py", _FEATURES_DIR / "derivations.py"
        ),
    )


def _persist_artifacts(
    directory: Path,
    transformer: HypertensionTransformer,
    estimator: ClassifierMixin,
    cv: CrossValidationResult,
    holdout: ClassificationMetrics,
    roc: dict[str, Any],
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline([("transformer", transformer), ("model", estimator)])
    files = {
        "transformer.joblib": transformer,
        "knn_imputers.joblib": transformer.imputation_.imputers,
        "imputation_base_encoder.joblib": transformer.imputation_.base_encoder,
        "onehot_encoder.joblib": transformer.encoder_,
        "model.joblib": estimator,
        "pipeline.joblib": pipeline,
    }
    written: list[Path] = []
    for filename, obj in files.items():
        path = directory / filename
        joblib.dump(obj, path)
        written.append(path)
    metrics_path = directory / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "cv": {"summary": cv.summary, "folds": [asdict(m) for m in cv.fold_metrics]},
                "holdout": asdict(holdout),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    written.append(metrics_path)
    roc_path = directory / "roc.json"
    roc_path.write_text(json.dumps(roc), encoding="utf-8")
    written.append(roc_path)
    return written


def execute_run(
    cfg: Settings, run: RunConfig, context: ExperimentContext, tracker: ExperimentTracker
) -> RunResult:
    """Train, cross-validate, evaluate on the holdout, persist and track one run."""
    if not library_available(run.algorithm):
        raise ModelNotFoundError(f"{run.name}: library for {run.algorithm} is not installed")
    mapping = get_mapping()
    features = features_for_run(cfg, run)
    threshold = cfg.model.decision_threshold
    y_train = context.train[TARGET_COLUMN]

    transformer = HypertensionTransformer(features, mapping, mode=run.transformer_mode)
    estimator = build_estimator(run.algorithm, run.params, cfg.project.seed, y_train)

    tracked = tracker.start_run(
        run.name,
        tags={
            "role": run.role,
            "algorithm": run.algorithm,
            "transformer_mode": run.transformer_mode,
            "smoking_bug_reproduced": str(run.smoking_bug_reproduced),
            "code_revision": context.code_revision,
            "package_version": __version__,
        },
    )
    try:
        tracked.log_params(
            {
                **{f"model__{k}": v for k, v in run.params.items()},
                "seed": cfg.project.seed,
                "n_splits": cfg.split.n_splits,
                "test_size": cfg.split.test_size,
                "threshold": threshold,
                "knn_neighbors": features.laboratory.knn_imputer_neighbors,
                "transformer_mode": run.transformer_mode,
                "smoking_bug_reproduced": run.smoking_bug_reproduced,
                "raw_dataset_sha256": context.raw_sha256,
                "development_frame_sha256": context.development_sha256,
                "transformer_source_sha256": context.transformer_source_sha256,
                "n_development": len(context.development),
                "n_train": len(context.train),
                "n_test": len(context.test),
            }
        )

        cv = cross_validate(
            context.development, transformer, estimator, cfg.split, cfg.project.seed, threshold
        )

        x_train = context.train.drop(columns=[TARGET_COLUMN])
        x_test = context.test.drop(columns=[TARGET_COLUMN])
        fitted_transformer = HypertensionTransformer(
            features, mapping, mode=run.transformer_mode
        ).fit(x_train)
        fitted_estimator = build_estimator(run.algorithm, run.params, cfg.project.seed, y_train)
        encoded_train = fitted_transformer.transform(x_train)
        fitted_estimator.fit(encoded_train, y_train)
        encoded_test = fitted_transformer.transform(x_test)
        test_prob = fitted_estimator.predict_proba(encoded_test)[:, 1]
        holdout = compute_metrics(
            context.test[TARGET_COLUMN],
            test_prob,
            threshold,
            y_pred=fitted_estimator.predict(encoded_test),
        )

        roc = {
            "holdout": asdict(roc_points(context.test[TARGET_COLUMN], test_prob)),
            "oof": asdict(roc_points(context.development[TARGET_COLUMN], cv.oof_probabilities)),
        }
        artifact_dir = resolve(cfg.paths.artifacts_dir) / run.name
        files = _persist_artifacts(
            artifact_dir, fitted_transformer, fitted_estimator, cv, holdout, roc
        )
        # Drift reference: the training window summarised next to the model (etapa 6).
        reference = build_reference(x_train, mapping, cfg.monitoring.drift)
        files.append(save_reference(reference, artifact_dir / cfg.monitoring.reference_file))
        # Global reading of the model (etapa 6, pagina Modelo da interface). Only linear
        # estimators have coefficients; the others simply have no interpretation file.
        interpretation = build_interpretation(
            fitted_estimator,
            encoded_train,
            encoded_test,
            mapping,
            top_dummies=cfg.model.interpretation.top_dummies,
            max_rows=cfg.model.interpretation.max_contribution_rows,
        )
        if interpretation is not None:
            path = artifact_dir / INTERPRETATION_FILENAME
            path.write_text(json.dumps(interpretation), encoding="utf-8")
            files.append(path)

        tracked.log_metrics({f"cv_{k}": v for k, v in cv.summary.items()})
        tracked.log_metrics({f"holdout_{k}": float(v) for k, v in holdout.as_dict().items()})
        for file in files:
            tracked.log_artifact(file)
        tracked.log_dict({"folds": [asdict(m) for m in cv.fold_metrics]}, "cv_folds.json")
        tracker.end_run(tracked, "FINISHED")
    except Exception:
        tracker.end_run(tracked, "FAILED")
        raise

    logger.info(
        "run finished",
        extra={
            "run": run.name,
            "cv_auc": cv.summary["auc_roc_mean"],
            "cv_sensitivity": cv.summary["sensitivity_mean"],
            "holdout_auc": holdout.auc_roc,
        },
    )
    return RunResult(
        name=run.name,
        role=run.role,
        cv=cv,
        holdout=holdout,
        artifact_dir=artifact_dir,
        run_id=tracked.run_id,
    )


# ------------------------------------------------------------- evidence
def write_golden_metrics(cfg: Settings, result: RunResult, context: ExperimentContext) -> Path:
    """Freeze the fidelity run's metrics (decision P4) without rounding."""
    import numpy  # noqa: PLC0415  (only needed to record the environment)
    import sklearn  # noqa: PLC0415

    payload = {
        "run": result.name,
        "protocol": "StratifiedKFold over the full development set, transformer in fidelity mode",
        "n_splits": cfg.split.n_splits,
        "seed": cfg.project.seed,
        "threshold": cfg.model.decision_threshold,
        "raw_dataset_sha256": context.raw_sha256,
        "development_frame_sha256": context.development_sha256,
        "code_revision": context.code_revision,
        "environment": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "numpy": numpy.__version__,
            "pandas": pd.__version__,
        },
        "cv": result.cv.summary,
        "cv_folds": [asdict(m) for m in result.cv.fold_metrics],
        "holdout": asdict(result.holdout),
    }
    path = cfg.paths.absolute("golden_metrics")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def write_subgroup_analysis(
    cfg: Settings, result: RunResult, context: ExperimentContext
) -> tuple[Path, pd.DataFrame]:
    """Out-of-fold metrics by sex and age group for the promoted run."""
    dev = context.development
    y = dev[TARGET_COLUMN].to_numpy()
    age_groups = pd.cut(
        dev["age"], bins=cfg.features.age.bin_edges, labels=cfg.features.age.labels, right=False
    ).astype(str)
    sex = dev["sex"].map({1: "masculino", 2: "feminino"})
    frames = []
    for name, groups in (("sex", sex), ("age_group", age_groups)):
        table = subgroup_metrics(
            y,
            result.cv.oof_probabilities,
            groups,
            cfg.model.decision_threshold,
            min_size=cfg.acceptance.subgroup_min_size,
        )
        table.insert(0, "dimension", name)
        frames.append(table)
    table = pd.concat(frames, ignore_index=True)
    path = cfg.paths.absolute("evidence_dir") / f"subgroup_metrics_{result.name}.csv"
    table.to_csv(path, index=False)
    # Same table next to the model, so the API can serve it without reading evidencias/.
    (result.artifact_dir / SUBGROUP_FILENAME).write_text(
        table.to_json(orient="records"), encoding="utf-8"
    )
    return path, table


def write_rule_baseline(cfg: Settings, context: ExperimentContext) -> tuple[Path, dict[str, Any]]:
    """Evaluate the clinical rule on the same splits and write the evidence file.

    The rule is the floor: it uses no training data, so its holdout metrics say what a
    clinician gets for free. A trained model that does not clearly beat it does not
    justify the pipeline that serves it.

    Returns
    -------
    tuple
        Path of the Markdown evidence and the metrics as a dictionary.
    """
    mapping = get_mapping()
    threshold = cfg.model.decision_threshold
    rule = cfg.model.rule_baseline
    sets: dict[str, dict[str, Any]] = {}
    for name, frame in (("development", context.development), ("test", context.test)):
        features = frame.drop(columns=[TARGET_COLUMN])
        transformer = HypertensionTransformer(cfg.features, mapping, mode="fidelity").fit(features)
        model_frame = transformer.transform_detailed(features).model_frame
        applied = apply_rule(model_frame, rule)
        metrics = compute_metrics(
            frame[TARGET_COLUMN], applied.scores, threshold, y_pred=applied.labels
        )
        sets[name] = {
            **asdict(metrics),
            "factor_distribution": {
                str(k): int(v) for k, v in applied.factor_counts.value_counts().sort_index().items()
            },
        }

    payload = {
        "positive_from": rule.positive_from,
        "factors": [
            {"variable": f.variable, "label_pt": f.label_pt, "categories": list(f.categories)}
            for f in rule.factors
        ],
        "sets": sets,
    }
    evidence = cfg.paths.absolute("evidence_dir")
    (evidence / "rule_baseline.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    holdout = sets["test"]
    lines = [
        "# Baseline por regra simples",
        "",
        "Piso clinico sem aprendizado: contagem de fatores de risco classicos. O escore e a",
        "fracao de fatores presentes, o que permite calcular AUC; o rotulo vem do corte",
        "declarado. Avaliado no mesmo holdout e com a mesma funcao de metricas dos modelos.",
        "",
        describe(rule),
        "",
        "| Metrica | Desenvolvimento | Holdout |",
        "|---|---|---|",
    ]
    for key, label in (
        ("auc_roc", "AUC-ROC"),
        ("sensitivity", "Sensibilidade"),
        ("specificity", "Especificidade"),
        ("precision", "Precisão"),
        ("f1", "F1"),
        ("accuracy", "Acurácia"),
    ):
        lines.append(f"| {label} | {sets['development'][key]:.4f} | {sets['test'][key]:.4f} |")
    lines += [
        "",
        f"Distribuicao de fatores no holdout: {sets['test']['factor_distribution']}",
        "",
        f"Holdout: AUC {holdout['auc_roc']:.4f}, sensibilidade {holdout['sensitivity']:.4f}.",
        "O campeao precisa superar esses numeros com folga para justificar o pipeline.",
        "",
    ]
    path = evidence / "rule_baseline.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(
        "rule baseline evaluated",
        extra={"holdout_auc": holdout["auc_roc"], "holdout_sensitivity": holdout["sensitivity"]},
    )
    return path, payload


def write_experiment_table(
    cfg: Settings, results: list[RunResult], skipped: dict[str, str]
) -> Path:
    """Markdown and JSON table of every run for the evidence folder."""
    rows = []
    for r in results:
        rows.append(
            {
                "run": r.name,
                "role": r.role,
                "run_id": r.run_id,
                "cv_auc_mean": r.cv.summary["auc_roc_mean"],
                "cv_auc_std": r.cv.summary["auc_roc_std"],
                "cv_sensitivity_mean": r.cv.summary["sensitivity_mean"],
                "cv_sensitivity_std": r.cv.summary["sensitivity_std"],
                "cv_specificity_mean": r.cv.summary["specificity_mean"],
                "cv_precision_mean": r.cv.summary["precision_mean"],
                "cv_f1_mean": r.cv.summary["f1_mean"],
                "holdout_auc": r.holdout.auc_roc,
                "holdout_sensitivity": r.holdout.sensitivity,
                "holdout_specificity": r.holdout.specificity,
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("cv_auc_mean", ascending=False)
    evidence = cfg.paths.absolute("evidence_dir")
    evidence.mkdir(parents=True, exist_ok=True)
    table.to_json(evidence / "experiments.json", orient="records", indent=1)
    lines = [
        "| Run | Papel | AUC CV (dp) | Sens. CV (dp) | Espec. CV | Prec. CV | F1 CV "
        "| AUC holdout | Sens. holdout |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['run']} | {r['role']} | {r['cv_auc_mean']:.4f} ({r['cv_auc_std']:.4f}) | "
            f"{r['cv_sensitivity_mean']:.4f} ({r['cv_sensitivity_std']:.4f}) | "
            f"{r['cv_specificity_mean']:.4f} | {r['cv_precision_mean']:.4f} | "
            f"{r['cv_f1_mean']:.4f} | "
            f"{r['holdout_auc']:.4f} | {r['holdout_sensitivity']:.4f} |"
        )
    if skipped:
        lines.append("")
        lines.append("Runs não executados neste ambiente:")
        lines.append("")
        for name, reason in skipped.items():
            lines.append(f"- `{name}`: {reason}")
    path = evidence / "experiments.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_experiments(
    cfg: Settings,
    *,
    only: list[str] | None,
    backend: str | None,
    materialize: bool,
    freeze_golden: bool,
    promote: bool,
) -> tuple[list[RunResult], dict[str, str]]:
    """Execute the configured runs and write all evidence files."""
    tracker = make_tracker(cfg.tracking, backend)
    context = build_context(cfg, materialize=materialize)
    selected = [r for r in cfg.model.runs if only is None or r.name in only]
    results: list[RunResult] = []
    skipped: dict[str, str] = {}
    for run in selected:
        try:
            results.append(execute_run(cfg, run, context, tracker))
        except ModelNotFoundError as error:
            skipped[run.name] = str(error)
            logger.warning("run skipped", extra={"run": run.name, "reason": str(error)})

    by_name = {r.name: r for r in results}
    fidelity = next((r for r in results if r.role == "fidelity"), None)
    golden_path = cfg.paths.absolute("golden_metrics")
    if fidelity is not None and (freeze_golden or not golden_path.is_file()):
        write_golden_metrics(cfg, fidelity, context)
        logger.info("golden metrics written", extra={"path": str(golden_path)})

    promoted = by_name.get(cfg.tracking.promote_run)
    if promoted is not None:
        subgroup_path, _ = write_subgroup_analysis(cfg, promoted, context)
        plot_roc(
            {r.name: (_holdout_roc(r), r.holdout.auc_roc) for r in results},
            cfg.paths.absolute("evidence_dir") / "roc_holdout.png",
            "Curva ROC no conjunto de teste (20 %)",
        )
        h = promoted.holdout
        plot_confusion_matrix(
            h.true_negatives,
            h.false_positives,
            h.false_negatives,
            h.true_positives,
            cfg.paths.absolute("evidence_dir") / f"confusion_matrix_{promoted.name}.png",
            f"Matriz de confusão, holdout, {promoted.name}",
        )
        logger.info("subgroup analysis written", extra={"path": str(subgroup_path)})
        if promote:
            tracked = tracker.start_run(f"{promoted.name}_registration", {"role": "registration"})
            version = tracker.register_model(
                tracked, promoted.artifact_dir, cfg.tracking.registered_model_name
            )
            tracker.end_run(tracked)
            tracker.promote(
                cfg.tracking.registered_model_name,
                version,
                cfg.tracking.promote_stage,
                cfg.tracking.promote_alias,
            )
            registration = {
                "name": cfg.tracking.registered_model_name,
                "version": version,
                "stage": cfg.tracking.promote_stage,
                "alias": cfg.tracking.promote_alias,
                "run": promoted.name,
                "run_id": promoted.run_id,
                "code_revision": context.code_revision,
                "raw_dataset_sha256": context.raw_sha256,
            }
            (promoted.artifact_dir / "registration.json").write_text(
                json.dumps(registration, indent=1), encoding="utf-8"
            )
            logger.info(
                "model promoted",
                extra={
                    "model": cfg.tracking.registered_model_name,
                    "version": version,
                    "stage": cfg.tracking.promote_stage,
                },
            )
    write_experiment_table(cfg, results, skipped)
    write_rule_baseline(cfg, context)
    if fidelity is not None and promoted is not None:
        write_fidelity_comparison(cfg, fidelity, promoted)
    return results, skipped


def write_fidelity_comparison(cfg: Settings, fidelity: RunResult, production: RunResult) -> Path:
    """Side-by-side table of the fidelity and production runs (decision P2)."""
    keys = [
        ("auc_roc_mean", "AUC-ROC CV (média)"),
        ("auc_roc_std", "AUC-ROC CV (dp)"),
        ("sensitivity_mean", "Sensibilidade CV (média)"),
        ("sensitivity_std", "Sensibilidade CV (dp)"),
        ("specificity_mean", "Especificidade CV (média)"),
        ("precision_mean", "Precisão CV (média)"),
        ("f1_mean", "F1 CV (média)"),
    ]
    tol = cfg.fidelity
    lines = [
        f"| Metrica | {fidelity.name} | {production.name} | Diferenca | Tolerancia |",
        "|---|---|---|---|---|",
    ]
    for key, label in keys:
        a, b = fidelity.cv.summary[key], production.cv.summary[key]
        tolerance = ""
        if key == "auc_roc_mean":
            tolerance = f"{tol.auc_tolerance}"
        elif key == "sensitivity_mean":
            tolerance = f"{tol.sensitivity_tolerance}"
        lines.append(f"| {label} | {a:.6f} | {b:.6f} | {b - a:+.6f} | {tolerance} |")
    for key, label in (("auc_roc", "AUC-ROC holdout"), ("sensitivity", "Sensibilidade holdout")):
        a, b = getattr(fidelity.holdout, key), getattr(production.holdout, key)
        lines.append(f"| {label} | {a:.6f} | {b:.6f} | {b - a:+.6f} | |")
    path = cfg.paths.absolute("evidence_dir") / "fidelity_vs_production.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _holdout_roc(result: RunResult) -> RocPoints:
    """ROC points of the holdout evaluation, read back from the persisted ``roc.json``."""
    payload = json.loads((result.artifact_dir / "roc.json").read_text(encoding="utf-8"))
    return RocPoints(**payload["holdout"])


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and track the configured runs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="run every configured run")
    group.add_argument("--run", action="append", help="run only this run (repeatable)")
    parser.add_argument("--backend", choices=["mlflow", "json"], default=None)
    parser.add_argument("--no-materialize", action="store_true", help="read the workbook directly")
    parser.add_argument("--freeze-golden", action="store_true", help="overwrite golden metrics")
    parser.add_argument("--no-promote", action="store_true", help="skip registry promotion")
    args = parser.parse_args(argv)

    cfg = get_config()
    configure_logging(cfg.logging)
    results, skipped = run_experiments(
        cfg,
        only=None if args.all else args.run,
        backend=args.backend,
        materialize=not args.no_materialize,
        freeze_golden=args.freeze_golden,
        promote=not args.no_promote,
    )
    logger.info("experiments finished", extra={"executed": len(results), "skipped": list(skipped)})
    return 0 if results else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())

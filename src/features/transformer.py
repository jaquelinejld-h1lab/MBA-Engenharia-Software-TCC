"""scikit-learn compatible transformer: raw record to one-hot design matrix.

``HypertensionTransformer`` chains three steps:

1. stateless derivations (``derive_stateless``);
2. KNN imputation of the three laboratory exams, using one-hot encoded
   categorical bases exactly as ``impute_knn`` in the original notebook;
3. categorization of the imputed exams and one-hot encoding of the 33 model
   variables with explicit categories and ``drop="first"``.

Two modes (decision P2):

* ``"fidelity"``: the imputers are re-fitted on every frame passed to
  ``transform``, reproducing the notebook, which fitted the imputer on the
  frame it was transforming (train, test and development set separately).
* ``"production"``: the imputers are fitted once in ``fit`` (on the training
  frame) and reused, so a single observation can be transformed and no
  information flows from evaluation data into the transformer.

The object is a plain estimator (all state in ``*_`` attributes), so it can be
cloned, pickled with ``joblib`` and logged to MLflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

from src.config import FeaturesConfig
from src.data.mapping import VariableMapping
from src.exceptions import TransformError
from src.features.derivations import derive_laboratory_classes, derive_stateless

LAB_COLUMNS: tuple[str, ...] = ("egfr_afro", "cholesterol", "glucose")

TransformerMode = Literal["fidelity", "production"]


@dataclass(frozen=True)
class _FittedImputation:
    """Fitted encoder for the categorical base plus one KNN imputer per exam."""

    base_encoder: OneHotEncoder
    imputers: dict[str, KNNImputer]


@dataclass(frozen=True)
class TransformResult:
    """Outputs of a transformation, kept together for the API and the tests."""

    encoded: pd.DataFrame
    model_frame: pd.DataFrame
    imputed: pd.DataFrame

    def imputed_fields(self, row: int = 0) -> list[str]:
        """Names of the laboratory exams that were imputed for ``row``."""
        mask = self.imputed.iloc[row]
        return [str(c) for c in self.imputed.columns if bool(mask[c])]


class HypertensionTransformer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Raw inference variables (English names) to the 85-column design matrix.

    Parameters
    ----------
    features : FeaturesConfig
        Cut points, KNN ``k`` and imputation base list from the config.
    mapping : VariableMapping
        Model variable names and explicit category lists.
    mode : {"fidelity", "production"}
        See module docstring.
    """

    def __init__(
        self,
        features: FeaturesConfig,
        mapping: VariableMapping,
        mode: TransformerMode = "production",
    ) -> None:
        self.features = features
        self.mapping = mapping
        self.mode = mode

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> HypertensionTransformer:
        """Fit the imputation state and the one-hot encoder on ``X``.

        Parameters
        ----------
        X : pandas.DataFrame
            Raw inference variables in English names.
        y : ignored
            Present for API compatibility.
        """
        self._check_mode()
        derived = derive_stateless(X, self.features)
        self.imputation_ = self._fit_imputation(derived)
        model_frame, _ = self._complete(derived, self.imputation_)
        self.encoder_ = OneHotEncoder(
            categories=[self.mapping.model_categories[c] for c in self.mapping.model_en_names],
            drop="first" if self.features.encoding.drop_first else None,
            sparse_output=False,
            handle_unknown="error",
            dtype=np.int64,
        )
        self.encoder_.fit(model_frame[self.mapping.model_en_names])
        self.feature_names_out_ = list(self.mapping.dummy_names)
        self.n_features_in_ = int(X.shape[1])
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the encoded design matrix (int 0/1) with the input index."""
        return self.transform_detailed(X).encoded

    def transform_detailed(self, X: pd.DataFrame) -> TransformResult:
        """Transform and also return the 33-variable frame and the imputation mask."""
        self._check_fitted()
        derived = derive_stateless(X, self.features)
        imputation = self._fit_imputation(derived) if self.mode == "fidelity" else self.imputation_
        model_frame, imputed = self._complete(derived, imputation)
        try:
            matrix = self.encoder_.transform(model_frame[self.mapping.model_en_names])
        except ValueError as error:
            raise TransformError(f"category outside the declared mapping: {error}") from error
        encoded = pd.DataFrame(matrix, columns=self.feature_names_out_, index=X.index)
        return TransformResult(encoded=encoded, model_frame=model_frame, imputed=imputed)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Return the dummy names in output order."""
        self._check_fitted()
        return np.asarray(self.feature_names_out_, dtype=object)

    # ------------------------------------------------------------- internals
    def _check_mode(self) -> None:
        if self.mode not in ("fidelity", "production"):
            raise TransformError(f"unknown transformer mode '{self.mode}'")

    def _check_fitted(self) -> None:
        if not hasattr(self, "encoder_"):
            raise TransformError("HypertensionTransformer is not fitted; call fit first")

    def _categories_of(self, column: str) -> list[str]:
        """Explicit category list of a model or intermediate variable."""
        if column in self.mapping.model_categories:
            return self.mapping.model_categories[column]
        for intermediate in self.mapping.intermediate_variables:
            if intermediate.en == column:
                return list(intermediate.categories)
        raise TransformError(f"no category list declared for imputation base '{column}'")

    def _base_matrix(self, derived: pd.DataFrame, encoder: OneHotEncoder) -> np.ndarray:
        base = derived[self.features.laboratory.imputation_base_variables]
        matrix: np.ndarray = encoder.transform(base)
        return matrix

    def _fit_imputation(self, derived: pd.DataFrame) -> _FittedImputation:
        base_columns = self.features.laboratory.imputation_base_variables
        missing = [c for c in base_columns if c not in derived.columns]
        if missing:
            raise TransformError(f"imputation base columns not derived: {missing}")
        # Same construction as the original impute_knn: drop='first' one-hot of the
        # categorical base, fitted on the frame being imputed, then KNNImputer(k).
        # Categories are explicit (from the mapping) so that a single new observation
        # with a category absent from the training frame is still encodable; an absent
        # category yields an all-zero column, which does not change any KNN distance.
        base_encoder = OneHotEncoder(
            categories=[self._categories_of(c) for c in base_columns],
            drop="first",
            sparse_output=False,
            handle_unknown="error",
        )
        base_encoder.fit(derived[base_columns])
        base = self._base_matrix(derived, base_encoder)
        imputers: dict[str, KNNImputer] = {}
        for column in LAB_COLUMNS:
            numeric = derived[[column]].to_numpy(dtype=float)
            imputer = KNNImputer(n_neighbors=self.features.laboratory.knn_imputer_neighbors)
            imputer.fit(np.concatenate([numeric, base], axis=1))
            imputers[column] = imputer
        return _FittedImputation(base_encoder=base_encoder, imputers=imputers)

    def _complete(
        self, derived: pd.DataFrame, imputation: _FittedImputation
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Impute the exams, categorize them and assemble the 33-variable frame."""
        try:
            base = self._base_matrix(derived, imputation.base_encoder)
        except ValueError as error:
            raise TransformError(f"imputation base outside fitted categories: {error}") from error
        completed = derived.copy()
        imputed = pd.DataFrame(index=derived.index)
        for column in LAB_COLUMNS:
            numeric = derived[[column]].to_numpy(dtype=float)
            imputed[column] = np.isnan(numeric[:, 0])
            filled = imputation.imputers[column].transform(np.concatenate([numeric, base], axis=1))
            completed[column] = filled[:, 0]
        labs = derive_laboratory_classes(completed, self.features)
        model_frame = pd.concat([completed, labs], axis=1)[self.mapping.model_en_names]
        return model_frame, imputed

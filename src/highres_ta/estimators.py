from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from joblib import Parallel, delayed
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_is_fitted


class CatBoostResidualRegressor(BaseEstimator, RegressorMixin):
    """Polynomial linear baseline + CatBoost residual model.

    Compatible with sklearn's BaggingRegressor, Pipeline, GridSearchCV, and joblib.
    """

    def __init__(
        self,
        linear_features: Sequence[str] | None = ("salinity",),
        feature_names: Sequence[str] | None = None,
        polynomial_degree: int = 2,
        include_bias: bool = True,
        iterations: int = 300,
        loss_function: str = "MAE",
        random_strength: float = 1.5,
        min_data_in_leaf: int = 50,
        catboost_verbose: bool | int = False,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.linear_features = linear_features
        self.feature_names = feature_names
        self.polynomial_degree = polynomial_degree
        self.include_bias = include_bias
        self.iterations = iterations
        self.loss_function = loss_function
        self.random_strength = random_strength
        self.min_data_in_leaf = min_data_in_leaf
        self.catboost_verbose = catboost_verbose
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X_df = self._to_frame(X)
        y = np.asarray(y)

        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        self.n_features_in_ = X_df.shape[1]
        self.linear_features_ = self._resolve_linear_features(X_df)

        self.linear_model_ = make_pipeline(
            PolynomialFeatures(degree=self.polynomial_degree, include_bias=self.include_bias),
            LinearRegression(),
        )
        self.linear_model_.fit(X_df[self.linear_features_], y)

        self.boosting_model_ = CatBoostRegressor(
            iterations=self.iterations,
            loss_function=self.loss_function,
            random_strength=self.random_strength,
            verbose=self.catboost_verbose,
            min_data_in_leaf=self.min_data_in_leaf,
            subsample=0.8,
            random_state=self.random_state,
            thread_count=self.n_jobs,
        )
        self.boosting_model_.fit(X_df, y - self.linear_model_.predict(X_df[self.linear_features_]))
        return self

    def predict(self, X):
        check_is_fitted(self, attributes=["linear_model_", "boosting_model_", "linear_features_"])
        X_df = self._validate_feature_frame(X)
        return self.linear_model_.predict(
            X_df[self.linear_features_]
        ) + self.boosting_model_.predict(X_df)

    def save(self, file_path: str | os.PathLike[str], compress: int = 3) -> None:
        """Save estimator to disk with joblib."""
        joblib_dump(self, file_path, compress=compress)

    @classmethod
    def load(cls, file_path: str | os.PathLike[str]) -> "CatBoostResidualRegressor":
        """Load estimator from disk."""
        model = joblib_load(file_path)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}.")
        return model

    # ------------------------------------------------------------------ #
    # joblib / pickle: use CatBoost's native .cbm format for the          #
    # boosting model, since pickle alone isn't reliable for CatBoost.     #
    # ------------------------------------------------------------------ #

    def __getstate__(self):
        state = self.__dict__.copy()
        if "boosting_model_" in state:
            with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as f:
                tmp = f.name
            try:
                state["boosting_model_"].save_model(tmp, format="cbm")
                with open(tmp, "rb") as f:
                    state["boosting_model_"] = f.read()
            finally:
                os.unlink(tmp)
            state["_boosting_serialized"] = True
        return state

    def __setstate__(self, state):
        if state.pop("_boosting_serialized", False):
            with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as f:
                f.write(state["boosting_model_"])
                tmp = f.name
            try:
                model = CatBoostRegressor()
                model.load_model(tmp, format="cbm")
                state["boosting_model_"] = model
            finally:
                os.unlink(tmp)
        self.__dict__.update(state)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _resolve_linear_features(self, X: pd.DataFrame) -> list[str]:
        features = list(X.columns) if self.linear_features is None else list(self.linear_features)
        missing = sorted(set(features) - set(X.columns))
        if missing:
            raise ValueError(f"Missing linear feature columns: {', '.join(missing)}")
        return features

    def _validate_feature_frame(self, X) -> pd.DataFrame:
        X_df = self._to_frame(X)
        missing = [c for c in self.feature_names_in_ if c not in X_df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {', '.join(missing)}")
        return X_df.loc[:, self.feature_names_in_]

    def _to_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        X = np.asarray(X)
        if X.ndim != 2:
            raise TypeError("X must be 2-D.")
        if self.feature_names is not None:
            cols = list(self.feature_names)
        elif hasattr(self, "feature_names_in_") and len(self.feature_names_in_) == X.shape[1]:
            cols = list(self.feature_names_in_)
        else:
            cols = [f"x{i}" for i in range(X.shape[1])]
        if len(cols) != X.shape[1]:
            raise ValueError("feature_names length must match the number of columns in X.")
        return pd.DataFrame(X, columns=cols)

    def _ensure_dataframe(self, X) -> pd.DataFrame:
        return self._to_frame(X)


class BaggingCatBoostResidualRegressor(BaggingRegressor):
    """BaggingRegressor with per-component prediction and uncertainty estimates.

    Extends sklearn's BaggingRegressor to expose the linear-baseline and
    CatBoost-residual components of each CatBoostResidualRegressor in the
    ensemble, with optional cross-estimator standard deviation.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_samples: float = 0.33,
        oob_score: bool = False,
        n_jobs: int | None = None,
        random_state: int | None = None,
        **residual_kwargs,
    ) -> None:
        self._residual_kwargs = residual_kwargs
        super().__init__(
            estimator=CatBoostResidualRegressor(**residual_kwargs),
            n_estimators=n_estimators,
            max_samples=max_samples,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, X, y, sample_weight=None):
        estimator = getattr(self, "estimator", None)
        if (
            isinstance(X, pd.DataFrame)
            and isinstance(estimator, CatBoostResidualRegressor)
            and estimator.feature_names is None
        ):
            estimator.set_params(feature_names=list(X.columns))
        if sample_weight is None:
            return super().fit(X, y)
        return super().fit(X, y, sample_weight=sample_weight)

    def save(self, file_path: str | os.PathLike[str], compress: int = 3) -> None:
        """Save estimator to disk with joblib."""
        joblib_dump(self, file_path, compress=compress)

    @classmethod
    def load(cls, file_path: str | os.PathLike[str]) -> "BaggingCatBoostResidualRegressor":
        """Load estimator from disk."""
        model = joblib_load(file_path)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}.")
        return model

    def predict_with_uncertainty(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Predict target and return ensemble std as a rough uncertainty estimate."""
        lin, cb = self._component_predictions(X)
        mean, std = self._aggregate(lin + cb, return_std=True)
        return mean, std

    def predict_linear(
        self, X, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict only the linear-baseline component."""
        lin, _ = self._component_predictions(X)
        return self._aggregate(lin, return_std)

    def predict_catboost(
        self, X, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict only the CatBoost-residual component."""
        _, cb = self._component_predictions(X)
        return self._aggregate(cb, return_std)

    def predict_components(self, X) -> pd.DataFrame:
        """Predict all components in one pass.

        Returns a DataFrame with columns:
        ``full_avg``, ``full_std``, ``linear_avg``, ``linear_std``,
        ``boosted_avg``, ``boosted_std``.
        """
        lin, cb = self._component_predictions(X)
        total = lin + cb

        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(
            {
                "full_avg": total.mean(axis=0),
                "full_std": total.std(axis=0),
                "linear_avg": lin.mean(axis=0),
                "linear_std": lin.std(axis=0),
                "boosted_avg": cb.mean(axis=0),
                "boosted_std": cb.std(axis=0),
            },
            index=index,
        )

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _component_predictions(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Return stacked linear and CatBoost predictions across estimators.

        Returns
        -------
        linear_preds   : np.ndarray of shape (n_estimators, n_samples)
        catboost_preds : np.ndarray of shape (n_estimators, n_samples)
        """
        check_is_fitted(self)
        # Resolve the feature frame once, using any fitted estimator as a guide.
        X_df = self.estimators_[0]._validate_feature_frame(X)

        if self.n_jobs == 1 or len(self.estimators_) == 1:
            results = [self._predict_one_estimator(est, X_df) for est in self.estimators_]
        else:
            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._predict_one_estimator)(est, X_df) for est in self.estimators_
            )

        linear_preds, catboost_preds = zip(*results)
        return np.array(linear_preds), np.array(catboost_preds)

    @staticmethod
    def _predict_one_estimator(
        est: CatBoostResidualRegressor, X_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        linear_pred = est.linear_model_.predict(X_df[est.linear_features_])
        catboost_pred = est.boosting_model_.predict(X_df)
        return linear_pred, catboost_pred

    @staticmethod
    def _aggregate(
        preds: np.ndarray, return_std: bool
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        mean = preds.mean(axis=0)
        if return_std:
            std = preds.std(axis=0)
            return mean, std
        else:
            return mean


__all__ = ["CatBoostResidualRegressor", "BaggingCatBoostResidualRegressor"]

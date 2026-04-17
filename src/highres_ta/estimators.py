from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_is_fitted


class CatBoostResidualRegressor(BaseEstimator, RegressorMixin):
    """Linear baseline plus bagged CatBoost residual model.

    This estimator reproduces the modeling procedure used in
    `scripts/inference.ipynb`:

    1. Fit a polynomial linear regression on a subset of features.
    2. Compute residuals between the target and the linear baseline.
    3. Fit a `BaggingRegressor` whose base estimator is `CatBoostRegressor`
       on the full feature set to learn those residuals.
    4. Predict by summing the baseline and residual estimates.
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
        n_jobs: int | None = None,  # Ignored, for API compatibility with sklearn's BaggingRegressor
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

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        X_df = self._ensure_dataframe(X)
        y_array = np.asarray(y)

        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        self.n_features_in_ = X_df.shape[1]
        self.linear_features_ = self._resolve_linear_features(X_df)

        self.linear_model_ = make_pipeline(
            PolynomialFeatures(
                degree=self.polynomial_degree,
                include_bias=self.include_bias,
            ),
            LinearRegression(),
        )
        self.linear_model_.fit(X_df[self.linear_features_], y_array)

        baseline = self.linear_model_.predict(X_df[self.linear_features_])
        residuals = y_array - baseline

        self.boosting_model_ = CatBoostRegressor(
            iterations=self.iterations,
            loss_function=self.loss_function,
            random_strength=self.random_strength,
            verbose=self.catboost_verbose,
            min_data_in_leaf=self.min_data_in_leaf,
            subsample=1.0,
            random_state=self.random_state,
            thread_count=self.n_jobs,
        )
        self.boosting_model_.fit(X_df, residuals)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["linear_model_", "boosting_model_", "linear_features_"])
        X_df = self._validate_feature_frame(X)

        baseline = self.linear_model_.predict(X_df[self.linear_features_])
        residuals = self.boosting_model_.predict(X_df)
        return baseline + residuals

    def __getstate__(self):
        state = self.__dict__.copy()
        if "boosting_model_" in state:
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as f:
                tmp_path = f.name
            try:
                state["boosting_model_"].save_model(tmp_path, format="cbm")
                with open(tmp_path, "rb") as f:
                    state["boosting_model_"] = f.read()
            finally:
                os.unlink(tmp_path)
            state["_boosting_model_serialized"] = True
        return state

    def __setstate__(self, state):
        if state.pop("_boosting_model_serialized", False):
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as f:
                f.write(state["boosting_model_"])
                tmp_path = f.name
            try:
                model = CatBoostRegressor()
                model.load_model(tmp_path, format="cbm")
                state["boosting_model_"] = model
            finally:
                os.unlink(tmp_path)
        self.__dict__.update(state)

    def _resolve_linear_features(self, X: pd.DataFrame) -> list[str]:
        if self.linear_features is None:
            return list(X.columns)

        linear_features = list(self.linear_features)
        missing_columns = sorted(set(linear_features) - set(X.columns))
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Missing linear feature columns: {missing}")

        return linear_features

    def _validate_feature_frame(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        X_df = self._ensure_dataframe(X)
        missing_columns = [
            column for column in self.feature_names_in_ if column not in X_df.columns
        ]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Missing feature columns: {missing}")

        return X_df.loc[:, self.feature_names_in_]

    @staticmethod
    def _default_feature_names(n_features: int) -> list[str]:
        return [f"x{i}" for i in range(n_features)]

    def _ensure_dataframe(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X

        array = np.asarray(X)
        if array.ndim != 2:
            raise TypeError("X must be a 2D pandas DataFrame or numpy array.")

        if self.feature_names is not None:
            columns = list(self.feature_names)
        else:
            columns = self._default_feature_names(array.shape[1])

        if len(columns) != array.shape[1]:
            raise ValueError("feature_names length must match the number of columns in X.")

        return pd.DataFrame(array, columns=columns)


class BaggingCatBoostResidualRegressor(BaseEstimator, RegressorMixin):
    """Bagged ensemble of CatBoostResidualRegressor models.

    Trains ``n_estimators`` :class:`CatBoostResidualRegressor` instances on
    bootstrap samples drawn from the training data.  At prediction time the
    ensemble is queried and the results are aggregated.

    Each predict method accepts ``return_std=True`` to also return the
    cross-estimator standard deviation, which is a rough measure of epistemic
    uncertainty.  The linear-baseline and CatBoost-residual components can be
    retrieved independently via :meth:`predict_linear` and
    :meth:`predict_catboost`.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_samples: float = 1.0,
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
        n_jobs: int | None = None,  # Ignored, for API compatibility with sklearn's BaggingRegressor
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
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

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        # Use a helper from the base estimator just to resolve feature names.
        _proto = CatBoostResidualRegressor(feature_names=self.feature_names)
        X_df = _proto._ensure_dataframe(X)
        y_array = np.asarray(y)

        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        self.n_features_in_ = X_df.shape[1]

        n_samples = X_df.shape[0]
        boot_size = max(1, int(n_samples * self.max_samples))
        rng = np.random.RandomState(self.random_state)

        self.estimators_: list[CatBoostResidualRegressor] = []
        for _ in range(self.n_estimators):
            seed = int(rng.randint(0, 2**31))
            indices = rng.choice(n_samples, size=boot_size, replace=True)
            X_boot = X_df.iloc[indices]
            y_boot = y_array[indices]

            est = CatBoostResidualRegressor(
                linear_features=self.linear_features,
                feature_names=self.feature_names,
                polynomial_degree=self.polynomial_degree,
                include_bias=self.include_bias,
                iterations=self.iterations,
                loss_function=self.loss_function,
                random_strength=self.random_strength,
                min_data_in_leaf=self.min_data_in_leaf,
                catboost_verbose=self.catboost_verbose,
                random_state=seed,
            )
            est.fit(X_boot, y_boot)
            self.estimators_.append(est)

        return self

    # ------------------------------------------------------------------
    # Public predict methods
    # ------------------------------------------------------------------

    def predict(
        self, X: pd.DataFrame | np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict the target (linear + catboost residual).

        Parameters
        ----------
        X:
            Feature matrix.
        return_std:
            If ``True`` also return the ensemble standard deviation.

        Returns
        -------
        mean : np.ndarray of shape (n_samples,)
        std  : np.ndarray of shape (n_samples,)  — only when ``return_std=True``
        """
        lin, cb = self._predict_all_components(X)
        total = lin + cb
        return self._aggregate(total, return_std)

    def predict_linear(
        self, X: pd.DataFrame | np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict only the linear-baseline component.

        Parameters
        ----------
        X:
            Feature matrix.
        return_std:
            If ``True`` also return the ensemble standard deviation.
        """
        lin, _ = self._predict_all_components(X)
        return self._aggregate(lin, return_std)

    def predict_catboost(
        self, X: pd.DataFrame | np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict only the CatBoost-residual component.

        Parameters
        ----------
        X:
            Feature matrix.
        return_std:
            If ``True`` also return the ensemble standard deviation.
        """
        _, cb = self._predict_all_components(X)
        return self._aggregate(cb, return_std)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(
        preds: np.ndarray, return_std: bool
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Reduce (n_estimators, n_samples) → mean [, std]."""
        mean = preds.mean(axis=0)
        if return_std:
            return mean, preds.std(axis=0)
        return mean

    def _predict_all_components(
        self, X: pd.DataFrame | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return linear and catboost predictions for every estimator.

        Returns
        -------
        linear_preds  : np.ndarray of shape (n_estimators, n_samples)
        catboost_preds: np.ndarray of shape (n_estimators, n_samples)
        """
        check_is_fitted(self, attributes=["estimators_", "feature_names_in_"])
        X_df = self._validate_feature_frame(X)

        linear_preds = []
        catboost_preds = []
        for est in self.estimators_:
            lin = est.linear_model_.predict(X_df[est.linear_features_])
            cb = est.boosting_model_.predict(X_df)
            linear_preds.append(lin)
            catboost_preds.append(cb)

        return np.array(linear_preds), np.array(catboost_preds)

    def _validate_feature_frame(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        _proto = CatBoostResidualRegressor(feature_names=self.feature_names)
        X_df = _proto._ensure_dataframe(X)
        missing = [c for c in self.feature_names_in_ if c not in X_df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {', '.join(missing)}")
        return X_df.loc[:, self.feature_names_in_]


__all__ = ["CatBoostResidualRegressor", "BaggingCatBoostResidualRegressor"]

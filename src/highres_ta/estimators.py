from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from sklearn.base import BaseEstimator, RegressorMixin
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
        linear_features: Sequence[str] = ("salinity",),
        feature_names: Sequence[str] | None = None,
        polynomial_degree: int = 1,
        **catboost_kwargs: Any,
    ) -> None:
        self.linear_features = linear_features
        self.feature_names = feature_names
        self.polynomial_degree = polynomial_degree
        self.catboost_kwargs = catboost_kwargs
        self.loss_function = catboost_kwargs.get("loss_function")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Expose CatBoost keyword arguments to sklearn cloning and search."""
        return {
            "linear_features": self.linear_features,
            "feature_names": self.feature_names,
            "polynomial_degree": self.polynomial_degree,
            **self.catboost_kwargs,
        }

    def set_params(self, **params: Any) -> CatBoostResidualRegressor:
        """Set residual-estimator or CatBoost parameters."""
        residual_params = {"linear_features", "feature_names", "polynomial_degree"}
        for name in residual_params.intersection(params):
            setattr(self, name, params.pop(name))
        self.catboost_kwargs.update(params)
        self.loss_function = self.catboost_kwargs.get("loss_function")
        return self

    def fit(self, X, y, eval_set=None, sample_weight=None, **boost_kwargs):
        X_df = self._to_frame(X)
        y = np.asarray(y)

        # Recreate both components for every fit. Besides following sklearn's
        # fitted-attribute convention, this prevents state carrying over when
        # an estimator is fitted more than once during cross-validation.
        self.linear_model_ = self._make_linear_model()
        self.boosting_model_ = self._make_catboost_model()

        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        self.n_features_in_ = X_df.shape[1]
        self.quantiles_ = self._quantiles_from_loss(self.loss_function)

        self.linear_model_.fit(X_df, y)
        yhat_linear = self.linear_model_.predict(X_df)

        if eval_set is not None:
            is_correct_type = isinstance(eval_set, tuple) and len(eval_set) == 2
            if not is_correct_type:
                raise ValueError(
                    f"eval_set must be a tuple of (X_eval, y_eval), but got {type(eval_set)}"
                )
            else:
                X_eval, y_eval = eval_set
                X_eval_df = self._validate_feature_frame(X_eval)
                yhat_eval_linear = self.linear_model_.predict(X_eval_df)
                eval_set = (X_eval_df, y_eval - yhat_eval_linear)

        self.boosting_model_.fit(
            X_df,
            y - yhat_linear,
            eval_set=eval_set,
            sample_weight=sample_weight,
            **boost_kwargs,
        )

        return self

    def predict(self, X):
        check_is_fitted(self, attributes=["linear_model_", "boosting_model_"])
        X_df = self._validate_feature_frame(X)

        yhat_linear = self.linear_model_.predict(X_df)
        yhat_boosted = self.boosting_model_.predict(X_df)

        return self._pred_helper(yhat_linear, yhat_boosted)

    def predict_quantiles(self, X) -> np.ndarray:
        """Return absolute-target quantiles for a fitted MultiQuantile model."""
        check_is_fitted(
            self,
            attributes=["linear_model_", "boosting_model_", "quantiles_"],
        )
        if self.quantiles_ is None:
            raise ValueError(
                "predict_quantiles requires CatBoost loss_function='MultiQuantile:alpha=...'."
            )

        prediction = np.asarray(self.predict(X), dtype=float)
        expected_shape = (len(X), len(self.quantiles_))
        if prediction.shape != expected_shape:
            raise ValueError(
                f"Expected quantile predictions with shape {expected_shape}, "
                f"got {prediction.shape}."
            )
        return prediction

    def score(self, X, y, sample_weight=None, metric: str = "r2_score") -> float:
        from sklearn import metrics

        func = getattr(metrics, metric)
        y = np.array(y).squeeze()
        yhat = self.predict(X)
        if self.loss_function == "RMSEWithUncertainty":
            yhat = yhat[:, 0]
        elif getattr(self, "quantiles_", None) is not None:
            median_matches = np.flatnonzero(np.isclose(self.quantiles_, 0.5))
            if len(median_matches) != 1:
                raise ValueError("MultiQuantile scoring requires a 0.5 quantile.")
            yhat = yhat[:, median_matches[0]]

        return func(y, yhat)

    def save(self, file_path: str | os.PathLike[str], compress: int = 3) -> None:
        """Save estimator to disk with joblib."""
        joblib_dump(self, file_path, compress=compress)

    @classmethod
    def load(cls, file_path: str | os.PathLike[str]) -> CatBoostResidualRegressor:
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
        if "quantiles_" not in state:
            self.quantiles_ = self._quantiles_from_loss(
                getattr(self, "loss_function", None)
            )

    def _pred_helper(self, yhat_linear, yhat_boosted):
        yhat_linear = np.asarray(yhat_linear, dtype=float)
        yhat_boosted = np.asarray(yhat_boosted, dtype=float)

        if self.loss_function == "RMSEWithUncertainty":
            yhat_mu = yhat_boosted[:, 0]
            yhat_var = yhat_boosted[:, 1]
            return np.c_[yhat_linear + yhat_mu, yhat_var]
        if getattr(self, "quantiles_", None) is not None:
            if yhat_boosted.ndim != 2:
                raise ValueError(
                    "CatBoost MultiQuantile prediction must be a two-dimensional array."
                )
            return yhat_linear[:, None] + yhat_boosted
        return yhat_linear + yhat_boosted

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

    def _make_linear_model(self) -> LinearRegression:
        from sklearn.compose import make_column_selector, make_column_transformer
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        linear_vars_pattern = r"^(?:" + "|".join(map(re.escape, self.linear_features)) + r")$"

        return make_pipeline(
            make_column_transformer(
                ("passthrough", make_column_selector(pattern=linear_vars_pattern)),
                remainder="drop",
                verbose_feature_names_out=False,
            ),
            PolynomialFeatures(self.polynomial_degree),
            StandardScaler(),
            PCA(None),
            LinearRegression(),
        )

    def _make_catboost_model(self) -> CatBoostRegressor:
        kwargs = dict(self.catboost_kwargs)
        return CatBoostRegressor(**kwargs)

    @staticmethod
    def _quantiles_from_loss(loss_function: str | None) -> np.ndarray | None:
        if not isinstance(loss_function, str) or not loss_function.startswith(
            "MultiQuantile:"
        ):
            return None

        options = loss_function.partition(":")[2].split(";")
        alpha_option = next(
            (option for option in options if option.strip().startswith("alpha=")),
            None,
        )
        if alpha_option is None:
            raise ValueError("MultiQuantile loss_function must define alpha values.")

        quantiles = np.asarray(
            [float(value) for value in alpha_option.split("=", 1)[1].split(",")],
            dtype=float,
        )
        if (
            quantiles.ndim != 1
            or len(quantiles) == 0
            or np.any((quantiles <= 0) | (quantiles >= 1))
            or np.any(np.diff(quantiles) <= 0)
        ):
            raise ValueError("MultiQuantile alpha values must be unique and increasing in (0, 1).")
        return quantiles


__all__ = ["CatBoostResidualRegressor"]

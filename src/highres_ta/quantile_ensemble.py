from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Literal

import numpy as np
from joblib import dump as joblib_dump
from joblib import load as joblib_load

from .estimators import CatBoostResidualRegressor
from .quantiles import quantile_column, validate_quantiles

Aggregation = Literal["mean", "median"]


class QuantileRegressionEnsemble:
    """Inference container for independently tuned quantile regressors."""

    def __init__(
        self,
        estimators: Sequence[CatBoostResidualRegressor],
        seeds: Sequence[int],
        quantiles: Sequence[float],
        aggregation: Aggregation = "mean",
    ) -> None:
        self.estimators = list(estimators)
        self.seeds = list(seeds)
        self.quantiles = validate_quantiles(quantiles)
        self.aggregation = aggregation
        self._validate_members()

    def predict_members(self, X) -> np.ndarray:
        """Return predictions shaped (members, observations, quantiles)."""
        predictions = np.asarray(
            [estimator.predict_quantiles(X) for estimator in self.estimators],
            dtype=float,
        )
        expected_shape = (len(self.estimators), len(X), len(self.quantiles))
        if predictions.shape != expected_shape:
            raise ValueError(
                f"Expected member predictions with shape {expected_shape}, "
                f"got {predictions.shape}."
            )
        return predictions

    def predict_quantiles(self, X) -> np.ndarray:
        """Aggregate corresponding quantiles across ensemble members."""
        member_predictions = self.predict_members(X)
        if self.aggregation == "mean":
            return np.mean(member_predictions, axis=0)
        if self.aggregation == "median":
            return np.median(member_predictions, axis=0)
        raise ValueError(f"Unsupported aggregation: {self.aggregation!r}.")

    def predict(self, X) -> np.ndarray:
        """Return the aggregated median prediction."""
        prediction = self.predict_quantiles(X)
        return prediction[:, quantile_column(self.quantiles, 0.5)]

    def save(self, file_path: str | os.PathLike[str], compress: int = 3) -> None:
        """Save the complete ensemble, including its members, with joblib."""
        joblib_dump(self, file_path, compress=compress)

    @classmethod
    def load(cls, file_path: str | os.PathLike[str]) -> QuantileRegressionEnsemble:
        """Load and type-check an ensemble saved with :meth:`save`."""
        ensemble = joblib_load(file_path)
        if not isinstance(ensemble, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}.")
        return ensemble

    def _validate_members(self) -> None:
        if len(self.estimators) == 0:
            raise ValueError("At least one estimator is required.")
        if len(self.estimators) != len(self.seeds):
            raise ValueError("estimators and seeds must have the same length.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Member seeds must be unique.")
        if self.aggregation not in ("mean", "median"):
            raise ValueError("aggregation must be either 'mean' or 'median'.")

        for estimator in self.estimators:
            member_quantiles = getattr(estimator, "quantiles_", None)
            if member_quantiles is None or not np.array_equal(
                np.asarray(member_quantiles), self.quantiles
            ):
                raise ValueError("Every fitted member must use the ensemble quantile grid.")


__all__ = ["QuantileRegressionEnsemble"]

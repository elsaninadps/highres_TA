from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Literal

import joblib
import numpy as np
from joblib import dump as joblib_dump
from joblib import load as joblib_load

from .estimators import CatBoostResidualRegressor
from .quantiles import quantile_column, validate_quantiles

Aggregation = Literal["mean", "median"]


def decompose_variance(predictions: np.ndarray) -> dict[str, np.ndarray]:
    """Split ensemble variance into data-split and HPO terms.

    ``predictions`` is shaped (splits, replicates, observations, quantiles). By the
    law of total variance (population variance), ``split_var + hpo_var`` equals the
    variance across all members.
    """
    predictions = np.asarray(predictions)
    if predictions.ndim != 4:
        raise ValueError(
            "predictions must be shaped (splits, replicates, observations, quantiles); "
            f"got {predictions.ndim} dimensions."
        )
    return {
        "mean": predictions.mean(axis=(0, 1)),
        "split_var": predictions.mean(axis=1).var(axis=0),
        "hpo_var": predictions.var(axis=1).mean(axis=0),
    }


class QuantileRegressionEnsemble:
    """Inference container for independently tuned quantile regressors."""

    def __init__(
        self,
        estimators: Sequence[CatBoostResidualRegressor],
        seeds: Sequence[int],
        quantiles: Sequence[float],
        aggregation: Aggregation = "mean",
        member_splits: Sequence[int] | None = None,
        member_replicates: Sequence[int] | None = None,
    ) -> None:
        self.estimators = list(estimators)
        self.seeds = list(seeds)
        self.member_splits = None if member_splits is None else list(member_splits)
        self.member_replicates = None if member_replicates is None else list(member_replicates)
        self.quantiles = validate_quantiles(quantiles)
        self.aggregation = aggregation
        self._validate_members()

    def optimise_cpus(self, max_cpus: int) -> None:
        """
        1. get the number of available CPUs
        2. set number of cpus to min(max_cpus)
        3. get the number of estimators
        4. set the number of cpus for each estimator to max(1, floor(n_cpus / n_estimators))
        """
        n_cpus = min(os.cpu_count() or 1, max_cpus)
        n_estimators = len(self.estimators)
        self.cpus_per_estimator = max(1, n_cpus // n_estimators)
        for estimator in self.estimators:
            estimator.catboost_kwargs["thread_count"] = self.cpus_per_estimator

    def predict_members(self, X) -> np.ndarray:
        """Return predictions shaped (members, observations, quantiles)."""
        n_estimators = len(self.estimators)
        outputs = joblib.Parallel(n_jobs=n_estimators)(
            joblib.delayed(estimator.predict_quantiles)(X) for estimator in self.estimators
        )
        predictions = np.asarray(outputs, dtype=np.float32)
        expected_shape = (len(self.estimators), len(X), len(self.quantiles))
        if predictions.shape != expected_shape:
            raise ValueError(
                f"Expected member predictions with shape {expected_shape}, got {predictions.shape}."
            )
        return predictions

    def predict_decomposition(self, X) -> dict[str, np.ndarray]:
        """Return the mean prediction and its split and HPO variance components."""
        member_splits = getattr(self, "member_splits", None)
        member_replicates = getattr(self, "member_replicates", None)
        if member_splits is None or member_replicates is None:
            raise ValueError("member_splits and member_replicates are required to decompose.")
        splits = sorted(set(member_splits))
        replicates = sorted(set(member_replicates))
        index = {
            (split, replicate): position
            for position, (split, replicate) in enumerate(zip(member_splits, member_replicates))
        }
        if len(index) != len(member_splits):
            raise ValueError("Each (split, replicate) pair must appear exactly once.")
        if any((split, replicate) not in index for split in splits for replicate in replicates):
            raise ValueError("Unbalanced design: every split needs the same set of replicates.")
        member_predictions = self.predict_members(X)
        grouped = np.stack([
            np.stack([member_predictions[index[split, replicate]] for replicate in replicates])
            for split in splits
        ])
        return decompose_variance(grouped)

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
        member_splits = getattr(self, "member_splits", None)
        member_replicates = getattr(self, "member_replicates", None)
        if (member_splits is None) != (member_replicates is None):
            raise ValueError("member_splits and member_replicates must be given together.")
        for name, values in (
            ("member_splits", member_splits),
            ("member_replicates", member_replicates),
        ):
            if values is not None and len(values) != len(self.estimators):
                raise ValueError(f"{name} and estimators must have the same length.")
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


__all__ = ["QuantileRegressionEnsemble", "decompose_variance"]

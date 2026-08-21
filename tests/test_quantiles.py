import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from highres_ta import (
    CatBoostResidualRegressor,
    QuantileRegressionEnsemble,
    quantile_column,
    quantile_crossing_rate,
    score_quantile_predictions,
)


class FixedQuantileEstimator:
    def __init__(self, prediction, quantiles):
        self.prediction = np.asarray(prediction, dtype=float)
        self.quantiles_ = np.asarray(quantiles, dtype=float)

    def predict_quantiles(self, X):
        return self.prediction[: len(X)]


class QuantileMetricTests(unittest.TestCase):
    def test_score_perfect_predictions(self):
        quantiles = [0.1, 0.5, 0.9]
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        prediction = np.repeat(observed[:, None], len(quantiles), axis=1)

        scores = score_quantile_predictions(observed, prediction, quantiles)

        self.assertEqual(quantile_column(quantiles, 0.5), 1)
        self.assertEqual(scores["median_rmse"], 0.0)
        self.assertEqual(scores["quantile_crps"], 0.0)
        self.assertEqual(scores["quantile_crossing_rate"], 0.0)

    def test_crossing_rate_is_row_fraction(self):
        prediction = np.array([[1.0, 2.0, 3.0], [1.0, 3.0, 2.0]])
        self.assertEqual(quantile_crossing_rate(prediction), 0.5)


class QuantileEnsembleTests(unittest.TestCase):
    def test_mean_aggregation_and_median_prediction(self):
        quantiles = [0.1, 0.5, 0.9]
        first = FixedQuantileEstimator([[0, 1, 2], [3, 4, 5]], quantiles)
        second = FixedQuantileEstimator([[2, 3, 4], [5, 6, 7]], quantiles)
        ensemble = QuantileRegressionEnsemble(
            [first, second], seeds=[43, 44], quantiles=quantiles
        )

        X = np.zeros((2, 1))
        np.testing.assert_allclose(
            ensemble.predict_quantiles(X), [[1, 2, 3], [4, 5, 6]]
        )
        np.testing.assert_allclose(ensemble.predict(X), [2, 5])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble.joblib"
            ensemble.save(path)
            loaded = QuantileRegressionEnsemble.load(path)
            np.testing.assert_allclose(
                loaded.predict_quantiles(X), [[1, 2, 3], [4, 5, 6]]
            )


class CatBoostResidualQuantileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(123)
        cls.X = pd.DataFrame(
            {"salinity": np.linspace(30, 37, 60), "aux": rng.normal(size=60)}
        )
        cls.y = 2 * cls.X["salinity"] + cls.X["aux"] + rng.normal(0, 0.1, 60)

    def make_model(self):
        return CatBoostResidualRegressor(
            linear_features=["salinity"],
            feature_names=["salinity", "aux"],
            loss_function="MultiQuantile:alpha=0.1,0.5,0.9",
            iterations=8,
            depth=3,
            random_seed=43,
            allow_writing_files=False,
            verbose=False,
        )

    def test_multiquantile_prediction_shape_and_round_trip(self):
        model = self.make_model().fit(self.X, self.y)
        prediction = model.predict_quantiles(self.X.iloc[:5])
        self.assertEqual(prediction.shape, (5, 3))
        self.assertEqual(model.score(self.X, self.y).__class__, float)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            model.save(path)
            loaded = CatBoostResidualRegressor.load(path)
            np.testing.assert_allclose(
                loaded.predict_quantiles(self.X.iloc[:5]), prediction
            )

    def test_sklearn_clone_preserves_catboost_parameters(self):
        cloned = clone(self.make_model())
        self.assertEqual(
            cloned.catboost_kwargs["loss_function"],
            "MultiQuantile:alpha=0.1,0.5,0.9",
        )
        self.assertEqual(cloned.catboost_kwargs["random_seed"], 43)


if __name__ == "__main__":
    unittest.main()

from .dataio import add_talk_adjustment, load_data
from .estimators import CatBoostResidualRegressor
from .features import add_cyclical_dayofyear, add_spherical_coords
from .inference import (
    open_gridded_data,
    open_inference_data,
    predict_map_for_date,
    select_inference_date,
)
from .quantile_ensemble import QuantileRegressionEnsemble
from .quantiles import (
    pinball_loss_by_quantile,
    quantile_calibration,
    quantile_column,
    quantile_crossing_rate,
    quantile_crps,
    score_quantile_predictions,
)
from .target_filtering import drop_bad_quality_talk, drop_extreme_salinities
from .train_test_split import make_train_test_folds

__all__ = [
    "CatBoostResidualRegressor",
    "QuantileRegressionEnsemble",
    "add_cyclical_dayofyear",
    "add_spherical_coords",
    "add_talk_adjustment",
    "drop_bad_quality_talk",
    "drop_extreme_salinities",
    "load_data",
    "make_train_test_folds",
    "open_gridded_data",
    "open_inference_data",
    "pinball_loss_by_quantile",
    "predict_map_for_date",
    "quantile_calibration",
    "quantile_column",
    "quantile_crossing_rate",
    "quantile_crps",
    "score_quantile_predictions",
    "select_inference_date",
]


def main() -> None:
    print("Hello from highres-ta!")

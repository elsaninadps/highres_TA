from .dataio import add_talk_adjustment, load_data
from .estimators import CatBoostResidualRegressor
from .features import add_cyclical_dayofyear, add_spherical_coords
from .target_filtering import drop_bad_quality_talk, drop_extreme_salinities
from .train_test_split import make_train_test_folds

__all__ = [
    "CatBoostResidualRegressor",
    "add_cyclical_dayofyear",
    "add_spherical_coords",
    "add_talk_adjustment",
    "drop_bad_quality_talk",
    "drop_extreme_salinities",
    "load_data",
    "make_train_test_folds",
]


def main() -> None:
    print("Hello from highres-ta!")

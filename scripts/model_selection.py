import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import catboost
import dotenv
import numpy as np
import pandas as pd
import sklearn.base
from loguru import logger
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

ModelSklearnAPI = catboost.CatBoostRegressor | sklearn.base.BaseEstimator


LOGGING_LEVEL = "INFO"
ROOT = Path(dotenv.find_dotenv("pyproject.toml")).parent
DATA_PATH = ROOT / "data/training/GLODAPv2023-raw_collocated-{y}.pq"
CV_SCORING_METRICS = [
    "median_absolute_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "r2_score",
    "mean_absolute_percentage_error",
]
INDEX_COLUMNS = (
    "expocode",
    "time",
    "lat",
    "lon",
)
COMPULSORY_COLUMNS = {
    "talk",
    "salinity",
} | set(INDEX_COLUMNS)
SALINITY_BINS = (
    0,
    32,
    34,
    36,
    np.inf,
)

logger.remove()
logger.add(sys.stderr, level=LOGGING_LEVEL)


@dataclass
class ModelCVParams:
    model_name: Literal["RandomForest", "CatBoost"]
    param_grid: list[dict[str, object]]


@dataclass
class ModelSelectionConfig:
    fname_data_parquet: str | Path
    yname_target: Literal["talk", "talk_normalized"]
    xname_features: list[str]
    num_cv_folds: int
    params: list[ModelCVParams]
    salinity_bins: tuple[float, ...] = SALINITY_BINS
    salinity_name: str = "salinity"
    salinity_norm_value: float = 34.5


def main():

    config_fname = ROOT / "scripts/cv_example_config.yaml"
    config = load_config(config_fname)
    data_raw = load_data()

    data = preprocess_data(data_raw, config)
    train_test_splitter = split_by_expocode_salinity_bin_based(data)

    # FIXME: This isn't working from here on. something wrong with the return type
    train, test = next(train_test_splitter.split(data))

    cv_splitter = split_by_expocode_salinity_bin_based(train)

    train_x = train.drop(columns=[config.yname_target])
    train_y = train[config.yname_target]
    test_x = test.drop(columns=[config.yname_target])
    test_y = test[config.yname_target]

    cv_models = ()
    cv_results = ()
    for model_cv_params in config.params:
        cv_model = train_model(train_x, train_y, cv_splitter, model_cv_params)
        cv_results += (extract_cv_results(cv_model),)
        cv_models += (cv_model,)

    # what do we do with the results
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html


def load_config(fname_config_yaml: str | Path) -> ModelSelectionConfig:
    import yaml

    with open(fname_config_yaml, "r") as f:
        config_dict = yaml.safe_load(f)

    return ModelSelectionConfig(**config_dict)
    ...


def load_data(compulsory_columns: set[str] = COMPULSORY_COLUMNS) -> pd.DataFrame:
    """
    Loads the training data from a parquet file

    Contains all the columns and rows of the training data
    No engineering or preprocessing is done here, just loading
    the data into a pandas dataframe

    Parameters
    ----------
    fname_data_parquet : str | Path
        The path to the parquet file containing the training data
    Returns
    -------
    pd.DataFrame
        The training data as a pandas dataframe
    """
    data_path = str(DATA_PATH)

    logger.info(f"Loading data from {data_path.format(y='YYYY')} for years 1982-2021")
    data = pd.concat([pd.read_parquet(data_path.format(y=y)) for y in range(1982, 2022)])

    columns = data.columns.intersection(compulsory_columns)

    if len(columns) < len(compulsory_columns):
        missing_cols = compulsory_columns - set(columns)
        raise ValueError(f"Missing columns in the data: {missing_cols}")

    logger.debug(f"data.head() = \n{data.head()}")
    return data


def preprocess_data(df: pd.DataFrame, config: ModelSelectionConfig) -> pd.DataFrame:
    """
    Select columns, engineer features, bin salinity

    Parameters
    ----------
    df : pd.DataFrame
        The raw training df as a pandas dataframe
    config : ModelSelectionConfig
        The configuration for the model selection process, containing any parameters needed for preprocessing

    Returns
    -------
    pd.DataFrame
        The preprocessed training df
    """
    index_columns = INDEX_COLUMNS

    keep_cols = set(config.xname_features + [config.yname_target]).union(COMPULSORY_COLUMNS)

    salinity_bins = config.salinity_bins
    salinity = df[config.salinity_name]
    salt_norm_value = config.salinity_norm_value

    df["salinity_bin"] = salinity_binning(salinity, bins=salinity_bins)
    df["talk_normalized"] = norm_alkalinity(df["talk"], salinity, salt_norm_value)

    index_columns = list(INDEX_COLUMNS + ("salinity_bin",))
    df = df.set_index(index_columns)

    valid_columns = list(keep_cols - set(index_columns))
    df = df[valid_columns]

    return df


def salinity_binning(
    salinity: pd.Series, bins: tuple[float, ...], bin_labels: None | list[str | float] = None
) -> pd.Series:
    return pd.cut(salinity, bins=bins, labels=bin_labels)


def norm_alkalinity(alkalinity: pd.Series, salinity: pd.Series, norm_value: float) -> pd.Series:
    return norm_value * alkalinity / salinity


def split_by_expocode_salinity_bin_based(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, n_folds: int = 5
) -> StratifiedGroupKFold:
    index = data.index.to_frame()
    grouper = index["expocode"]
    stratifier = index["salinity_bin"]
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return splitter.split(data, stratifier, grouper)


def train_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    splitter: StratifiedGroupKFold,
    model_cv_params: ModelCVParams,
    **kwargs,
) -> GridSearchCV:

    grid_search = GridSearchCV(
        model_cv_params.model,
        model_cv_params.param_grid,
        cv=splitter,
        scoring=CV_SCORING_METRICS,
    )

    grid_search.fit(train_x, train_y)

    return grid_search


def extract_cv_results(cv_model: GridSearchCV) -> pd.DataFrame:
    return pd.DataFrame(cv_model.cv_results_)


if __name__ == "__main__":
    main()

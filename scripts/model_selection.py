import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Type

import catboost
import dotenv
import joblib
import lightgbm
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
import matplotlib.pyplot as plt

ModelSklearnAPI = (
    Type[RandomForestRegressor] | Type[catboost.CatBoostRegressor] | Type[lightgbm.LGBMRegressor]
)

CONFIG_FNAME = "model_selection_config.yaml"

LOGGING_LEVEL = "INFO"
CV_VERBOSITY = 3
N_CPUS = -1  # -1 = all available CPUs

ROOT = Path(dotenv.find_dotenv("pyproject.toml")).parent
DATA_PATH = ROOT / "data/training/GLODAPv2023-raw_collocated-{y}.pq"

CV_SCORING_METRICS = [
    "neg_root_mean_squared_error",
    "neg_median_absolute_error",
    "neg_mean_absolute_error",
    "neg_mean_absolute_percentage_error",
    "r2",
]

DF_INDEX_COLUMNS = (
    "expocode",
    "time",
    "lat",
    "lon",
)

COMPULSORY_COLUMNS = {
    "talk",
    "salinity",
} | set(DF_INDEX_COLUMNS)

SALINITY_BIN_EDGES = (
    0,
    32,
    34,
    36,
    np.inf,
)

logger.remove()
logger.add(sys.stderr, level=LOGGING_LEVEL)

ESTIMATOR_NAMES = Literal["RandomForest", "CatBoost", "LightGBM"]
ESTIMATORS = {
    "RandomForest": RandomForestRegressor,
    "CatBoost": catboost.CatBoostRegressor,
    "LightGBM": lightgbm.LGBMRegressor,
}

@dataclass
class ModelCVParams:
    model_name: ESTIMATOR_NAMES
    param_grid: list[dict[str, object]]
    model: ModelSklearnAPI
    default_kwargs: dict[str, object] = field(default_factory=dict)


@dataclass
class ModelSelectionConfig:
    fname_data_parquet: str | Path
    yname_target: Literal["talk", "talk_normalized"]
    xname_features: list[str]
    num_cv_folds: int
    params: list[ModelCVParams]
    salinity_bins: tuple[float, ...] = SALINITY_BIN_EDGES
    salinity_name: str = "salinity"
    salinity_norm_value: float = 34.5
    
    
def main():
    
    config_path = ROOT / "scripts/{CONFIG_FNAME}"
    config = load_config(config_path)
    data_raw = load_data()
    
    # IDEA: consider removing outliers or computing a weighting for these outliers
    data = preprocess_data(data_raw, config)

    # NOTE: This is a bit messy and could be neater in a function, but for now, OK
    
    train_df, test_df = get_train_test_split_by_expocode_salinity_bin_based(data)
    cv_splitter = get_splits_by_expocode_salinity_bin_based(train_df) 
    
    train_x = train_df.drop(columns=[config.yname_target])
    train_y = train_df[config.yname_target]
    
    # test_x = test_df.drop(columns=[config.yname_target])
    # test_y = test_df[config.yname_target]

    cv_models = ()
    cv_results = ()
    
    for model_cv_params in config.params:
        
        cv_model = train_model(
            train_x,
            train_y,
            splitter=cv_splitter,  # convert to list, so can be saved with pickle later
            model_cv_params=model_cv_params,
            n_jobs=N_CPUS,
            verbose=CV_VERBOSITY,
        )

        # TODO: Implement scoring on test set - here, again, stratifying by salinity bin might not be the best idea

        save_cv_model(cv_model, model_cv_params.model_name)
        cv_result = extract_cv_results(cv_model)

        cv_models += (cv_model,)
        cv_results += (cv_result,)
        
    boxplot(config, cv_results)

    cv_results_combined = combine_cv_results(cv_results)
    logger.info(f"Combined CV results: \n{cv_results_combined.T.to_markdown()}")
    
    # what do we do with the results
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html


def load_config(fname_config_yaml: str | Path) -> ModelSelectionConfig:
    import yaml

    with open(fname_config_yaml, "r") as f:
        config_dict = yaml.safe_load(f)

    config = ModelSelectionConfig(**config_dict)

    if not config.params:
        raise ValueError("No model parameters provided in the config file")
    else:
        for i, model_cv_params in enumerate(config.params):
            model_cv_params["model"] = ESTIMATORS[model_cv_params["model_name"]]  # type: ignore
            config.params[i] = ModelCVParams(**model_cv_params)  # type: ignore

    return config


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

    # Check that the compulsory columns are present in the data
    columns = data.columns.intersection(compulsory_columns)

    if len(columns) < len(compulsory_columns):
        missing_cols = compulsory_columns - set(columns)
        raise ValueError(f"Missing columns in the data: {missing_cols}")

    logger.debug(f"data.head() = \n{data.head().T.head(50)}")
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

    # salinity binning
    salinity_bins = config.salinity_bins
    salinity = df[config.salinity_name]
    df["salinity_bin"] = salinity_binning(salinity, bins=salinity_bins)
    
    # alkalinity normalization
    salt_norm_value = config.salinity_norm_value
    df["talk_normalized"] = normalize_alkalinity(df["talk"], salinity, salt_norm_value)

    # set quadruple index 
    index_columns = DF_INDEX_COLUMNS
    index_columns = list(DF_INDEX_COLUMNS + ("salinity_bin",))
    df = df.set_index(index_columns)

    # select columns and drop rows with missing values in the selected columns
    keep_cols = set(config.xname_features + [config.yname_target]).union(COMPULSORY_COLUMNS)
    valid_columns = list(keep_cols - set(index_columns))
    df = df[valid_columns].dropna()

    logger.debug(f"Preprocessed data head: \n{df.head()}")

    return df


def salinity_binning(
    salinity: pd.Series, bins: tuple[float, ...], bin_labels: None | list[str | float] = None
) -> pd.Series:
    n_bins = len(bins)
    bin_label = bin_labels or range(1, n_bins)
    return pd.cut(salinity, bins=bins, labels=bin_label)


def normalize_alkalinity(
    alkalinity: pd.Series, salinity: pd.Series, norm_value: float
) -> pd.Series:
    return norm_value * alkalinity / salinity


def get_splits_by_expocode_salinity_bin_based(
    data: pd.DataFrame, random_state: int = 42, n_folds: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    
    index = data.index.to_frame()

    grouper = index["expocode"]
    stratifier = index["salinity_bin"]

    #TODO:verify how the shuffle affects stratification and grouping
    # PROBLEM: default random_state is 42, so shuffle will always be True.
    
    shuffle = False if random_state is None else True  # if random state provided, then True
    #splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    splitter = StratifiedGroupKFold(n_splits=n_folds)

    splits = splitter.split(data, y=stratifier, groups=grouper)

    # return a list so that we can pickle the CV splitter later
    return list(splits)

def get_train_test_split_by_expocode_salinity_bin_based(
    data: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Generate a 5-fold StratifiedGroupKFold, yielding a 20% validation and 80% train distribution in each of the 5 folds
    train_test_splitter = get_splits_by_expocode_salinity_bin_based(data, random_state=random_state)
    
    # only take the first fold instance to generate our train-test split according to its train-validation (80:20) distribution 
    idx_train, idx_test = train_test_splitter[0]
    
    # generate train and test dfs according to the generated indices
    train = data.iloc[idx_train]
    test = data.iloc[idx_test]

    return train, test


def train_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    splitter: Iterable[tuple[np.ndarray, np.ndarray]],
    model_cv_params: ModelCVParams,
    **kwargs,
) -> GridSearchCV:
    
    logger.info(f"Training model with parameters: \n{model_cv_params}")
    logger.debug(f"Training data shapes: train_x = {train_x.shape}, train_y = {train_y.shape}")
    logger.debug(f"Splitter: \n{splitter}")

    props = {
        "scoring": CV_SCORING_METRICS,
        "refit": CV_SCORING_METRICS[0],
    } | kwargs
    # verbose is int otherwise, LightGBM fail
    estimator = model_cv_params.model(verbose=0, **model_cv_params.default_kwargs)

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=model_cv_params.param_grid,
        cv=splitter,
        **props,
    )
    
    grid_search.fit(train_x, train_y)

    logger.success(f"Finished training model {model_cv_params.model_name}")

    return grid_search


def save_cv_model(cv_model: GridSearchCV, model_name: str):

    save_path = ROOT / f"models/cv_{model_name}.joblib"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(cv_model, save_path, compress=1)

    logger.success(f"Saved CV model to {save_path}")


def extract_cv_results(cv_model: GridSearchCV) -> pd.DataFrame:
    results = pd.DataFrame(cv_model.cv_results_)
    model_name = cv_model.estimator.__class__.__name__
    results["model_name"] = model_name
    logger.debug(f"Extracted CV results: \n{results.head(15).T}")
    return results


def combine_cv_results(cv_results: Iterable[pd.DataFrame]) -> pd.DataFrame:
    combined_results = pd.concat(cv_results, ignore_index=True)
    logger.debug(f"Combined CV results: \n{combined_results.head(15).T}")
    return combined_results


def failsafe_checks():

    estimator_names = set(ESTIMATOR_NAMES.__args__)
    estimator_keys = set(ESTIMATORS.keys())

    assert estimator_names == estimator_keys, (
        f"ESTIMATOR_NAMES and ESTIMATORS keys must match, but got {estimator_names} and {estimator_keys}"
    )


if __name__ == "__main__":
    failsafe_checks()
    main()

















def boxplot(config, cv_results):
    
    
    nplots = len(CV_SCORING_METRICS)
    ncols = 2
    nrows = nplots // ncols + nplots % ncols
    fig, axes = plt.subplots(n_rows, ncols, figsize=(base_size[0]*ncols, base_size[1]*n_rows))
    axes = axes.flatten()
    
    model_labels = []
    nfolds = len(config.num_cv_folds)
    
    for model_cv_result in cv_results:
        
        scores_per_metric = concatenate_all_folds_scores_per_metric(nfolds, model_cv_result)
        
        model_name = model_cv_result.T['model_name'].unique()
        model_labels += model_name
    
        for i, metric in enumerate(CV_SCORING_METRICS):
            axes[i].boxplot(scores_per_metric[metric], tick_labels = model_labels)
            axes[i].set_ylabel(metric)
        
        plt.tight_layout()
        plt.show();
    
    
            
            
def concatenate_all_folds_scores_per_metric(nfolds, model_cv_result):
    results_in_cols = model_cv_result.T
    
    model_scores = pd.DataFrame()
    for metric in CV_SCORING_METRICS:        
        model_scores[metric]= pd.concat([results_in_cols[f"split0_test_{metric}"] for y in range(1, nfolds)])
        
    return model_scores


def fit_best_estimator_on_test(cv_models, test_x, test_y):
    
    for cv_model in cv_models:
        best_estimator = cv_model.best_estimator_
        test_score = best_estimator.score(test_x, test_y)
        
        residuals = test_y - best_estimator.predict(test_x)
        
        
        logger.info(f"Test score for model {cv_model.estimator.__class__.__name__}: {test_score}")
        
        
# TODO: do cv on best estimator and on test set, and do visualization on those ones.
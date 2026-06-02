# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
# ---


# %%
from dataclasses import dataclass, field
from typing import Literal, Generator, Iterator

from loguru import logger
import sys
from pathlib import Path
import dotenv
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
import numpy as np
import catboost as cb
#import functions from preprocessing.py script
import yaml
from functools import lru_cache
import xarray as xr
 


# %%
# global variables
ROOT = Path(dotenv.find_dotenv("pyproject.toml")).parent
DATA_PATH = ROOT / "data/training/GLODAPv2023-raw_collocated-{y}.pq"
CONFIG_PATH = ROOT / "scripts/config/example_training_config.yaml"


LOGGING_LEVEL = "DEBUG"
#CV_VERBOSITY = 3
N_CPUS = -1  # -1 = all available CPUs

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

ALL_FEATURES = (
    "salinity",
    "temperature",
    "bottomdepth",
    "mld_dens_soda",
    "ssh_adt",
    "ssh_sla",
    "chl_globcolour"
) #FIXME: should this be a set or a tuple or a list?


LIN_BASELINE_PARAM = {
    'fit_intercept': True,
    'copy_X': True,
    'tol': 1e-06,
    'n_jobs':None, 
    'positive': False
}

SALINITY_BIN_EDGES = (
    0,
    32,
    34,
    36,
    np.inf,
)

logger.remove()
logger.add(sys.stderr, level=LOGGING_LEVEL)

# %%

@dataclass
class TrainingConfig:

    save_folder: str
    training_mode: Literal['GridSearch','SingleRun'] # two training pipelines
    catboost_param_grid: dict 
      
    num_cv_folds: int = 5
    
    run_param_single: dict[str, object] | None = None #could be aggregated in one single run_parameter
    run_param_gridsearch: dict[str, object]  | None = None
        
    xname_features: tuple[str, ...] = ALL_FEATURES
    yname_target: str = 'talk_normalized'
    
    salinity_bins: tuple[float, ...] = SALINITY_BIN_EDGES
    salinity_name: str = "salinity"
    salinity_norm_value: float = 34.5
    
    baseline_regression: Literal["LinearRegression","None"] = 'LinearRegression' # do we want possibility for ridge, lasso etc or only numpy?
    baseline_features: tuple[str, ...] = ALL_FEATURES
    #baseline_param: dict[str, object] = field(default= LIN_BASELINE_PARAM)
    
    
    

# %%


# load configuration
# load data
# preprocess: columns to keep, normalize, bin by salinity, reindex on coords
# get train test split and cv_folds
# train a baseline linear regression
# train catboost model with gridsearch or with single parameters
# collect best estimator and refit on the whole dataset if not already done
# uncertainty: 
#   RMSEWithUncertainty --> can it work on gridsearch?
#   sample_gaussian_processes --> how does it work in our case?



# Outputs to collect
# a trained regressor
# cv scores ? 
# scores on heldout test
# prediction and residuals to do analysis

# knobs to turn on the run:
# baseline predicton: with or without, or type
# general catboost parameters
# 



def main():

    # load config
    config = load_config()

    # load and preprocess data
    data = load_data()
    data = preprocess_data(data, config)
    
    # train-test split and CV splitter generation
    train_df, test_df = get_train_test_split_by_expocode_salinity_bin_based(data)
    cv_splitter = get_splits_by_expocode_salinity_bin_based(train_df) 
    
    # train_x = train_df[list(config.xname_features)]
    # train_y = train_df[config.yname_target]
    
    # test_x = test_df[list(config.xname_features)]
    # test_y = test_df[config.yname_target]
    
    # logger.debug(f"train with col {train_x.columns} and head: {train_x.head()}")
    # logger.debug(f"train_y with and head: {train_y.head()}")
    
    
    # prepare training pools
    train_pool= make_cb_pool(config, train_df)
    test_pool = make_cb_pool(config, test_df)
    
    if config.baseline_regression == 'LinearRegression': #maybe a possibility of different linear regressions
        baseline_regressor, train_baseline, test_baseline = predict_baseline(config=config, train_df= train_df, test_df= test_df)
        train_pool.set_baseline(baseline=train_baseline)
        test_pool.set_baseline(baseline=test_baseline)
        
    catboost_model = gridsearch_train_catboost(config=config, train_pool = train_pool, cv_splitter = cv_splitter)
    
    y_test_pred, test_scores = predict_on_test(catboost_model, test_pool)
    
    inference_dataset = load_inference_data()
    inference_df = preprocess_inference_data(inference_dataset, list(config.xname_features))
    inference_predictions = predict_inference(inference_df, catboost_model)
    
    
    
    # if config.training_mode == 'SingleRun':
    #     trained_cb = single_train_catboost(config, X= train_x, y = train_y)
    # elif config.training_mode == 'GridSearch':
    #     trained_cb = gridsearch_train_catboost(config, X= train_x, y = train_y, cv_splitter=cv_splitter)
     
def make_cb_pool(config, df) -> cb.Pool:
    
    cb_pool = cb.Pool(data = df[list(config.xname_features)],
                         label= df[config.yname_target])

    
    return cb_pool

def load_config() -> TrainingConfig:
    global SAVE_PATH

    config_path = CONFIG_PATH
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config = TrainingConfig(**config_dict)


    SAVE_PATH = ROOT / config.save_folder
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    NFOLDS = config.num_cv_folds #maybe dont define NFOLDS before anything else
    
    
    if not config.training_mode:
        raise ValueError("No training mode provided in configuration file")
    
    #TODO: other verification ifs
    
    logger.debug(f"Training configuration: {config}")
    logger.success(f"Loaded config and created output folder at: {SAVE_PATH}")
    
    return config

#%%
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
    
    logger.success("Loaded data")

    logger.debug(f"data.head() = \n{data.head().head(50)}")
    return data



def preprocess_data(df: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    
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
    
    # filter outliers
    df = filter_outliers(df, column="salinity", lower_abs=20, upper_abs=40)

    # set quadruple index 
    index_columns = DF_INDEX_COLUMNS
    index_columns = list(DF_INDEX_COLUMNS + ("salinity_bin",))
    df = df.set_index(index_columns)

    # select columns and drop rows with missing values in the selected columns
    keep_cols = set(list(config.xname_features) + [config.yname_target]).union(COMPULSORY_COLUMNS)
    valid_columns = list(keep_cols - set(index_columns))
    df = df[valid_columns].dropna()


    logger.debug(f"Preprocessed data: {df.describe}")
    logger.success(f"Peprocessed data with kept column {valid_columns} and index: {index_columns}")

    return df

#%%
def filter_outliers(df: pd.DataFrame, column: str, lower_abs: float, upper_abs: float) -> pd.DataFrame:
    if lower_abs is not None:
        df = df[df[column] >= lower_abs]
    if upper_abs is not None:
        df = df[df[column] <= upper_abs]

    return df

#%%
def salinity_binning(
    salinity: pd.Series, bins: tuple[float, ...], bin_labels: None | list[str | float] = None
) -> pd.Series:
    n_bins = len(bins)
    bin_label = bin_labels or range(1, n_bins)
    return pd.cut(salinity, bins=bins, labels=bin_label)


#%%
def normalize_alkalinity(alkalinity: pd.Series, salinity: pd.Series, norm_value: float) -> pd.Series:
    return norm_value * alkalinity / salinity


#%%
def get_splits_by_expocode_salinity_bin_based(data: pd.DataFrame, random_state: int = 42, n_folds: int = 5):
    
    index = data.index.to_frame()

    grouper = index["expocode"]
    stratifier = index["salinity_bin"]

    #TODO:verify how the shuffle affects stratification and grouping
    # PROBLEM: default random_state is 42, so shuffle will always be True.
    
    shuffle = False if random_state is None else True  # if random state provided, then True
    #splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    
    splitter = StratifiedGroupKFold(n_splits=n_folds)
    logger.debug(f"Splitter with {n_folds} folds: {splitter}, type {type(splitter)}")

    splits = splitter.split(data, y=stratifier, groups=grouper)

    logger.debug(f"Splits with {n_folds} folds: {splits}, type {type(splits)}")
    
    # return a list so that we can pickle the CV splitter later
    
    return splits

#%%
def get_train_test_split_by_expocode_salinity_bin_based(
    data: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Generate a 5-fold StratifiedGroupKFold, yielding a 20% validation and 80% train distribution in each of the 5 folds
    train_test_splitter = get_splits_by_expocode_salinity_bin_based(data, random_state=random_state)
    
    train_test_splitter_list = list(train_test_splitter)
    
    # only take the first fold instance to generate our train-test split according to its train-validation (80:20) distribution 
    idx_train, idx_test = train_test_splitter_list[0]
    
    # generate train and test dfs according to the indices
    train = data.iloc[idx_train]
    test = data.iloc[idx_test]

    return train, test
    

#%% 
def train_baseline(X : pd.DataFrame, y : pd.Series, baseline_param: dict
                   ) ->  LinearRegression:

    lin_regression = LinearRegression(**baseline_param)
    lin_regression = lin_regression.fit(X, y)
    
    logger.debug(f"Regressor coefs: {lin_regression.coef_}")
    logger.success("Trained linear regression baseline")

    return lin_regression

def predict_baseline(config, train_df, test_df) -> tuple[LinearRegression, np.ndarray, np.ndarray]:
    
    train_x_base = train_df[list(config.baseline_features)]
    test_x_base = test_df[list(config.baseline_features)]
        
    baseline_regressor = train_baseline(
            X= train_x_base, 
            y= train_df[config.yname_target], 
            baseline_param=LIN_BASELINE_PARAM)          
        
    x_train_pred = baseline_regressor.predict(train_x_base)
    x_test_pred = baseline_regressor.predict(test_x_base)
    
    logger.debug(f"Linear prediction array (shape:{x_train_pred.size}): {x_train_pred}")
    logger.debug(f"Linear prediction array (shape:{x_test_pred.size}): {x_test_pred}")
    
    if x_train_pred.size != train_x_base.shape[0]:
        raise IndexError(f"Training sample size {train_x_base.shape[0]} not equal to baseline predictions size {x_train_pred.size}")
    
    if x_test_pred.size != test_x_base.shape[0]:
        raise IndexError(f"Training sample size {test_x_base.shape[0]} not equal to baseline predictions size {x_test_pred.size}")
    
    return baseline_regressor, x_train_pred, x_test_pred
    
def gridsearch_train_catboost(config : TrainingConfig, 
                              train_pool : cb.Pool, 
                              cv_splitter : StratifiedGroupKFold | Iterator[tuple[np.ndarray, np.ndarray]]
                              ) -> cb.CatBoostRegressor: #FIXME: the generator type
    

    cb_regressor = cb.CatBoostRegressor()
    
    cb_regressor.grid_search(
        param_grid = config.catboost_param_grid,
        X = train_pool,
        cv = cv_splitter,
        refit = True
    )
    #maybe do a gridsearch_run_param
    
    logger.info(f"CV_folds_scores: {cb_regressor.best_score_}")
    logger.success(f"Trained regressor")
    
    return cb_regressor


def predict_on_test(cb_model, test_pool):
    
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

    y_pred = cb_model.predict(test_pool, verbose = True)
    test_y = test_pool.get_label()

    # dataframe with scores on metrics from CV scoring metrics
    
    scores = pd.DataFrame({
        "root_mean_squared_error": [root_mean_squared_error(test_y, y_pred)],
        "median_absolute_error": [median_absolute_error(test_y, y_pred)],
        "mean_absolute_error": [mean_absolute_error(test_y, y_pred)],
        "r2": [r2_score(test_y, y_pred)],
    })
    
    #publish_test_scores(scores, cv_model.estimator.__class__.__name__)
    
    logger.info(f"Test score for model: {scores.T.to_markdown()}")
    #logger.info(f"Test score per sample for model {cv_model.estimator.__class__.__name__}: {score_sample}")
    
    return y_pred, scores

# def publish_test_scores(scores, model_name: str):
#     fig, ax = plt.subplots(figsize=(10, 4))
#     ax.axis('tight')
#     ax.axis('off')
#     table = ax.table(cellText=scores.values, colLabels=scores.columns, rowLabels=scores.index, cellLoc='center', loc='center')
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.scale(1.2, 1.5)
    
#     plt.title("Test Scores of Best Estimator on Test Set", fontsize=14)
#     plt.savefig(SAVE_PATH / f"{model_name}_test_scores.png", bbox_inches='tight')
    
#     logger.success(f"Saved test scores to {SAVE_PATH} / {model_name}_test_scores.png")
    
# def single_train_catboost(config : TrainingConfig, 
#                           X: pd.DataFrame, y : pd.DataFrame, 
# ):
    
#     # maybe a safety check on the param_grid: should only have single values for each key
    
#     cb_regressor = cb.CatBoostRegressor([*config.param_grid,*config.run_param_single])
    
#     cb_regressor.fit(X, y)
    
    
#def failsafe_checks():

    estimator_names = set(ESTIMATOR_NAMES.__args__)
    estimator_keys = set(ESTIMATORS.keys())

    assert estimator_names == estimator_keys, (
        f"ESTIMATOR_NAMES and ESTIMATORS keys must match, but got {estimator_names} and {estimator_keys}"
    )


@lru_cache(maxsize=1)
def load_inference_data() -> xr.Dataset:
    
    drop_at_import = [
        "chl_filled_flag",
        "chl_flag",
        "chl_sigma",
        "chl_sigma_bias",
        "chl_sigma_regrid",
        "chl_sigma_uncert",
        "fco2atm_noaa",
        "ice",
        "pres_std",
        "press",
        "ssh_sigma",
        "ssh_sigma_regrid",
        "ssh_sigma_uncert",
        "sss_flag",
        "sss_old",
        "sss_old_flag",
        "sss_sigma",
        "sss_sigma_uncert",
        "sst_flag",
        "sst_sigma",
        "sst_sigma_regrid",
        "sst_sigma_uncert",
        "wind_std",
        "windspeed_moment1",
        "windspeed_moment2",
        "xco2atm_mauna_loa",
        "xco2mbl_noaa",
    ]
    
    url = "https://data.up.ethz.ch/shared/OceanSODA-ETHZv2/inference_for_gregor2024/data_8daily_25km_v01.zarr/"
    ds_raw = xr.open_zarr(url, consolidated=True, group="2004", drop_variables=drop_at_import)

    url = "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/60s_bed_elev_netcdf/ETOPO_2022_v1_60s_N90W180_bed.nc"

    #bath = xr.open_dataset(url, engine="netcdf4")["z"]
    #bath = xr.open_dataset("https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_bed_elev_netcdf/ETOPO_2022_v1_60s_N90W180_bed.nc", chunks={})["z"]
    #bath = bath.coarsen(lat=15, lon=15).mean().interp_like(ds_raw).rename('bottomdepth').compute()

    #bath.to_netcdf("bath_coarse.nc")
    
    ds = ds_raw
    #ds = xr.merge([ds_raw[["sss", "sst", "ssh", "ssh_anom", "mld"]], bath])
    
    ds = ds.rename(
        sss="salinity", sst="temperature", ssh="ssh_adt", ssh_anom="ssh_sla", mld="mld_dens_soda"
    )
    return ds


def preprocess_inference_data(inference_dataset : xr.Dataset, train_columns: list[str]) -> pd.DataFrame:

    t = 0
    #if "df" not in locals() or t != t_prev or (df.columns.difference(train_x.columns)).any():

    inference_df = inference_dataset.isel(time=[t]).to_dataframe()
    coords = inference_df.index.to_frame()
    inference_df["lat"] = coords["lat"]
    inference_df["lon_sin"] = np.sin(np.radians(coords["lon"]))
    inference_df["lon_cos"] = np.cos(np.radians(coords["lon"]))
    day_of_year = coords["time"].dt.dayofyear
    inference_df["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
    inference_df["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)

    #t_prev = t

    df = inference_df.rename(columns = {'chl_filled':'chl_globcolour'})[train_columns]
    dfnn = inference_df.dropna()
    
    
    logger.debug(f"Inference dataframe: {inference_df.to_markdown()}")
    logger.success("Preprocessed inference data")
    return dfnn


def predict_inference(inference_df: pd.DataFrame, model: cb.CatBoostRegressor | LinearRegression):
    
    #loop on t
    predictions = model.predict(inference_df)
    
    logger.debug(f"Predictions for model {model._estimator_type}: {inference_df.to_markdown()}")
    logger.success("Predictions on inference data")
    return predictions


if __name__ == "__main__":
    #failsafe_checks()
    main()
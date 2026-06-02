from loguru import logger
from pathlib import Path
import dotenv
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.model_selection import StratifiedGroupKFold
import yaml
import pooch
from dataclasses import dataclass
from typing import Iterable, Literal, Type
from functools import lru_cache


# %%
# global variables
ROOT = Path(dotenv.find_dotenv("pyproject.toml")).parent
DATA_PATH = ROOT / "data/training/GLODAPv2023-raw_collocated-{y}.pq"
#CONFIG_PATH = ROOT / "scripts/config/example_training_config.yaml"


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


ALL_FEATURES = ['salinity', 'temperature', 'bottomdepth', 'mld_dens_soda', 'ssh_adt', 'ssh_sla', 'chl_globcolour', 'coordsA', 'coordsB', 'coordsC']

SALINITY_BIN_EDGES = (
    0,
    32,
    34,
    36,
    np.inf,
)

SALINITY_NORM_VALUE = 34.5


@dataclass
class ModelConfig:
    yname_target: Literal["talk", "talk_normalized"] 
    xname_features: list[str]
    run_name: str
    #num_cv_folds: int = 5
    salinity_bins: tuple[float, ...] = SALINITY_BIN_EDGES
    salinity_name: str = "salinity"
    salinity_norm_value: float = 34.5
    fig_save_path: str | Path = f"{ROOT}/outputs"
    model_save_path: str | Path = f"{ROOT}/models"

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

   
def compute_n_coords(lat, lon):
    """
    Spherical coordinates 
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return x, y, z

def add_n_coords(df: pd.DataFrame) -> pd.DataFrame:
    n_coords = compute_n_coords(df["lat"], df["lon"])
    df["ncoord_a"] = n_coords[0]
    df["ncoord_b"] = n_coords[1]
    df["ncoord_c"] = n_coords[2]
    return df

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


def train_test_split(
    df: pd.DataFrame, config: ModelConfig
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    itrain, itest = get_splits_by_expocode_salinity_bin_based(df, n_folds=7)[1]
    train_x = df.iloc[itrain][config.xname_features]
    train_y = df.iloc[itrain][config.yname_target]
    test_x = df.iloc[itest][config.xname_features]
    test_y = df.iloc[itest][config.yname_target]
    return train_x, train_y, test_x, test_y


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    filter = (
        (df["salinity"] > 25)
        & (df["salinity"] < 40)
        & (df["bottomdepth"] > 100)
        & (df["talk"] > 1500)
        & (df["talk"] < 3000)
    )
    df = df[filter]
    return df


def load_config(fname_config_yaml: str | Path) -> ModelConfig:

    with open(fname_config_yaml, "r") as f:
        config_dict = yaml.safe_load(f)

    config = ModelConfig(**config_dict)
    
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

    #logger.debug(f"data.head() = \n{data.head().T.head(50)}")
    return data



def preprocess_data(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    
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

    if not (config.salinity_name == 'salinity'):
        df = df.drop('salinity', axis = 1)
        df = df.rename(columns = {config.salinity_name : 'salinity'})
        

    # filter outliers
    df = filter_outliers(df)

    
    # salinity binning
    salinity_bins = config.salinity_bins
    salinity = df['salinity']
    df["salinity_bin"] = salinity_binning(salinity, bins=salinity_bins)
    
    # alkalinity normalization
    salt_norm_value = config.salinity_norm_value
    df["talk_normalized"] = normalize_alkalinity(df["talk"], salinity, salt_norm_value)

    df["lon"] = ((df["lon"] + 180) % 360) - 180
    
    # add spherical coordinates
    df = add_n_coords(df)
    
    df = add_coastal_flag(df)
    
    # coordinates transformation
    #df["lon"] = (df["lon"] - 180)
    

    # set quadruple index 
    index_columns = DF_INDEX_COLUMNS
    index_columns = list(DF_INDEX_COLUMNS + ("salinity_bin","is_coastal","depth"))
    df = df.set_index(index_columns)

    # select columns and drop rows with missing values in the selected columns
    keep_cols = set(config.xname_features + [config.yname_target]).union(COMPULSORY_COLUMNS)
    valid_columns = list(keep_cols - set(index_columns))
    df = df[valid_columns].dropna()

    #df = df.drop_duplicates()

    #logger.debug(f"Preprocessed data head: \n{df.head()}")

    return df




# def add_coastal_flag(df, to_index = False):
#     # add coastal flag
#     coast_mask = get_coastal_mask()
#     selector = df[["lat", "lon"]].reset_index(drop=True).to_xarray()
#     df["is_coastal"] = coast_mask.sel(selector, method="nearest", tolerance=0.6)
    
#     if to_index:
#         df = df.set_index(["is_coastal"], append=True)
        
#     return df

@lru_cache(1)
def get_coastal_mask() -> xr.DataArray:
    
    from .utils import make_target_grid
    
    target_grid = make_target_grid()
    url = "https://raw.githubusercontent.com/RECCAP2-ocean/R2-shared-resources/refs/heads/master/data/regions/RECCAP2_region_masks_all_v20221025.nc"
    fname = pooch.retrieve(url, None, fname="RECCAP2_region_masks_all_v20221025.nc")
    ds = xr.open_dataset(fname)
    coast = ds.coast.assign_coords(lon=lambda x: (x.lon + 180) % 360 - 180).sortby("lon").compute()
    coast = coast.interp_like(target_grid, method="nearest").astype(bool)
    return coast


def add_coastal_flag(
    df: pd.DataFrame,
    to_index: bool = False,
) -> pd.DataFrame:

    coast_mask = get_coastal_mask()

    # retrieve coordinates whether they are columns or index levels
    if "lat" in df.columns:
        lat = df["lat"].values
    else:
        lat = df.index.get_level_values("lat").values

    if "lon" in df.columns:
        lon = df["lon"].values
    else:
        lon = df.index.get_level_values("lon").values

    # # ensure longitude convention matches coast_mask
    # lon = ((lon + 180) % 360) - 180

    selector = {
        "lat": xr.DataArray(lat, dims="points"),
        "lon": xr.DataArray(lon, dims="points"),
    }

    df["is_coastal"] = coast_mask.sel(
        selector,
        method="nearest",
        tolerance=0.6,
    ).values.astype(bool)

    if to_index:
        df = df.set_index(["is_coastal"], append=True)

    return df


# def load_config(fname_config_yaml):
    
#     with open(fname_config_yaml, "r") as f:
#         config = yaml.safe_load(f)
        
#     config = munch.munchify(config)

#     return config


# def load_data() -> pd.DataFrame:

    
#     data_path = str(DATA_PATH)
#     #compulsory_columns = COMPULSORY_COLUMNS

#     logger.info(f"Loading data from {data_path.format(y='YYYY')} for years 1982-2021")
#     data = pd.concat([pd.read_parquet(data_path.format(y=y)) for y in range(1982, 2022)])   
    
#     return data

# TODO: add possibility to save train test split
# TODO: clarify the save_path procedure (eg mkdir, image outputs vs models outputs etc)/maybe a run_name
# TODO: maybe a function that does loading and preprocessing and saving in one go to use in external codes
# TODO: clarify coordinates transormation!!
from loguru import logger
from pathlib import Path
import dotenv
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
import yaml
import munch

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


def load_config(fname_config_yaml):
    
    with open(fname_config_yaml, "r") as f:
        config = yaml.safe_load(f)
        
    config = munch.munchify(config)

    return config


def load_data() -> pd.DataFrame:

    
    data_path = str(DATA_PATH)
    #compulsory_columns = COMPULSORY_COLUMNS

    logger.info(f"Loading data from {data_path.format(y='YYYY')} for years 1982-2021")
    data = pd.concat([pd.read_parquet(data_path.format(y=y)) for y in range(1982, 2022)])   
    
    return data

    

def preprocess_data(df: pd.DataFrame, config:dict) -> pd.DataFrame:
    

    salinity_bins = SALINITY_BIN_EDGES
    salinity_norm_value = SALINITY_NORM_VALUE
    
    n_coords = compute_n_coords(df["lat"], df["lon"])
    df["ncoord_a"] = n_coords[0]
    df["ncoord_b"] = n_coords[1]
    df["ncoord_c"] = n_coords[2]
    
    #filter outliers
    df = filter_outliers(df, **config.outliers_filter)
    
    # salinity binning
    salinity = df['salinity']
    df["salinity_bin"] = salinity_binning(salinity, bins=salinity_bins)
    
    # alkalinity normalization
    df["talk_normalized"] = normalize_alkalinity(df["talk"], salinity, salinity_norm_value)
    
    # n_coords = compute_n_coords(df["lat"], df["lon"])
    # df = df.join(n_coords)

    # set quadruple index 
    index_columns = DF_INDEX_COLUMNS
    index_columns = list(DF_INDEX_COLUMNS + ("salinity_bin",))
    df = df.set_index(index_columns)
    

    # select columns and drop rows with missing values in the selected columns
    keep_cols = set(config.xname_features + [config.yname_target])
    valid_columns = list(keep_cols - set(index_columns))
    
    df = df[valid_columns].dropna()
    
    
    
    #logger.debug(f"Preprocessed data head: \n{df.head()}")


    return df

#%%
# def filter_outliers(df: pd.DataFrame, column: str, lower_abs: float, upper_abs: float) -> pd.DataFrame:
#     if lower_abs is not None:
#         df = df[df[column] >= lower_abs]
#     if upper_abs is not None:
#         df = df[df[column] <= upper_abs]

    return df


def filter_outliers(df, lower_sal, upper_sal, lower_depth):
    
    filter = (df["salinity"] > lower_sal) & (df["salinity"] < upper_sal) & (df["bottomdepth"] > lower_depth)
    df = df[filter]
    
    logger.info(f"Filter outliers ({lower_sal} < salinity < {upper_sal}) & ({lower_depth} < bottomdepth)")
    
    return df

    
    
#%%
def salinity_binning(
    salinity: pd.Series, bins: tuple[float, ...], bin_labels: None | list[str | float] = None
) -> pd.Series:
    n_bins = len(bins)
    bin_label = bin_labels or range(1, n_bins)
    return pd.cut(salinity, bins=bins, labels=bin_label)


#%%
def normalize_alkalinity(
    alkalinity: pd.Series, salinity: pd.Series, norm_value: float
) -> pd.Series:
    return norm_value * alkalinity / salinity


# def compute_n_coords(lat: pd.Series, lon: pd.Series) -> pd.DataFrame:
    
#     """spherical coordinates"""
#     lat_rad = np.radians(lat)
#     lon_rad = np.radians(lon)
#     ncoordA = np.cos(lat_rad) * np.cos(lon_rad)
#     ncoordB = np.cos(lat_rad) * np.sin(lon_rad)
#     ncoordC = np.sin(lat_rad)
    
#     spherical_coords_df =  pd.DataFrame({
#         'coordsA': ncoordA,
#         'coordsB': ncoordB,
#         'coordsC': ncoordC
#         })
    
#     return spherical_coords_df


# def compute_n_coords(lat: pd.Series, lon: pd.Series) -> pd.DataFrame:
#     """spherical coordinates"""
#     lat_rad = np.radians(lat)
#     lon_rad = np.radians(lon)
#     ncoordA = np.cos(lat_rad) * np.cos(lon_rad)
#     ncoordB = np.cos(lat_rad) * np.sin(lon_rad)
#     ncoordC = np.sin(lat_rad)
    
#     spherical_coords_df =  pd.DataFrame({
#         'coordsA': ncoordA,
#         'coordsB': ncoordB,
#         'coordsC': ncoordC
#         })
    
#     return spherical_coords_df

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


#%%
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



def get_train_test_data(data, config)-> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    
    train_test_splitter = get_splits_by_expocode_salinity_bin_based(data)
    
    logger.info(f"Selecting the {config.ith_split}^th split iteration")
    idx_train, idx_test = train_test_splitter[config.ith_split]

    train_df = data.iloc[idx_train]
    test_df = data.iloc[idx_test]

    train_x = train_df[config.xname_features]
    train_y = train_df[config.yname_target]
    
    test_x = test_df[config.xname_features]
    test_y = test_df[config.yname_target]
    
    logger.success(f"X_features {config.xname_features} enforced in train_x, test_x")
    logger.success(f"Target variable {config.yname_target} enforced in train_y, test_y")
    
    # assert 'talk' in train_x.columns or 'talk_normalized' in train_x.columns
     
    # if ('talk' in colname for colname in train_x.columns) or ('talk' in colname for colname in test_x.columns) :
    #      raise ValueError('alk variable in training or test x_datasets')
 
    return train_x, train_y, test_x, test_y
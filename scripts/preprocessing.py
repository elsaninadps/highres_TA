def load_config() -> TrainingConfig:


    # import model/run config
    # assign a global variable save_path a path from config for saving the model and results
    
    config_fname_yaml = CONFIG_FNAME
    
    global SAVE_PATH

    SAVE_PATH = ROOT / config.save_path
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

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

    logger.debug(f"data.head() = \n{data.head().T.head(50)}")
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
    keep_cols = set(config.xname_features + [config.yname_target]).union(COMPULSORY_COLUMNS)
    valid_columns = list(keep_cols - set(index_columns))
    df = df[valid_columns].dropna()

    logger.debug(f"Preprocessed data head: \n{df.head()}")

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
def normalize_alkalinity(
    alkalinity: pd.Series, salinity: pd.Series, norm_value: float
) -> pd.Series:
    return norm_value * alkalinity / salinity


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

#%%
def get_train_test_split_by_expocode_salinity_bin_based(
    data: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Generate a 5-fold StratifiedGroupKFold, yielding a 20% validation and 80% train distribution in each of the 5 folds
    train_test_splitter = get_splits_by_expocode_salinity_bin_based(data, random_state=random_state)
    
    # only take the first fold instance to generate our train-test split according to its train-validation (80:20) distribution 
    idx_train, idx_test = train_test_splitter[0]
    
    # generate train and test dfs according to the indices
    train = data.iloc[idx_train]
    test = data.iloc[idx_test]

    return train, test

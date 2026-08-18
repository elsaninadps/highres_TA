import pathlib

import pandas as pd
from loguru import logger


def load_data(glob_path_to_files: str = "../data/training/*.pq") -> pd.DataFrame:
    """
    Load all parquet files in a folder into a single pandas DataFrame.
    """

    # Get a list of all parquet files in the folder
    path = pathlib.Path(glob_path_to_files)
    parent_folder = path.parent
    parquet_files = list(parent_folder.glob(path.name))
    n_files = len(parquet_files)

    logger.debug(f"Loading {n_files} {path.suffix} files from {parent_folder}")

    # Load each parquet file into a DataFrame and concatenate them
    df_list = [pd.read_parquet(file) for file in parquet_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    return combined_df


def load_talk_adjustment(fname: str) -> pd.Series:
    """
    Downloaded from : https://glodapv2.geomar.de/adjustments/select_cruises
    Should be able to use this directly
    """
    return (
        pd
        .read_csv(
            fname,
            skiprows=2,
            skipinitialspace=True,
            na_values=[-999, -888, -777, 0],
            usecols=["cruise_expocode", "alkalinity_adj"],
            index_col="cruise_expocode",
        )
        .dropna()
        .alkalinity_adj
    )


def add_talk_adjustment(df: pd.DataFrame, fname: str) -> pd.DataFrame:
    """
    Add a new column to the DataFrame with talk adjustments based on expocode.

    Parameters:
    - df: pd.DataFrame containing the data.
    - fname: Path to the CSV file containing talk adjustments.

    Returns:
    - pd.DataFrame with an additional column 'talk_adj' containing the adjustments.
    """

    adjustments = load_talk_adjustment(fname).to_dict()
    df = df.copy()
    df["talk_adj"] = df["expocode"].map(adjustments).fillna(0)

    return df

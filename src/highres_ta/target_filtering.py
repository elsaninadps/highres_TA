import pandas as pd
from loguru import logger


def drop_extreme_salinities(
    df, salinity_col_name: str = "salinity", min: float = 20, max: float = 40
):
    """
    Drop rows from the DataFrame where salinity is outside the specified range.

    Parameters:
    - df: pd.DataFrame containing the data.
    - salinity_col_name: Name of the column in df that contains salinity values.
    - min: Minimum acceptable salinity value (inclusive).
    - max: Maximum acceptable salinity value (inclusive).

    Returns:
    - pd.DataFrame with rows outside the specified salinity range removed.
    """
    return df[(df[salinity_col_name] >= min) & (df[salinity_col_name] <= max)]


def drop_bad_quality_talk(
    df: pd.DataFrame,
    talkf_col_name: str = "talkf",
    talk_adj_col_name: str = "talk_adj",
    adjustment_threshold_molkg: float = 6.0,
) -> pd.DataFrame:
    """
    Drop rows from the DataFrame where the quality control columns indicate bad quality.

    Parameters:
    - df: pd.DataFrame containing the data.
    - talkf_col_name: Name of the column in df that contains additional quality flags.
    - talk_adj_col_name: Name of the column in df that contains talk adjustments.

    Returns:
    - pd.DataFrame with rows indicating bad quality removed.
    """
    adjustment = df[talk_adj_col_name].abs()
    small_adjustment = adjustment <= adjustment_threshold_molkg
    good_flags = df[talkf_col_name] == 2

    large_adjustment_count = (~small_adjustment).sum()
    notgood_flags_count = (~good_flags).sum()
    filtered_count = (~(good_flags & small_adjustment)).sum()
    total_count = len(df)

    logger.debug(
        f"TA values with large adjustments (<= {adjustment_threshold_molkg} mol/kg): {large_adjustment_count}"
    )
    logger.debug(f"TA values without good flags (!= 2): {notgood_flags_count}")
    logger.info(
        f"Number of rows filtered due to large adjustments and bad flags: {filtered_count} of {total_count} ({filtered_count / total_count:.0%})"
    )

    return df[good_flags & small_adjustment]

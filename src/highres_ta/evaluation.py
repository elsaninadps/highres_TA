from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from scipy.special import huber
import pandas as pd
from loguru import logger
import numpy as np
from typing import Literal

INDEXES_LITERAL = Literal["expocode", "time", "lat", "lon"]

def scoring(prediction_df:pd.DataFrame)->pd.Series:
    
    y_true = prediction_df.y_true
    y_pred = prediction_df.y_pred
    residuals = prediction_df.residuals
    
    
    scores = pd.Series({
        "root_mean_squared_error": root_mean_squared_error(y_true, y_pred),
        "mean_absolute_error": mean_absolute_error(y_true, y_pred),
        "huber_loss": huber(1.35, residuals).mean(),
        "median_absolute_error": median_absolute_error(y_true, y_pred),
        "mean_bias": residuals.mean(),
        "median_bias":residuals.median(),
        "r2_score": r2_score(y_true, y_pred)
    })

    
    logger.info(f"{scores.T.to_markdown()}")
    
    return scores
    
    #logger.info(f"Test score per sample for model {cv_model.estimator.__class__.__name__}: {score_sample}")
    
    
def make_prediction_df(y_pred:np.ndarray, y_true: pd.Series)-> pd.DataFrame:   
    
    predictions_df = pd.DataFrame({
        'y_pred': y_pred,
        'y_true': y_true,
    }, index =  y_true.index)

    predictions_df['residuals'] = predictions_df.y_true-predictions_df.y_pred

    return predictions_df


def grouped_scoring(predictions_df: pd.DataFrame, time_resampling_freq = 'YE'):

    # if isinstance(groupby, str):
    #     grouped = predictions_df.groupby(level=groupby)

    # else:
    #     # normalize to aligned Series
    #     groupby = pd.Series(groupby, index=predictions_df.index, name="group")
    #     grouped = predictions_df.groupby(groupby)
        
    #lat = predictions_df.index.get_level_values("lat")

    grouped = predictions_df.groupby([
        pd.Grouper(level="time", freq=time_resampling_freq),
        pd.cut(predictions_df.index.get_level_values("lat"), bins=3)
    ])
        

    return grouped.apply(scoring)




def build_groupers(
    df: pd.DataFrame,
    time_freq: str = "year",  # "year" or "month"
    lat_bins=None,
    lon_bins=None,
    salinity_group=False,
):
    """
    Build grouping keys for MultiIndex-based dataframe.
    Returns list of aligned groupers.
    """

    groupers = []

    # -----------------------
    # TIME GROUPING
    # -----------------------
    time = df.index.get_level_values('time')

    if time_freq == "year":
        groupers.append(time.year)

    elif time_freq == "month":
        groupers.append(time.to_period("M"))

    else:
        raise ValueError("time_freq must be 'year' or 'month'")

    # -----------------------
    # LAT GROUPING
    # -----------------------
    if lat_bins is not None:
        lat = df.index.get_level_values('lat')
        groupers.append(pd.cut(lat, bins=lat_bins))

    # -----------------------
    # LON GROUPING
    # -----------------------
    if lon_bins is not None:
        lon = df.index.get_level_values('lon')
        groupers.append(pd.cut(lon, bins=lon_bins))

    # -----------------------
    # SALINITY GROUPING
    # -----------------------
    if salinity_group:
        sal = df.index.get_level_values('salinity_bin').unique()
        sal = groupers.append(sal)


    return groupers


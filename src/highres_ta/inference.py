import string
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from highres_ta.quantile_ensemble import QuantileRegressionEnsemble

from .features import latlon_to_spherical_coords

FNAME_PROCESSED = "/net/sea/work/gregorl/projects/oceansoda_v2/data/processed/OceanSODA_ETHZv2-gridded_data-8D_025deg.zarr"
FNAME_BATHYMETRY = "/net/sea/work/datasets/grd/ocean/2d/obs/bathymetry/ETOPO1/ETOPO15_mean_bath.nc"
FNAME_NUTRIENTS = "/net/sea/work/gregorl/projects/OceanSODA-ETHZv2/data-in/carbsys/nutrients_woa18_t46y720x1440.zarr"


@lru_cache(1)
def open_gridded_data(fname_processed=FNAME_PROCESSED, fname_bathymetry=FNAME_BATHYMETRY):
    data_8D_025deg = xr.open_zarr(fname_processed)

    bottomdepth = (
        xr
        .open_dataarray(fname_bathymetry)
        .rename({"latitude": "lat", "longitude": "lon"})
        .interp_like(data_8D_025deg)
    )
    coords = bottomdepth.to_series().index.to_frame()
    ncoords = latlon_to_spherical_coords(coords.lat, coords.lon).to_xarray()

    data_8D_025deg["bottomdepth"] = bottomdepth
    for key in ncoords.data_vars:
        data_8D_025deg[key] = ncoords[key]

    return data_8D_025deg.astype(np.float32)


@lru_cache(1)
def open_inference_data(x_names: tuple[str, ...]):
    ds = open_gridded_data()
    x_names_list = list(x_names)

    features = xr.Dataset()
    features["salinity"] = ds.sss_cci.fillna(ds.sss_soda).fillna(ds.sss_multi_nrt)
    features["temperature"] = ds.sst_cci.fillna(ds.sst_c3s_icdr)
    features["ssh_adt"] = ds.adt_duacs
    features["bottomdepth"] = ds.bottomdepth
    features["silicate"] = ds.silicate_woa
    features["nitrate"] = ds.nitrate_woa
    features["phosphate"] = ds.phosphate_woa
    features["ncoord_x"] = ds.ncoord_x
    features["ncoord_y"] = ds.ncoord_y
    features["ncoord_z"] = ds.ncoord_z

    features = features[x_names_list]

    return features


def create_labels_for_members(model: QuantileRegressionEnsemble) -> pd.MultiIndex:
    """
    Creates labels for each member of the ensemble based on their split and replicate.
    """
    member_splits = np.array(model.member_splits, dtype=int)
    n_splits = max(member_splits) - min(member_splits) + 1
    n_replicates = int(len(member_splits) / n_splits)

    replicates = string.ascii_lowercase[:n_replicates] * (max(member_splits) + 1)
    labels = list(zip(member_splits, replicates))
    labels = pd.MultiIndex.from_tuples(labels, names=["split", "replicate"])

    return labels


def select_inference_date(
    ds: xr.Dataset, date: str | list[str | pd.Timestamp] | pd.Timestamp
) -> pd.DataFrame:
    """
    Selects the nearest date from the dataset.
    """
    time = ds.time.sel(time=date, method="nearest")
    dayofyear = time.dt.dayofyear

    if not isinstance(date, list):
        date = [date]

    ds = ds.sel(time=date, dayofyear=dayofyear, method="nearest").drop_vars("dayofyear")
    logger.debug(ds)
    df = ds.to_dataframe().dropna(subset="salinity")
    return df


def predict_map_for_date(
    model: QuantileRegressionEnsemble, features: xr.Dataset, date: str
) -> xr.DataArray:
    """
    Predicts the target variable using the provided model and features.
    """
    logger.info(f"Selecting features for {date}")
    features_df = select_inference_date(features, date)

    logger.info(f"Predicting ensemble members for {date}")
    yhat = model.predict_members(features_df)
    logger.debug(f"yhat shape: {yhat.shape}")

    da = (
        xr
        .DataArray(
            yhat,
            dims=["member", "coords", "quantile"],
            coords={
                "member": create_labels_for_members(model),
                "coords": features_df.index,
                "quantile": model.quantiles,
            },
        )
        .unstack(["member", "coords"])
        .chunk({"lat": 180, "lon": 180, "quantile": 1})
    )

    da.replicate.attrs = {
        "description": "Replicate of the model for each split, each with its own hyperparameters."
    }
    da.split.attrs = {
        "description": "Split of the model for each fold of cross-validation (shared across replicates)."
    }
    da.lat.attrs = {"units": "degrees_north", "long_name": "Latitude"}
    da.lon.attrs = {"units": "degrees_east", "long_name": "Longitude"}
    da["quantile"].attrs = {"units": "dimensionless", "long_name": "Quantile"}

    da["replicate"] = da.replicate.astype("str")
    da["split"] = da.split.astype("int32")

    return da

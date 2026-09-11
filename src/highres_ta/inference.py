from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr

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


def select_inference_date(ds: xr.Dataset, date: str) -> pd.DataFrame:
    """
    Selects the nearest date from the dataset.
    """
    time = ds.time.sel(time=date, method="nearest")
    dayofyear = time.dt.dayofyear
    ds = ds.sel(time=[date], dayofyear=[dayofyear], method="nearest")
    df = ds.to_dataframe().dropna()
    return df


def predict_map_for_date(model, features: xr.Dataset, date: str) -> xr.Dataset:
    """
    Predicts the target variable using the provided model and features.
    """

    features_df = select_inference_date(features, date)

    yhat = model.predict_members(features_df)

    yhat_avg_df = pd.DataFrame(
        np.median(yhat, axis=0),
        index=features_df.index,
        columns=model.quantiles,
    )

    yhat_q70_df = pd.DataFrame(
        np.std(yhat, axis=0),
        index=features_df.index,
        columns=model.quantiles,
    )

    yhat_avg_da = (
        yhat_avg_df
        .to_xarray()
        .to_array(dim="quantile", name="talk_pred")
        .transpose("time", "dayofyear", "quantile", "lat", "lon")
    )

    yhat_q70_da = (
        yhat_q70_df
        .to_xarray()
        .to_array(dim="quantile", name="talk_pred_q70")
        .transpose("time", "dayofyear", "quantile", "lat", "lon")
    )

    yhat_ds = xr.merge([yhat_avg_da, yhat_q70_da])

    return yhat_ds

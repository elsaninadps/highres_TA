from functools import lru_cache
import pandas as pd
import pooch
import xarray as xr
from loguru import logger
import dotenv
from pathlib import Path

from .preprocessing import compute_n_coords
from . import estimators as models


ROOT = Path(dotenv.find_dotenv("pyproject.toml")).parent

def inference(model, train_x, time: str | pd.Timestamp = "2004-02-01"):
    
    pred_x = get_inference_x(inference_time=time, train_x=train_x)
    ds = inference_from_x_data(model=model, inference_x=pred_x)
    return ds


def get_inference_x(inference_time: str | pd.Timestamp, train_x):
    
    logger.info("Getting inference data...")
    
    if not isinstance(inference_time, pd.Timestamp):
        inference_time = pd.Timestamp(inference_time)
    assert isinstance(inference_time, pd.Timestamp), "Time must be a pandas Timestamp"
    
    inference_x = load_inference_data(train_x, date=inference_time)

    return inference_x

def inference_from_x_data(model, inference_x):
    
    logger.info("Getting inference data...")
    logger.info("Predicting all components for inference data...")
    
    pred_y = model.predict_components(inference_x)
    
    logger.info("Inference completed.")

    ds = pred_y.to_xarray().sortby(["lat", "lon"])
    
    return ds

@lru_cache(12)
def _load_zarr_data(url, group=None) -> xr.Dataset:
    ds = xr.open_zarr(url, consolidated=True, group=group)
    return ds


def _get_data(year: int) -> xr.Dataset:
    url = "https://data.up.ethz.ch/shared/OceanSODA-ETHZv2/.inference_for_gregor2024/data_8daily_25km_v01.zarr/"
    ds = _load_zarr_data(url, group=str(year))
    return ds


def _get_woa(year: int) -> xr.Dataset:
    url = "https://data.up.ethz.ch/shared/OceanSODA-ETHZv2/.inference_for_gregor2024/WOA18_nutrients_8daily.zarr/"
    ds = _load_zarr_data(url)
    time = ds.dayofyear.astype("timedelta64[D]") + pd.to_datetime(f"{year}-01-01")
    ds = ds.rename({"dayofyear": "time"}).assign_coords(time=time)
    return ds


def _get_clim(year: int) -> xr.Dataset:
    url = "https://data.up.ethz.ch/shared/OceanSODA-ETHZv2/.inference_for_gregor2024/clims_8daily_25km_v01.zarr/"
    ds = _load_zarr_data(url)
    time = ds.dayofyear.astype("timedelta64[D]") + pd.to_datetime(f"{year}-01-01")
    ds = ds.rename({"dayofyear": "time"}).assign_coords(time=time)
    return ds



@lru_cache(1)
def get_bottom_depth() -> xr.DataArray:
    from .utils import make_target_grid
    target_grid = make_target_grid()
    return (
        xr.open_dataarray(ROOT / "data/bathymetry_etopo2022_25km.nc")
        .compute()
        .interp_like(target_grid, method="nearest")
    )


def load_raw_inf_data(date: pd.Timestamp):
    
    
    from .preprocessing import get_coastal_mask
    
    
    ds = _get_data(date.year)
    woa = _get_woa(date.year)
    clim = _get_clim(date.year)
    xr.align(ds, woa, clim, join="exact")  # ensure same time coordinate for merging

    vars_avail = [c for c in clim.data_vars if c in ds.data_vars]
    vars_missing = [c for c in clim.data_vars if c not in ds.data_vars]

    ds = xr.merge([ds, woa, clim[vars_missing]], compat="override", join="exact")
    ds[vars_avail] = ds[vars_avail].fillna(clim[vars_avail])  # fill missing values with climatology

    ds["bottomdepth"] = get_bottom_depth()
    ds["is_coastal"] = get_coastal_mask()
    
    
    ds = ds.sel(time=date, method="nearest")
    
    return ds
    

# TODO: rename to preprocess_inference_data and check where used

def load_inference_data(train_x, date: pd.Timestamp):

    ds = load_raw_inf_data(date)

    rename = dict(
        sss="salinity",
        sst="temperature",
        ssh="ssh_adt",
        ssh_anom="ssh_sla",
        phosphate="phosphate",
        nitrate="nitrate",
        silicate="silicate",
        chl_filled="chl_globcolour",
        bottomdepth="bottomdepth",
        is_coastal="is_coastal",
    )
    
    ds = ds.rename(rename)[list(rename.values())]
    

    df = ds.to_dataframe()
    coords = df.index.to_frame()
    df["ncoord_a"], df["ncoord_b"], df["ncoord_c"] = compute_n_coords(coords["lat"], coords["lon"])
    df["depth"] = 5

    pred_X = df[train_x.columns].dropna()
    
    return pred_X


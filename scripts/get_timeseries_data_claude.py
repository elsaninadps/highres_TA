"""
Download BATS and HOT timeseries of total alkalinity, with temperature and salinity to match
"""

from __future__ import annotations

import pathlib
import sys
from functools import lru_cache

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pooch
import xarray as xr
from loguru import logger
from scipy.special import huber
from sklearn import metrics

from highres_ta import BaggingCatBoostResidualRegressor

logger.remove()  # Remove default logger to avoid duplicate logs when imported as a module
logger.add(sys.stdout, level="INFO")

ROOT = pathlib.Path(dotenv.find_dotenv("pyproject.toml")).parent

STANDARD_NAMES = [
    "lat",
    "lon",  # -180 to 180
    "time",  # YYYY-MM-DD
    "salinity",
    "temperature",
    "talk",
]

BATS_BOTTLE_URL = (
    "https://datadocs.bco-dmo.org/dataset/3782/file/wn3zAMZuP0Arwj/3782_v9_bats_bottle.csv"
)

HOT_CSV_URL = "https://datadocs.bco-dmo.org/dataset/3773/file/QArEXBMh6mlRyk/3773_v3_niskin_hot001_yr01_to_hot348_yr35.csv"
BOTTLE_MAX_DEPTH = 20.0  # meters


def _resample_8d_jan1(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out: list[pd.DataFrame] = []
    for year, g in df.groupby(df["time"].dt.year):
        g2 = g.set_index("time").sort_index()
        start = f"{year}-01-01"
        r = g2.resample(pd.Timedelta("8D"), origin=start).mean().reset_index()
        out.append(r)

    out_df = pd.concat(out, ignore_index=True)
    return out_df


def download_bco_dmo_dataset(url: str) -> str:

    fname = pooch.retrieve(
        url=url,
        known_hash=None,
        fname=url.split("/")[-1],
    )

    return fname


def download_spot() -> pd.DataFrame:
    url = "https://datadocs.bco-dmo.org/dataset/896862/file/m7XKy7gc6krX2K/spots.csv"
    fname = download_bco_dmo_dataset(url)
    df = pd.read_csv(fname, na_values=[-999, -999.0])
    logger.info(f"Downloaded {len(df)} SPOTS samples")

    # DATE is YYYYMMDD int, TIME is HHMM int — combine into datetime
    df["time"] = pd.to_datetime(
        df["DATE"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    cols = {
        "TimeSeriesSite": "site",
        "time": "time",
        "LATITUDE": "lat",
        "LONGITUDE": "lon",
        "CTDPRS": "depth",
        "CTDTMP": "temperature",
        "ALKALI": "talk",
        "ALKALI_FLAG_W": "alkali_flag",
        "SALNTY": "salinity_bottle",
        "SALNTY_FLAG_W": "salinity_bottle_flag",
        "CTDSAL": "salinity_ctd",
        "CTDSAL_FLAG_W": "salinity_ctd_flag",
        "NITRAT": "nitrate",
        "SILCAT": "silicate",
        "PHSPHT": "phosphate",
    }

    df = df[list(cols)].rename(columns=cols)

    # Prefer bottle salinity (flag == 2), fall back to CTD salinity
    sal_bottle = df["salinity_bottle"].where(df["salinity_bottle_flag"] == 2)
    sal_ctd = df["salinity_ctd"].where(df["salinity_ctd_flag"] == 2)
    df["salinity"] = sal_bottle.combine_first(sal_ctd)

    # Keep only good-quality alkalinity
    df = df[df["alkali_flag"] == 2]
    df = df[df["depth"] <= BOTTLE_MAX_DEPTH]

    return _common_processing(df)


def download_irminger_timeseries() -> pd.DataFrame:
    url = "https://datadocs.bco-dmo.org/dataset/911407/file/vmvAX9NU4gqY2V/911407_v1_ooi_irminger_sea_discrete_water_sampling_data.csv"

    cols = {
        "CTD_Bottle_Closure_Time": "time",
        "CTD_Latitude": "lat",
        "CTD_Longitude": "lon",
        "CTD_Depth": "depth",
        "CTD_Temperature_1": "temperature",
        "CTD_Salinity_1": "salinity_ctd",
        "Discrete_Salinity": "salinity_bottle",
        "Discrete_Alkalinity": "talk",
    }

    fname = download_bco_dmo_dataset(url)
    df = pd.read_csv(fname)
    logger.info(f"Downloaded {len(df)} Irminger Sea samples")

    df = df[list(cols)].rename(columns=cols)

    # Prefer discrete (bottle) salinity, fall back to CTD salinity
    df["salinity"] = df["salinity_bottle"].combine_first(df["salinity_ctd"])

    df = df[df["depth"] <= 30]

    return _common_processing(df)


def download_bats_timeseries() -> pd.DataFrame:
    cols = {
        "ISO_DateTime_UTC": "time",
        "Latitude": "lat",
        "Longitude": "lon",
        "Depth": "depth",
        "Salinity": "salinity",
        "Temperature": "temperature",
        "Alkalinity": "talk",
        "PO4": "phosphate",
        "NO3_plus_NO2": "nitrate_nitrite",
        "NO2": "nitrite",
        "Silicate": "silicate",
    }
    fname = download_bco_dmo_dataset(BATS_BOTTLE_URL)
    df = pd.read_csv(fname)
    logger.info(f"Downloaded {len(df)} BATS surface samples")

    print(df.columns.tolist())  # Debugging line to check available columns
    df = df[list(cols)].rename(columns=cols)
    df["nitrate"] = df["nitrate_nitrite"] - df["nitrite"]

    df = df[df["depth"] <= BOTTLE_MAX_DEPTH]

    return _common_processing(df)


def download_hot_timeseries() -> pd.DataFrame:
    cols = {
        "Sampling_ISO_DateTime_UTC": "time",
        "Latitude": "lat",
        "Longitude": "lon",
        "CTDPRS": "depth",
        "CTDTMP": "temperature",
        "ALKALIN": "talk",
        "CTDSAL": "salinity_ctd",
        "SALNITY": "salinity_bottle",
    }

    fname = download_bco_dmo_dataset(HOT_CSV_URL)
    df = pd.read_csv(fname)
    logger.info(f"Downloaded {len(df)} HOT surface samples")

    df = df[list(cols)].rename(columns=cols)

    # Prefer bottle salinity, fall back to CTD salinity
    df["salinity"] = df["salinity_bottle"].combine_first(df["salinity_ctd"])

    # Station ALOHA only (22°45'N, 158°00'W); excludes other HOT stations
    df = df[df["lat"].between(22.25, 23.25) & df["lon"].between(-158.5, -157.5)]
    df = df[df["depth"] <= BOTTLE_MAX_DEPTH]

    return _common_processing(df)


def download_estoc_timeseries() -> pd.DataFrame:
    url = "https://doi.pangaea.de/10.1594/PANGAEA.959856?format=textfile"
    fname = pooch.retrieve(url, None, fname="estoc_timeseries.txt")

    # PANGAEA text files wrap metadata in /* ... */; find the closing marker
    with open(fname) as f:
        for n_skip, line in enumerate(f):
            if line.startswith("*/"):
                break

    df = pd.read_csv(fname, sep="\t", skiprows=n_skip + 1)
    logger.info(f"Downloaded {len(df)} ESTOC samples")

    cols = {
        "Date/Time": "time",
        "Latitude": "lat",
        "Longitude": "lon",
        "Depth water [m]": "depth",
        "Temp [°C]": "temperature",
        "Sal": "salinity",
        "AT [µmol/kg]": "talk",
    }

    df = df[list(cols)].rename(columns=cols)

    df = df[df["depth"] <= BOTTLE_MAX_DEPTH]
    df = df.dropna(subset=["time"]).dropna(subset=["talk", "salinity", "temperature"], how="all")

    return _common_processing(df)


def _common_processing(df: pd.DataFrame) -> pd.DataFrame:
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_convert(None)
    for c in ["lat", "lon", "salinity", "temperature", "talk", "depth"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # df = _resample_8d_jan1(df)
    df = df.dropna(subset=["time"]).dropna(subset=["talk", "salinity", "temperature"], how="any")
    demo = pd.concat(
        [
            df.head(10),
            df.tail(10),
        ]
    )

    logger.info(f"Processed dataframe with {len(df)} and columns: {df.columns.tolist()}")
    logger.debug(f"Dataset structure after cleaning and resampling: \n{demo}")

    return df


def _get_data(year: int) -> xr.Dataset:
    url = "https://data.up.ethz.ch/shared/OceanSODA-ETHZv2/.inference_for_gregor2024/data_8daily_25km_v01.zarr/"
    ds = _load_zarr_data(url, group=str(year))
    return ds


@lru_cache(12)
def _load_zarr_data(url, group=None) -> xr.Dataset:
    ds = xr.open_zarr(url, consolidated=True, group=group)
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


def get_inference_data(date: pd.Timestamp, lat: float, lon: float) -> pd.Series:

    ds = _get_data(date.year)
    woa = _get_woa(date.year)
    clim = _get_clim(date.year)
    xr.align(ds, woa, clim, join="exact")  # ensure same time coordinate for merging

    vars_avail = [c for c in clim.data_vars if c in ds.data_vars]
    vars_missing = [c for c in clim.data_vars if c not in ds.data_vars]

    ds = xr.merge([ds, woa, clim[vars_missing]], compat="override")
    ds[vars_avail] = ds[vars_avail].fillna(clim[vars_avail])  # fill missing values with climatology

    ds["bottomdepth"] = get_bottom_depth()
    ds["is_coastal"] = get_coastal_mask()

    ds = ds.sel(time=date, lat=lat, lon=lon, method="nearest")

    rename = dict(
        sss="salinity_sss",
        sst="temperature_sst",
        nitrate="nitrate",
        phosphate="phosphate",
        silicate="silicate",
        ssh="ssh_adt",
        chl_filled="chl_globcolour",
        bottomdepth="bottomdepth",
        is_coastal="is_coastal",
    )
    ds = ds.rename(rename)[list(rename.values())].load()

    ser = ds.to_pandas()
    assert isinstance(ser, pd.Series)

    return ser


@lru_cache(1)
def make_target_grid(res=0.25):
    import numpy as np

    lat = np.arange(-90 + res / 2, 90, res)
    lon = np.arange(-180 + res / 2, 180, res)
    return xr.Dataset(coords={"lat": lat, "lon": lon})


@lru_cache(1)
def get_coastal_mask() -> xr.DataArray:
    target_grid = make_target_grid()
    url = "https://raw.githubusercontent.com/RECCAP2-ocean/R2-shared-resources/refs/heads/master/data/regions/RECCAP2_region_masks_all_v20221025.nc"
    fname = pooch.retrieve(url, None, fname="RECCAP2_region_masks_all_v20221025.nc")
    ds = xr.open_dataset(fname)
    coast = ds.coast.assign_coords(lon=lambda x: (x.lon + 180) % 360 - 180).sortby("lon").compute()
    coast = coast.interp_like(target_grid, method="nearest").astype(bool)
    return coast


@lru_cache(1)
def get_bottom_depth() -> xr.DataArray:
    target_grid = make_target_grid()
    return (
        xr.open_dataarray(ROOT / "data/bathymetry_etopo2022_25km.nc")
        .compute()
        .interp_like(target_grid, method="nearest")
    )


def collocate(df: pd.DataFrame) -> pd.DataFrame:
    out: list[pd.Series] = []
    total = len(df)
    warned = False
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            ser = get_inference_data(row["time"], row["lat"], row["lon"])
            combo = pd.concat([row, ser])
            out.append(combo)
            if i % 10 == 0:
                logger.info(
                    f"Collocated row {i + 1}/{total} (time={row['time']:%Y-%m-%d}, lat={row['lat']:03.1f}, lon={row['lon']:03.1f})"
                )
        except Exception as e:
            if not warned:
                logger.warning(f"Failed to collocate row {i + 1}/{total}: {e}")
            warned = True
            continue

    return pd.concat(out, axis=1).T


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


def save_collocated_data():
    spot_df = download_spot()
    spot_df = collocate(spot_df)
    spot_df.to_csv(ROOT / "data/spot_timeseries_collocated.csv", index=False)

    bats_df = download_bats_timeseries()
    bats_df = collocate(bats_df)
    bats_df.to_csv(ROOT / "data/bats_timeseries_collocated.csv", index=False)

    # hot_df = download_hot_timeseries()
    # hot_df = collocate(hot_df)
    # hot_df.to_csv(ROOT / "data/hot_timeseries_collocated.csv", index=False)

    # irminger_df = download_irminger_timeseries()
    # irminger_df = collocate(irminger_df)
    # irminger_df.to_csv(ROOT / "data/irminger_timeseries_collocated.csv", index=False)

    # estoc_df = download_estoc_timeseries()
    # estoc_df = collocate(estoc_df)
    # estoc_df.to_csv(ROOT / "data/estoc_timeseries_collocated.csv", index=False)


def plot_y_yhat(y, yhat, ax=None, **scatter_kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    scores = scoring(y, yhat)

    y = np.array(y).flatten()
    yhat = np.array(yhat).flatten()

    scores_md = scores.to_markdown(floatfmt=".2f").replace("|", "").strip().replace(":", " ") + " "

    ax.scatter(y, yhat, **scatter_kwargs)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", label="1:1 line")
    ax.set_xlabel("Observed Alkalinity (µmol kg$^{-1}$)")
    ax.set_ylabel("Predicted Alkalinity (µmol kg$^{-1}$)")
    ax.legend()
    ax.text(
        0.98,
        0.02,
        scores_md,
        transform=ax.transAxes,
        fontsize=6,
        va="bottom",
        ha="right",
        fontfamily="monospace",
        bbox=dict(edgecolor="none", facecolor="white", alpha=0.8),
    )
    return ax


def plot_timeseries(ser: pd.Series, ax=None, **scatter_kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    time = pd.to_datetime(ser.index.to_frame()["time"])
    ser = ser.reset_index(drop=True).set_axis(time).sort_index()
    ax.plot(ser.index, ser.values, **scatter_kwargs)
    # ser.plot(y="talk", ax=ax, **scatter_kwargs)

    ax.set_xlabel("")
    ax.set_ylabel("")

    from matplotlib.dates import DateFormatter, YearLocator

    ax.xaxis.set_major_locator(YearLocator(5))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.xaxis.set_tick_params(rotation=0)
    ax.xaxis.set_minor_locator(YearLocator(5))

    return fig, ax


def scoring(y_true, y_pred) -> pd.DataFrame:

    residuals = y_pred - y_true
    scores = pd.Series(
        {
            "RMSE": metrics.root_mean_squared_error(y_true, y_pred),
            "Huber Loss (∆=1.34)": huber(1.35, residuals).mean(),
            "MAE": metrics.mean_absolute_error(y_true, y_pred),
            "Median AE": metrics.median_absolute_error(y_true, y_pred),
            "Mean Bias": residuals.mean(),
            "Median Bias": residuals.median(),
            "RMSE / σ(y)": metrics.root_mean_squared_error(y_true, y_pred) / y_true.std(),
            "r² score": metrics.r2_score(y_true, y_pred),
        },
        name="Scores",
    ).to_frame()
    scores.index.name = "Metric"

    logger.info(f"\n{scores.to_markdown(floatfmt='.3f')}")

    return scores


def predict_time_series_station(df: pd.DataFrame, name: str):

    model = BaggingCatBoostResidualRegressor.load(
        ROOT / "models/bagged_catboost_residual_model.pkl"
    )

    linear_features = model.estimators_[0].linear_features_
    df = add_n_coords(df)
    df = df.set_index(["time", "lat", "lon"])
    df = df.groupby("time").mean(numeric_only=True)
    df = df.dropna(subset=["talk"] + linear_features, how="any")

    names = model.estimators_[0].feature_names
    x = df[names]
    y = df["talk"]
    yhat = pd.Series(model.predict(x), index=df.index)

    fig, axs = plt.subplots(
        1, 2, figsize=(12, 4), width_ratios=[1, 2], sharey=True, constrained_layout=True
    )

    plot_y_yhat(y, yhat, alpha=0.7, ax=axs[0])

    plot_timeseries(y, alpha=0.7, label="Observed", ax=axs[1])
    plot_timeseries(yhat, alpha=0.7, label="Predicted", ax=axs[1])

    fig.suptitle(f"Station {name}")
    return fig, axs


def main():
    csv_path = ROOT / "data/bats_timeseries_collocated.csv"
    df = pd.read_csv(csv_path)
    fig, ax = predict_time_series_station(df, name="BATS")
    figlist = [fig]

    # the remaining stations are in SPOT
    csv_path = ROOT / "data/spot_timeseries_collocated.csv"
    df = pd.read_csv(csv_path)

    groups = df.groupby("site", as_index=False)
    for site, group in groups:
        logger.info(f"Predicting time series for site {site} with {len(group)} samples")
        fig, ax = predict_time_series_station(group, name=site)
        figlist.append(fig)

    from inference import save_figs_to_pdf

    save_figs_to_pdf(figlist, ROOT / "timeseries_predictions.pdf")


if __name__ == "__main__":
    main()

import pathlib
from functools import lru_cache

import dotenv
import numpy as np
import pandas as pd
import pooch
import xarray as xr
from cartopy.crs import PlateCarree
from loguru import logger
from matplotlib import pyplot as plt
from model_selection import (
    ModelSelectionConfig,
    get_splits_by_expocode_salinity_bin_based,
    load_config,
    load_data,
    preprocess_data,
)
from scipy.special import huber
from sklearn import metrics

from highres_ta import BaggingCatBoostResidualRegressor
from highres_ta import estimators as models

ROOT = pathlib.Path(dotenv.find_dotenv("pyproject.toml")).parent

LINEAR_FEATURES = [
    "salinity",
    "temperature",
]

MODEL_PARAMS = dict(
    n_estimators=24,
    iterations=850,
    polynomial_degree=2,
    max_samples=0.66,
    random_strength=1,
    loss_function="MAE",
    linear_features=LINEAR_FEATURES,
    min_data_in_leaf=40,
    n_jobs=8,
)


def main():

    config = load_config(ROOT / "scripts/cv_example_config.yaml")
    df_raw = load_data()
    df_raw["lon"] = (df_raw["lon"] - 180) % 360 - 180

    coast_mask = get_coastal_mask()
    selector = df_raw[["lat", "lon"]].reset_index(drop=True).to_xarray()
    df_raw["is_coastal"] = coast_mask.sel(selector, method="nearest", tolerance=0.6)

    df = add_n_coords(df_raw)
    df = preprocess_data(df, config)
    df = filter_outliers(df)
    train_x, train_y, test_x, test_y = train_test_split(df, config)

    figs = []
    if True:
        fig0, fig1, fig2, fig3 = train(train_x, train_y, test_x, test_y)
        figs.extend([fig0, fig1, fig2, fig3])

    bagged_model = models.BaggingCatBoostResidualRegressor.load(
        ROOT / "models/bagged_catboost_residual_model.pkl"
    )
    yhat_test = bagged_model.predict(test_x)

    ds = inference(bagged_model, train_x)

    fig4, axs = plot_predictions(ds)
    plot_scores_map(test_x, test_y, yhat_test, ax=axs[4], vmin=-40, vmax=40)
    figs.append(fig4)

    save_figs_to_pdf(figs)


def save_figs_to_pdf(figs: list[plt.Figure], filename="training_results.pdf", **props):
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(filename) as pdf:
        for fig in figs:
            props = dict(dpi=300, bbox_inches="tight") | props
            pdf.savefig(fig, **props)


def train(train_x, train_y, test_x, test_y):

    tr_coast = train_x.pop("is_coastal").astype(bool)
    te_coast = test_x.pop("is_coastal").astype(bool)

    bagged_model = models.BaggingCatBoostResidualRegressor(**MODEL_PARAMS)

    logger.info(f"Fitting model with columns: {train_x.columns.tolist()}")
    bagged_model.fit(train_x, train_y)
    bagged_model.save(ROOT / "models/bagged_catboost_residual_model.pkl")

    yhat_train = bagged_model.predict(train_x)
    yhat_test = bagged_model.predict(test_x)

    scores = pd.concat(
        [
            scoring(train_y.loc[~tr_coast], yhat_train[~tr_coast]).rename("Train Open Ocean"),
            scoring(test_y.loc[~te_coast], yhat_test[~te_coast]).rename("Test Open Ocean"),
            scoring(train_y.loc[tr_coast], yhat_train[tr_coast]).rename("Train Coastal"),
            scoring(test_y.loc[te_coast], yhat_test[te_coast]).rename("Test Coastal"),
        ],
        axis=1,
    )
    # plot scores as table in figure
    fig0, ax0 = plt.subplots(figsize=(8, 3))
    ax0.axis("off")
    ax0.table(scores.round(3), loc="center", cellLoc="center")
    ax0.set_title("Model Performance Metrics", fontsize=14, fontweight="bold")

    logger.info(f"\n{scores.to_markdown(floatfmt='.3f')}")

    fig1, axs = plt.subplots(2, 1, figsize=(12, 10), subplot_kw={"projection": PlateCarree(205)})
    plot_scores_map(
        train_x,
        train_y,
        yhat_train,
        ax=axs[0],
    )
    plot_scores_map(test_x, test_y, yhat_test, ax=axs[1])
    axs[0].set_title("Train Residuals (Predicted - True)", fontsize=14, fontweight="bold")
    axs[1].set_title("Test Residuals (Predicted - True)", fontsize=14, fontweight="bold")
    [ax.coastlines(lw=0.5) for ax in axs]
    cbar = plt.colorbar(
        axs[0].collections[0],
        ax=axs,
        location="right",
        label="Residual (µmol kg$^{-1}$)",
        shrink=0.6,
    )
    cbar.set_ticks(range(-20, 22, 4))

    fig2, axs2 = plt.subplots(
        2, 1, figsize=(12, 7), sharey=True, sharex=True, constrained_layout=True
    )
    plot_residuals_y(train_y, yhat_train, ax=axs2[0])
    plot_residuals_y(test_y, yhat_test, ax=axs2[1])
    axs2[0].set_title("Train Residuals (Predicted - True)", fontsize=14, fontweight="bold")
    axs2[1].set_title("Test Residuals (Predicted - True)", fontsize=14, fontweight="bold")
    axs2[0].set_xlabel("Observed Total Alkalinity (µmol/kg)")
    axs2[1].set_xlabel("Observed Total Alkalinity (µmol/kg)")
    axs2[0].set_ylabel("Residual (µmol/kg)")
    axs2[1].set_ylabel("Residual (µmol/kg)")
    axs2[1].set_ylabel("")

    fig3, _ = plot_feature_importance(bagged_model, train_x)

    return fig0, fig1, fig2, fig3


def inference(model, train_x):
    logger.info("Getting inference data...")

    time = pd.Timestamp("2004-02-01")
    assert isinstance(time, pd.Timestamp), "Time must be a pandas Timestamp"

    pred_x = load_inference_data(train_x, date=time)

    logger.info("Predicting all components for inference data...")
    pred_y = model.predict_components(pred_x)
    logger.info("Inference completed.")

    ds = pred_y.to_xarray().sortby(["lat", "lon"])
    return ds


def plot_feature_importance(model: BaggingCatBoostResidualRegressor, train_x, ax=None):
    feature_importances = []
    for estimator in model.estimators_:
        feature_importances.append(estimator.boosting_model_.get_feature_importance())
    df = pd.DataFrame(feature_importances, columns=train_x.columns)
    mean_importance = df.mean().sort_values(ascending=False)
    std_importance = df.std()[mean_importance.index]

    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    mean_importance.plot.barh(xerr=std_importance, ax=ax, capsize=4)
    ax.set_ylabel("Feature Importance")
    ax.set_title("Mean Feature Importance with Std Dev")
    ax.grid(axis="x")

    return fig, ax


def plot_residuals_y(y, yhat, ax=None, **props):

    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    y = np.array(y).flatten()
    yhat = np.array(yhat).flatten()

    residuals = pd.Series(yhat - y).set_axis(y).sort_index()
    ax = residuals.plot(marker="o", linestyle="", alpha=0.5, ax=ax, **props)
    ax.grid(axis="x", alpha=0.3)

    ax.set_ylim(-50, 50)
    ax.axhline(0, color="black", lw=1, zorder=-1)

    return fig, ax


def add_n_coords(df: pd.DataFrame) -> pd.DataFrame:
    n_coords = compute_n_coords(df["lat"], df["lon"])
    df["ncoord_a"] = n_coords[0]
    df["ncoord_b"] = n_coords[1]
    df["ncoord_c"] = n_coords[2]
    return df


def train_test_split(
    df: pd.DataFrame, config: ModelSelectionConfig
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


def scoring(y_true, y_pred) -> pd.DataFrame:
    from loguru import logger

    residuals = y_pred - y_true
    scores = pd.Series(
        {
            "root_mean_squared_error": metrics.root_mean_squared_error(y_true, y_pred),
            "mean_absolute_error": metrics.mean_absolute_error(y_true, y_pred),
            "huber_loss": huber(1.35, residuals).mean(),
            "median_absolute_error": metrics.median_absolute_error(y_true, y_pred),
            "mean_bias": residuals.mean(),
            "median_bias": residuals.median(),
            "r2_score": metrics.r2_score(y_true, y_pred),
        },
        name="Scores",
    ).to_frame()
    scores.index.name = "Metric"

    logger.debug(f"\n{scores.to_markdown(floatfmt='.3f')}")

    return scores.Scores


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


def load_inference_data(train_x, date: pd.Timestamp):

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
    ds = ds.sel(time=date, method="nearest")

    df = ds.to_dataframe()
    coords = df.index.to_frame()
    df["ncoord_a"], df["ncoord_b"], df["ncoord_c"] = compute_n_coords(coords["lat"], coords["lon"])
    df["depth"] = 5

    pred_X = df[train_x.columns].dropna()

    return pred_X


def plot_scores_map(train_x, train_y, yhat, ax=None, **kwargs):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("RdBu_r", lut=11)

    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1, projection=PlateCarree(205))
        ax.coastlines(lw=0.5)
    else:
        fig = ax.get_figure()

    coords = train_x.index.to_frame()
    props = dict(cmap=cmap, s=10, vmin=-22, vmax=22, transform=PlateCarree()) | kwargs
    ax.scatter(
        coords["lon"],
        coords["lat"],
        c=(yhat.flatten() - train_y.values.flatten()),
        **props,
    )

    return fig, ax


def plot_predictions(pred_y: xr.Dataset):
    import matplotlib.pyplot as plt
    from cartopy import crs as ccrs

    fig, axs = plt.subplots(
        3,
        2,
        figsize=(12, 8),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
        dpi=100,
        subplot_kw={"projection": ccrs.PlateCarree(205)},
    )

    target_grid = make_target_grid()
    pred_y = pred_y.reindex_like(target_grid).ffill("lon", limit=2).bfill("lon", limit=2)

    proj = ccrs.PlateCarree()
    props = dict(transform=proj, rasterized=True)
    props_alk = dict(vmin=2200, vmax=2450, cmap="Spectral_r", **props)
    img1 = pred_y.full_avg.plot.imshow(ax=axs[0, 0], **props_alk)
    img3 = pred_y.linear_avg.plot.imshow(ax=axs[1, 0], **props_alk)
    img5 = pred_y.boosted_avg.plot.imshow(ax=axs[2, 0], vmin=-40, vmax=40, cmap="RdBu_r", **props)

    props = dict(vmin=0, vmax=15, rasterized=True, transform=proj)
    img2 = pred_y.full_std.plot.imshow(**props, ax=axs[0, 1])
    img4 = pred_y.linear_std.plot.imshow(**props, ax=axs[1, 1])
    img6 = pred_y.boosted_std.plot.imshow(**props, ax=axs[2, 1])

    [ax.set_ylabel("") for ax in axs.flat]
    [ax.set_xlabel("") for ax in axs.flat]

    img1.colorbar.set_label("Total Alkalinity (µmol/kg)")
    img2.colorbar.set_label("∆ Total Alkalinity (µmol/kg)")
    img3.colorbar.set_label("Total Alkalinity (µmol/kg)")
    img4.colorbar.set_label("σ Residual (µmol/kg)")
    img6.colorbar.set_label("σ Residual (µmol/kg)")

    text_props = dict(fontsize=14, fontweight="bold", color="black", loc="left", va="top")
    axs = axs.flatten()
    axs[0].set_title(" Final prediction", **text_props)
    axs[1].set_title(" Combined σ", **text_props)
    axs[2].set_title(" Linear baseline", **text_props)
    axs[3].set_title(" Linear σ", **text_props)
    axs[4].set_title(" CatBoost residual", **text_props)
    axs[5].set_title(" Catboost σ", **text_props)

    for ax in axs.flatten():
        ax.set_xlabel("")
        ax.set_ylabel("Latitude")
        ax.coastlines(lw=0.5)

    return fig, axs


if __name__ == "__main__":
    from get_timeseries_data_claude import main as plot_timeseries

    main()
    plot_timeseries()

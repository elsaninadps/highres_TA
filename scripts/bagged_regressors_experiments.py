import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dotenv
from pathlib import Path
from loguru import logger
from cartopy.crs import PlateCarree
from cartopy.feature import COASTLINE
from sklearn import metrics
from scipy.special import huber
import xarray as xr

from highres_ta.preprocessing import load_config, load_data, preprocess_data, train_test_split, add_coastal_flag
from highres_ta import BaggingCatBoostResidualRegressor
from highres_ta import estimators as models
from highres_ta.inference import inference
from highres_ta.utils import save_figs_to_pdf
from highres_ta.evaluation import scoring

ROOT = Path(dotenv.find_dotenv("pyproject.toml")).parent

LINEAR_FEATURES = [
    "salinity",
    "temperature",
]

DEFAULT_PARAMS = dict(
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

    config = load_config(fname_config_yaml="example_config.yaml")
    global RUN_NAME
    RUN_NAME = config.run_name
    
    logger.info(f"Starting run: {RUN_NAME}")
    
    df = load_data()
    df = preprocess_data(df, config)
    df = add_coastal_flag(df)
    
    train_x, train_y, test_x, test_y = train_test_split(df, config)
    
    figs = []
    if True:
        fig0, fig1, fig2, fig3 = train(train_x, train_y, test_x, test_y, run_name=RUN_NAME)
        figs.extend([fig0, fig1, fig2, fig3])

    bagged_model = models.BaggingCatBoostResidualRegressor.load(
        ROOT / f"models/{RUN_NAME}.pkl"
    )
    yhat_test = bagged_model.predict(test_x)

    ds = inference(bagged_model, train_x)

    fig4, axs = plot_predictions(ds)
    plot_scores_map(test_x, test_y, yhat_test, ax=axs[4], vmin=-40, vmax=40)
    figs.append(fig4)

    save_figs_to_pdf(figs, f"{ROOT}/outputs/{RUN_NAME}.pdf")


def train(train_x, train_y, test_x, test_y, run_name, model_params = None,):

    tr_coast = train_x.index.get_level_values("is_coastal").values.astype(bool)
    te_coast = test_x.index.get_level_values("is_coastal").values.astype(bool)

    if model_params == None:    
        model_params = DEFAULT_PARAMS
    bagged_model = models.BaggingCatBoostResidualRegressor(**model_params)

        
    logger.info(f"Fitting model with columns: {train_x.columns.tolist()}")
    bagged_model.fit(train_x, train_y)
    bagged_model.save(ROOT / f"models/{run_name}.pkl")

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


def plot_predictions(pred_y: xr.Dataset):
    import matplotlib.pyplot as plt
    from cartopy import crs as ccrs
    from highres_ta.utils import make_target_grid

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


if __name__ == "__main__":
    main()
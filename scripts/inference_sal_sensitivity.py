# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: highres_TA (3.11.15)
#     language: python
#     name: python3
# ---

# %%
from model_selection import (
    ModelSelectionConfig,
    get_splits_by_expocode_salinity_bin_based,
    load_config,
    load_data,
    preprocess_data,
    salinity_binning,
)
import dotenv
import pathlib
from scipy.special import huber
from highres_ta import BaggingCatBoostResidualRegressor
from highres_ta import estimators as models
from inference import train_test_split
from optuna_inference import prepare_data, save_figs_to_pdf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from loguru import logger
import matplotlib.pyplot as plt
from inference import load_inference_data
import pathlib
from highres_ta.evaluation import get_set_props
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from inference import make_target_grid

# %%
ROOT = pathlib.Path(dotenv.find_dotenv("pyproject.toml")).parent
MODEL_FILE = "bagged_catboost_residual_modelstructural_optuna_4.pkl"     
OUTPUTS_FOLDER = ROOT/f"outputs/inference_sensitivities"


# %%
@dataclass
class InferenceSet:
    inference_x: pd.DataFrame
    pred_ds : pd.DataFrame
    label: str 
    color : str 
    marker: str
    linestyle: str
    noise: np.ndarray | None | pd.Series =None


# %%
def main():
    
    
    plt.close("all")

    # Loads and config
    trained_model =   models.BaggingCatBoostResidualRegressor.load(
        ROOT / f"models/{MODEL_FILE}"
    )
    
    config = load_config(ROOT / "scripts/cv_example_config.yaml")
    config.xname_features = trained_model.feature_names_in_.tolist() #TODO: clarify how config is used here
    
    df = prepare_data(config)
    train_x, train_y, test_x, test_y = train_test_split(df, config)
    
    global INFERENCE_TIME, TRAINED_MODEL, FEATURES

    TRAINED_MODEL = trained_model
    FEATURES = trained_model.feature_names_in_.tolist()
    INFERENCE_TIME = pd.Timestamp("2004-02-01")
    
    #inference_x = get_inference_x(INFERENCE_TIME, train_x)
    
    inference_x_ds = get_inference_x_ds()
    
    pred_ds = inference_from_x_data(trained_model, inference_x)
    
    noisy_inf_x = inference_x.copy()
    noise = 0.05*noisy_inf_x['salinity']
    noisy_inf_x['salinity'] += noise
    noisy_pred_ds = inference_from_x_data(trained_model, noisy_inf_x)
    
    set_props = get_set_props(0)
    original_inf_set = InferenceSet(**{
        'inference_x':inference_x_ds,
        'pred_ds': pred_ds,
        'label': 'original SSS'
    },**set_props)
    
    set_props = get_set_props(1)
    noisy_inf_set = InferenceSet(**{
        'inference_x': noisy_inf_x,
        'pred_ds': noisy_pred_ds,
        'noise': noise,
        'label': "1.05*SSS noise"
    }, **set_props)
    
    
    
    
    # compare_predictions_map(original_inf_set, noisy_inf_set, plotted_columns=['salinity', 'full_avg', 'full_std'])
    # compare_predictions_map(original_inf_set, noisy_inf_set, plotted_columns=['salinity', 'boosted_avg', 'boosted_std'])
    # compare_predictions_map(original_inf_set, noisy_inf_set, plotted_columns=['salinity', 'linear_avg', 'linear_std'])
    
    #fig0 = compare_noise_distributions([original_inf_set, noisy_inf_set])
    #fig1 = compare_predictions_map(original_inf_set, noisy_inf_set, plotted_columns=['salinity', 'full_avg','linear_avg', 'boosted_avg'])



   


# %%
def make_inference_x_ds(years = range(1990, 2011)):
    
    import pandas as pd
    import xarray as xr

    ds_list = []

    for y in years:
        t = pd.Timestamp(f"{y}-02-01")
        df = get_inference_x(t, train_x)   # pandas DataFrame (MultiIndex lat/lon)
        ds = df.to_xarray()                # → Dataset with (lat, lon)
        ds = ds.expand_dims(time=[t])      # add time dimension
        ds_list.append(ds)
        

    # concatenate along time
    inference_ds = xr.concat(ds_list, dim="time")
    
    return inference_ds
    
    


def map_variable(ds, ax, cmap, vmin = None, vmax = None):

    target_grid = make_target_grid()
    ds = ds.reindex_like(target_grid).ffill("lon", limit=2).bfill("lon", limit=2)
    
    if vmin == None or vmax == None:
        vmin = ds.min()- 0.1*abs(ds.min())
        vmax = ds.max()+ 0.1*abs(ds.max())

    proj = ccrs.PlateCarree()
    props = dict(transform=proj, rasterized=True)
    props_var = dict(vmin=vmin, vmax=vmax, cmap=cmap, **props)
    im = ds.plot.imshow(ax=ax, **props_var)
    
    return im
#ax.colorbar.set_label('label')

# def compare_var_map(inference_set1, inference_set2, var):
    
#     fig, axs = plt.subplots(
#         nrows = 3,
#         figsize=(12, 8),
#         squeeze=False,
#         sharex=True,
#         sharey=True,
#         constrained_layout=True,
#         dpi=100,
#         subplot_kw={"projection": ccrs.PlateCarree(205)},
#     )
    
#     var1 = inference_set1.inference_x['var']
#     var2 = inference_set2.inference_x['var']
#     delta_var = var1-var2
    
#     axs[0] = map_variable(var1, axs[0,0])
#     axs[1] = map_variable(var2, axs[1,0])
#     axs[2] = map_variable(delta_var, axs[2,0], label = f"∆  {var}")

#     plt.show()



# %%
def compare_predictions_map(original_inference_set, noisy_inference_set, plotted_columns):
    import matplotlib.pyplot as plt
    from cartopy import crs as ccrs

    ncols = len(plotted_columns)

    fig, axs = plt.subplots(
        3, ncols,
        figsize=(4 * ncols, 8),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
        dpi=100,
        subplot_kw={"projection": ccrs.PlateCarree(205)},
    )

    # --- Data ---
    original_pred_y = original_inference_set.pred_ds
    original_sss = original_inference_set.inference_x.salinity.to_xarray()

    noisy_pred_y = noisy_inference_set.pred_ds
    noisy_sss = noisy_inference_set.inference_x.salinity.to_xarray()

    delta_sss = noisy_sss - original_sss
    delta_pred_y = noisy_pred_y - original_pred_y

    # --- Config per variable ---
    var_config = {
        "salinity": {
            "data": (original_sss, noisy_sss, delta_sss),
            "title": "Salinity",
            "props": dict(vmin=25, vmax=40, cmap="viridis",),
            "cbar_label": "SSS",
        },
        "full_avg": {
            "data": (
                original_pred_y.full_avg,
                noisy_pred_y.full_avg,
                delta_pred_y.full_avg,
            ),
            "title": "Stacked TA",
            "props": dict(vmin=2200, vmax=2850, cmap="Spectral_r"),
            "cbar_label": "TA (µmol/kg)",
        },
        "full_std": {
            "data": (
                original_pred_y.full_std,
                noisy_pred_y.full_std,
                delta_pred_y.full_std,
            ),
            "title": "Stacked TA Std",
            "props": dict(vmin=0, vmax=15, cmap="plasma"),
            "cbar_label": "std TA (µmol/kg)",
        },
        "boosted_avg": {
            "data": (
                original_pred_y.boosted_avg,
                noisy_pred_y.boosted_avg,
                delta_pred_y.boosted_avg,
            ),
            "title": "Boosted TA",
            #"props": dict(vmin=2200, vmax=2450, cmap="Spectral_r"),
            "props": dict(cmap = "Spectral_r"),
            "cbar_label": "TA (µmol/kg)",
        },
        "boosted_std": {
            "data": (
                original_pred_y.boosted_std,
                noisy_pred_y.boosted_std,
                delta_pred_y.boosted_std,
            ),
            "title": "Boosted TA Std",
            "props": dict(vmin=0, vmax=15, cmap="plasma"),
            "cbar_label": "std TA (µmol/kg)",
        },
        "linear_avg": {
            "data": (
                original_pred_y.linear_avg,
                noisy_pred_y.linear_avg,
                delta_pred_y.linear_avg,
            ),
            "title": "Linear TA",
            "props": dict(vmin=2200, vmax=2850, cmap="Spectral_r"),
            "cbar_label": "TA (µmol/kg)",
        },
        "linear_std": {
            "data": (
                original_pred_y.linear_std,
                noisy_pred_y.linear_std,
                delta_pred_y.linear_std,
            ),
            "title": "Linear TA Std",
            "props": dict(vmin=0, vmax=15, cmap="plasma"),
            "cbar_label": "Std TA (µmol/kg)",
        },
    }

    
    # --- Plot loop ---
    ims = []

    for j, var in enumerate(plotted_columns):
        cfg = var_config[var]

        orig, noisy, delta = cfg["data"]
        delta_props = dict(cmap = "RdBu_r")

        im1 = map_variable(orig, axs[0, j], **cfg["props"])
        im2 = map_variable(noisy, axs[1, j], **cfg["props"])
        im3 = map_variable(delta, axs[2, j], **delta_props)
        
        im1.colorbar.set_label(cfg["cbar_label"])
        im3.colorbar.set_label(f"∆ {cfg['cbar_label']}")
        im2.colorbar.set_label(cfg["cbar_label"])

        ims.append(im1)

        # Column title
        axs[0, j].set_title(cfg["title"], fontsize=12)

    # --- Row labels ---
    row_labels = ["Original", "Noisy", "Δ (Noisy - Original)"]
    nrows = 3
    for i, label in enumerate(row_labels):
        fig.text(
            0.01,
            1 - (i + 0.5) / nrows,
            label,
            va="center",
            ha="left",
            fontsize=12,
            rotation=90
        )

    # # --- Colorbars per column ---
    # for j, var in enumerate(plotted_columns):
    #     cfg = var_config[var]
    #     cbar = fig.colorbar(
    #         ims[j],
    #         ax=axs[:, j],
    #         orientation="vertical",
    #         shrink=0.8,
    #     )

    # adjust label for delta row implicitly

    # --- Coastlines ---
    for ax_row in axs:
        for ax in ax_row:
            ax.coastlines(lw=0.5)

    plt.show()
    
    return fig


def plot_distribution_sss_and_noise(inference_set : InferenceSet, fig=None):
    
    sss = inference_set.inference_x['salinity']
    noise = inference_set.noise
    color = inference_set.color
    label = inference_set.label
    linestyle = inference_set.linestyle

    if fig is None:
        fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
    else:
        ax = fig.axes


    # --- Left: salinity distributions ---
    ax[0].hist(
        sss,
        bins=50,
        density=True,
        alpha=0.4,
        label=f"{label}",
        color = color
    )

    sss.plot(kind="kde", ax=ax[0], linewidth=1, label=f"{label} KDE", color= color, linestyle=linestyle)


    ax[0].set_title("Salinity Distribution")
    ax[0].set_xlabel("Salinity")
    ax[0].grid(True)
    ax[0].legend()

    # --- Right: noise distribution ---
    if noise is not None:
        ax[1].hist(
            noise,
            bins=50,
            density=True,
            alpha=0.6,
            label = f"{label}",
            color = color
        )

        #noise.plot(kind="kde", ax=ax[1], linewidth=1, label=f"{label} KDE", color= color)
        # show mean and std explicitly
        mean_noise = np.mean(noise)

        ax[1].axvline(mean_noise, linestyle="--", linewidth=2, label=f"{label} mean={mean_noise:.2f}")

        ax[1].set_title("Noise Distribution")
        ax[1].set_xlabel("Noise")
        ax[1].grid(True)
        ax[1].legend()

    fig.tight_layout()

    return fig


def compare_noise_distributions(inference_set_list):
        
    fig = plot_distribution_sss_and_noise(inference_set_list[0])
    for inference_set in inference_set_list[1:]:
        fig = plot_distribution_sss_and_noise(inference_set, fig=fig)
    
    plt.show();
    
    return fig
    

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

def get_inference_x(inference_time: pd.Timestamp, train_x):
    
    assert isinstance(inference_time, pd.Timestamp), "Time must be a pandas Timestamp"
    inference_x = load_inference_data(train_x, date=inference_time)

    return inference_x

def inference_from_x_data(model, inference_x):
    
    logger.info("Getting inference data...")

    logger.info("Predicting all components for inference data...")
    pred_y = model.predict_components(inference_x.to_dataframe)
    logger.info("Inference completed.")

    ds = pred_y.to_xarray().sortby(["time","lat", "lon"])
    
    return ds

if __name__ == "__main__":
    main()


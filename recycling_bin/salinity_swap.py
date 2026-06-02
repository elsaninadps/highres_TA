import pathlib
import dotenv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import deepcopy
import cartopy.crs as crs
from highres_ta import estimators as models
from highres_ta.preprocessing import load_data, load_config, preprocess_data
from highres_ta.evaluation import scoring, plot_residuals_map, plot_residuals_y, scatter_residuals_vs_feature

ROOT = pathlib.Path(dotenv.find_dotenv("pyproject.toml")).parent

TRAINED_MODEL =   models.BaggingCatBoostResidualRegressor.load(
        ROOT / f"models/{'bagged_catboost_residual_modelstructural_optuna_4.pkl'}"
    )

DATA_FOLDER_NAME = "data/catboost_optuna_4/salinity_swap"


SWAPPED_SALINITY_NAME = "salt_soda"

LOW_CBAR_LIMIT = 10
HIGH_CBAR_LIMIT = 90

# Reset to Matplotlib defaults
matplotlib.rcdefaults()

# Explicitly use the default style
plt.style.use('default')

def main():

    trained_model = TRAINED_MODEL
    swapped_salinity_name = SWAPPED_SALINITY_NAME

    xname_features = trained_model.feature_names_in_.tolist()
    
    original_df = pd.read_parquet(ROOT / f"{DATA_FOLDER_NAME}/{swapped_salinity_name}/original.parquet")
    swapped_df = pd.read_parquet(ROOT / f"{DATA_FOLDER_NAME}/{swapped_salinity_name}/swapped.parquet")
    
    # 5. Generate predictions and residuals on the aligned datasets
    original_pred = trained_model.predict(original_df[xname_features])
    swapped_pred  = trained_model.predict(swapped_df[xname_features])


    original_residuals = original_pred - original_df['talk']
    swapped_residuals  = swapped_pred - swapped_df['talk']

    delta_sal = swapped_df['salinity'] - original_df['salinity']
    delta_pred = swapped_pred - original_pred
    delta_residuals = swapped_residuals - original_residuals
    
    lat = swapped_df.index.get_level_values('lat')
    lon = swapped_df.index.get_level_values('lon')
    
    
    #6. visualization
    
    fig0 = plot_scores(original_df=original_df,
                       original_pred= original_pred,
                       swapped_df= swapped_df,
                       swapped_pred= swapped_pred)
    
    fig1 = plot_salinities_distributions(original_df= original_df, swapped_df= swapped_df, delta_sal= delta_sal)
    fig2 = plot_residuals_distributions(original_residuals=original_residuals, swapped_residuals= swapped_residuals, delta_residuals=delta_residuals)
    

    fig3 = scatter_residuals_vs_true(original_df= original_df, original_pred=original_pred, 
                                     swapped_df=swapped_df, swapped_pred= swapped_pred)
    
    fig4 = scatter_residuals_vs_sal(original_df=original_df, original_residuals=original_residuals,
                                     swapped_df=swapped_df, swapped_residuals=swapped_residuals)
    
    fig5 = compare_deltas_maps(delta_sal=delta_sal, delta_pred= delta_pred, delta_residuals=delta_residuals)
    
    fig6 = compare_salinity_maps(original_df=original_df, swapped_df=swapped_df, delta_sal=delta_sal)
    fig7 = compare_predictions_maps(original_pred=original_pred, 
                                    swapped_pred=swapped_pred, 
                                    delta_pred=delta_pred, 
                                    lat = lat, lon = lon)
    
    fig8 = compare_residuals_maps(original_residuals=original_residuals, 
                                  swapped_residuals = swapped_residuals, 
                                  delta_residuals=delta_residuals, 
                                  lat = lat, lon = lon)
    
    
    from highres_ta.utils import save_figs_to_pdf
    
    save_figs_to_pdf([fig0, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8], filename = f"{ROOT}/outputs/salinity_swaps/salinity_swap_{SWAPPED_SALINITY_NAME}.pdf")
    

def plot_scores( original_df, original_pred, swapped_df, swapped_pred):
    
    swapped_scores = scoring(y_true = swapped_df['talk'], y_pred = swapped_pred).rename(f"{SWAPPED_SALINITY_NAME}")
    original_scores = scoring(y_true = original_df['talk'], y_pred = original_pred).rename("GLODAP")
    scores = pd.concat([original_scores, swapped_scores], axis=1)

    fig0, ax0 = plt.subplots(figsize=(8, 3))
    ax0.axis("off")
    ax0.table(scores.round(3), loc="center", cellLoc="center")
    ax0.set_title("Model Performance Metrics", fontsize=14, fontweight="bold")
    
    return fig0


def plot_salinities_distributions(original_df, swapped_df, delta_sal):
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,10))

    # --- Left: salinity distributions ---
    ax[0].hist(
        original_df["salinity"],
        bins=50,
        density=True,
        alpha=0.4,
        label=f"GLODAP",
        color = "blue"
    )

    #original_df["salinity"].plot(kind="kde", ax=ax[0], linewidth=1, label=f"GLODAP sal KDE", color= 'blue')

    ax[0].hist(
        swapped_df["salinity"],
        bins=50,
        density=True,
        alpha=0.4,
        label=f"{SWAPPED_SALINITY_NAME}",
        color = "red"
    )

    #swapped_df["salinity"].plot(kind="kde", ax=ax[0], linewidth=1, label=f"CCI + SSS KDE", color= 'red')

    ax[0].set_title("Salinity Distribution")
    ax[0].set_xlabel("Salinity")
    ax[0].grid(True)
    ax[0].legend()



    # --- Right: noise distribution ---

    ax[1].hist(
        delta_sal,
        bins=50,
        density=True,
        alpha=0.6
    )

    mean_delta_sal = np.mean(delta_sal)
    med_delta_sal = np.median(delta_sal)

    ax[1].axvline(mean_delta_sal, linewidth=2, label=f"mean={mean_delta_sal:.2f}")
    ax[1].axvline(med_delta_sal, linestyle="--", linewidth=2, label=f"median={med_delta_sal:.2f}")

    ax[1].set_title("Mismatch distribution (∆S = SSS - sal)")
    ax[1].set_xlabel("∆S")
    ax[1].grid(True)
    ax[1].legend()

    fig.tight_layout()
    
    return fig


def plot_residuals_distributions(original_residuals, swapped_residuals, delta_residuals):
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,10))

    # --- Left: salinity distributions ---
    ax[0].hist(
        original_residuals,
        bins=50,
        density=True,
        alpha=0.4,
        label=f"GLODAP",
        color = "blue"
    )

    #original_residuals.plot(kind="kde", ax=ax[0], linewidth=1, label=f"GLODAP KDE", color= blue, linestyle=linestyle)

    ax[0].hist(
        swapped_residuals,
        bins=50,
        density=True,
        alpha=0.4,
        label=f"{SWAPPED_SALINITY_NAME}",
        color = "red"
    )

    #swapped_residuals.plot(kind="kde", ax=ax[0], linewidth=1, label=f"CCI SSS KDE", color= red, linestyle=linestyle)

    ax[0].set_title("Residuals (pred-true) Distribution")
    ax[0].set_xlabel("Residuals")
    ax[0].grid(True)
    ax[0].legend()



    ax[1].hist(
        delta_residuals,
        bins=50,
        density=True,
        alpha=0.6
    )

    mean_delta_res = np.mean(delta_residuals)
    med_delta_res = np.median(delta_residuals)

    ax[1].axvline(mean_delta_res, linewidth=2, label=f"mean={mean_delta_res:.2f}")
    ax[1].axvline(med_delta_res, linestyle="--", linewidth=2, label=f"median={med_delta_res:.2f}")

    ax[1].set_title("Mismatch distribution (∆residuals = $r_{SSS}$ - $r_{GLODAP}$)")
    ax[1].set_xlabel("∆residuals")
    ax[1].grid(True)
    ax[1].legend()

    fig.tight_layout()
    
    return fig


def compare_deltas_maps(delta_sal, delta_pred, delta_residuals):
    
    
    delta_res_kwargs = dict(
        vmin=np.percentile(delta_residuals, LOW_CBAR_LIMIT),
        vmax=np.percentile(delta_residuals, HIGH_CBAR_LIMIT),
    )
    
    delta_sal_kwargs = dict(
        vmin=np.percentile(delta_sal, LOW_CBAR_LIMIT),
        vmax=np.percentile(delta_sal, HIGH_CBAR_LIMIT),
    )
    
    delta_pred_kwargs = dict(
        vmin=np.percentile(delta_pred, LOW_CBAR_LIMIT),
        vmax=np.percentile(delta_pred, HIGH_CBAR_LIMIT),
    )

    
    
    lat = delta_sal.index.get_level_values('lat')
    lon = delta_sal.index.get_level_values('lon')
    
    fig, axes = plt.subplots(
    nrows=3, 
    ncols=1, 
    figsize=(12, 20), 
    subplot_kw={'projection': crs.PlateCarree()}  # This sends the projection to the axes
    )

    plot_residuals_map(delta_sal, lat = lat , lon = lon, ax = axes[0],  **delta_sal_kwargs, title = "∆sal")
    plot_residuals_map(delta_pred, lat = lat , lon = lon, ax = axes[1], **delta_pred_kwargs,  title = "∆ pred")
    plot_residuals_map(delta_residuals, lat = lat , lon = lon, ax = axes[2],**delta_res_kwargs,  title = "∆residuals")
    
    return fig


def compare_salinity_maps(original_df, swapped_df, delta_sal):
    
    
      
    sal_kwargs = dict(vmin=20, vmax = 40)
    
    delta_sal_kwargs = dict(
        vmin=np.percentile(delta_sal, LOW_CBAR_LIMIT),
        vmax=np.percentile(delta_sal, HIGH_CBAR_LIMIT),
    )
    
    lat = swapped_df.index.get_level_values('lat')
    lon = swapped_df.index.get_level_values('lon')


    fig, axes = plt.subplots(
        nrows=3, 
        ncols=1, 
        figsize=(12, 20), 
        subplot_kw={'projection': crs.PlateCarree()}  # This sends the projection to the axes
    )


    plot_residuals_map(original_df['salinity'], lat = lat , lon = lon, ax = axes[0],**sal_kwargs, title = 'GLODAP')
    plot_residuals_map(swapped_df['salinity'], lat = lat , lon = lon, ax = axes[1], **sal_kwargs, title = f"{SWAPPED_SALINITY_NAME}")
    plot_residuals_map(delta_sal, lat = lat , lon = lon, ax = axes[2],  **delta_sal_kwargs, title = "∆sal")


    return fig


def compare_predictions_maps(original_pred, swapped_pred, delta_pred, lat, lon):
    
    
    pred_kwargs = dict(
        vmin= min(np.percentile(original_pred, LOW_CBAR_LIMIT), np.percentile(swapped_pred, LOW_CBAR_LIMIT)),
        vmax= max(np.percentile(original_pred, HIGH_CBAR_LIMIT), np.percentile(swapped_pred, HIGH_CBAR_LIMIT))
    )
    
    delta_pred_kwargs = dict(
        vmin=np.percentile(delta_pred, LOW_CBAR_LIMIT),
        vmax=np.percentile(delta_pred, HIGH_CBAR_LIMIT),
    )
    
    fig, axes = plt.subplots(
    nrows=3, 
    ncols=1, 
    figsize=(12, 20), 
    subplot_kw={'projection': crs.PlateCarree()}  # This sends the projection to the axes
    )



    pred_kwargs = dict(vmin = 2200, vmax = 2500)
    plot_residuals_map(original_pred, lat = lat , lon = lon, ax = axes[0], **pred_kwargs, title = 'GLODAP predictions')
    plot_residuals_map(swapped_pred, lat = lat , lon = lon, ax = axes[1], **pred_kwargs, title = f"{SWAPPED_SALINITY_NAME} predictions")
    plot_residuals_map(delta_pred, lat = lat , lon = lon, ax = axes[2], **delta_pred_kwargs,  title = "∆ pred")
    
    return fig


def compare_residuals_maps(original_residuals, swapped_residuals, delta_residuals, lat, lon):
    
    
    
    
    res_kwargs = dict(
        vmin= min(np.percentile(original_residuals, LOW_CBAR_LIMIT), np.percentile(swapped_residuals, LOW_CBAR_LIMIT)),
        vmax= max(np.percentile(original_residuals, HIGH_CBAR_LIMIT), np.percentile(swapped_residuals, HIGH_CBAR_LIMIT))
    )
    
    
    delta_res_kwargs = dict(
        vmin=np.percentile(delta_residuals, LOW_CBAR_LIMIT),
        vmax=np.percentile(delta_residuals, HIGH_CBAR_LIMIT),
    )
        
    fig, axes = plt.subplots(
    nrows=3, 
    ncols=1, 
    figsize=(12, 20), 
    subplot_kw={'projection': crs.PlateCarree()}  # This sends the projection to the axes
    )


    plot_residuals_map(original_residuals, lat = lat , lon = lon, ax = axes[0], **res_kwargs, title = 'GLODAP residuals')
    plot_residuals_map(swapped_residuals, lat = lat , lon = lon, ax = axes[1],**res_kwargs, title = f"{SWAPPED_SALINITY_NAME} residuals")
    plot_residuals_map(delta_residuals, lat = lat , lon = lon, ax = axes[2],**delta_res_kwargs,  title = "∆residuals")
    
    return fig


def scatter_residuals_vs_true(original_df, original_pred, swapped_df, swapped_pred):

    fig2, axs2 = plt.subplots(
        2, 1, figsize=(12, 7), sharey=True, sharex=True, constrained_layout=True
    )

    plot_residuals_y(original_df['talk'], original_pred, ax=axs2[0], **dict(color='blue'))
    axs2[0].set_title("GLODAP salinity Residuals (Predicted - True)", fontsize=14, fontweight="bold")
    axs2[0].set_xlabel("Observed Total Alkalinity (µmol/kg)")
    axs2[0].set_ylabel("Residual (µmol/kg)")

    plot_residuals_y(swapped_df['talk'], swapped_pred, ax=axs2[1], **dict(color='red'))
    axs2[1].set_title(f"{SWAPPED_SALINITY_NAME} Residuals (Predicted - True)", fontsize=14, fontweight="bold")
    axs2[1].set_xlabel("Observed Total Alkalinity (µmol/kg)")
    axs2[1].set_ylabel("Residual (µmol/kg)")
    axs2[1].set_ylabel("")
    
    return fig2

def scatter_residuals_vs_sal(original_df, original_residuals, swapped_df, swapped_residuals):
    
    fig3, axs3 = plt.subplots(
    2, 1, figsize=(12, 7), sharey=True, sharex=True, constrained_layout=True
    )

    scatter_residuals_vs_feature(residuals = original_residuals, feature = original_df['salinity'], ax=axs3[0], **dict(color='blue'))
    axs3[0].set_title("GLODAP salinity Residuals (Predicted - True)", fontsize=14, fontweight="bold")
    axs3[0].set_xlabel("GLODAP salinity")
    axs3[0].set_ylabel("Residual (µmol/kg)")

    scatter_residuals_vs_feature(residuals = swapped_residuals, feature = swapped_df['salinity'], ax=axs3[1], **dict(color='red'))
    axs3[1].set_title(f"{SWAPPED_SALINITY_NAME} Residuals (Predicted - True)", fontsize=14, fontweight="bold")
    axs3[1].set_xlabel(f"{SWAPPED_SALINITY_NAME}")
    axs3[1].set_ylabel("Residual (µmol/kg)")
    axs3[1].set_ylabel("")
    
    return fig3


if __name__ == "__main__":
    #failsafe_checks()
    main()
import cmocean.cm as cmo
import pathlib
from copy import deepcopy

import cartopy.crs as crs
import dotenv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl


from highres_ta import estimators as models
from highres_ta.evaluation import (
    plot_residuals_map,
    plot_residuals_y,
    scatter_residuals_vs_feature,
    scoring,
)
from highres_ta.preprocessing import (
    load_config,
    load_data,
    preprocess_data,
)

# ============================================================
# RUN CONFIGURATION
# ============================================================

SWAPPED_SALINITY_NAME = "salt_soda"
MODEL_NAME = "bagged_catboost_residual_modelstructural_optuna_4"
WHOLE_DATASET = True  # if False, only evaluate on the GLODAP test set (samples not seen during training)
# ============================================================
# PLOT SETTINGS
# ============================================================

FIG_DPI = 300
LOW_CBAR_LIMIT = 10
HIGH_CBAR_LIMIT = 90

matplotlib.rcdefaults()
plt.style.use("default")

matplotlib.rcParams.update({
    "figure.dpi": FIG_DPI,
    "savefig.dpi": FIG_DPI,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
})

salinity_plot_dict = {
    
    'original':{
        'product_name': 'GLODAP',
        'sal_label' : 'salinity'
    },
    'salt_soda': {
        'product_name': 'SODA',
        'sal_label' : 'salinity'
    },
    'sss_cci': {
        'product_name': 'ESA SSS CCI',
        'sal_label' : 'SSS'
    },
    'sss_glorys': {
        'product_name': 'CMEMS GLORYS',
        'sal_label' : 'SSS'
    },
    'sss_multiobs': {
        'product_name': 'CMEMS Multiobs',
        'sal_label' : 'SSS'
    }
}



# ============================================================
# PATHS
# ============================================================

ROOT = pathlib.Path(
    dotenv.find_dotenv("pyproject.toml")
).parent

# OUTPUT_DIR = (
#     ROOT
#     / f"outputs/salinity_swaps/whole_dataset/{SWAPPED_SALINITY_NAME}"
# )

if WHOLE_DATASET:
    OUTPUT_DIR = (
        ROOT
        / f"outputs/salinity_swaps/whole_dataset/{SWAPPED_SALINITY_NAME}"
    )
else:
    OUTPUT_DIR = (
        ROOT
        / f"outputs/salinity_swaps/test_set/{SWAPPED_SALINITY_NAME}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATA_FOLDER_NAME = f"data/{MODEL_NAME}/salinity_swap"

# ============================================================
# MODEL
# ============================================================


TRAINED_MODEL = (
    models.BaggingCatBoostResidualRegressor.load(
        ROOT
        / f"models/{MODEL_NAME}.pkl"
    )
)


# ============================================================
# PLOT SETTINGS
# ============================================================
FIG_DPI = 300

LOW_CBAR_LIMIT = 10
HIGH_CBAR_LIMIT = 90

matplotlib.rcdefaults()
plt.style.use("default")

matplotlib.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "savefig.bbox": "tight",
    }
)

# ============================================================
# UTILITIES
# ============================================================

def save_figure(fig, name):

    fig.savefig(
        OUTPUT_DIR / f"{name}.png",
        dpi=FIG_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    
    trained_model = TRAINED_MODEL
    swapped_salinity_name = SWAPPED_SALINITY_NAME

    xname_features = trained_model.feature_names_in_.tolist()
    
    if WHOLE_DATASET:
        original_df = pd.read_parquet(ROOT / f"{DATA_FOLDER_NAME}/{swapped_salinity_name}/original_df.pq")
        swapped_df = pd.read_parquet(ROOT / f"{DATA_FOLDER_NAME}/{swapped_salinity_name}/swapped_df.pq")
    else:
        original_df = pd.read_parquet(ROOT / f"{DATA_FOLDER_NAME}/{swapped_salinity_name}/original_test_df.pq")
        swapped_df = pd.read_parquet(ROOT / f"{DATA_FOLDER_NAME}/{swapped_salinity_name}/swapped_test_df.pq")
    
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


    # ========================================================
    # FIGURES
    # ========================================================

    fig0 = plot_scores(
        original_df=original_df,
        original_pred=original_pred,
        swapped_df=swapped_df,
        swapped_pred=swapped_pred,
    )

    fig1 = plot_salinities_distributions(
        original_df=original_df,
        swapped_df=swapped_df,
        delta_sal=delta_sal,
    )

    fig2 = plot_residuals_distributions(
        original_residuals=original_residuals,
        swapped_residuals=swapped_residuals,
        delta_residuals=delta_residuals,
    )

    fig3 = scatter_residuals_vs_true(
        original_df=original_df,
        original_pred=original_pred,
        swapped_df=swapped_df,
        swapped_pred=swapped_pred,
    )

    fig4 = scatter_residuals_vs_sal(
        original_df=original_df,
        original_residuals=original_residuals,
        swapped_df=swapped_df,
        swapped_residuals=swapped_residuals,
    )

    fig5a, fig5b, fig5c = compare_deltas_maps(
        delta_sal=delta_sal,
        delta_pred=delta_pred,
        delta_residuals=delta_residuals,
    )

    fig6a, fig6b = compare_salinity_maps(
        original_df=original_df,
        swapped_df=swapped_df,
    )

    fig7a, fig7b = compare_predictions_maps(
        original_pred=original_pred,
        swapped_pred=swapped_pred,
        lat=lat,
        lon=lon,
    )

    fig8a, fig8b = compare_residuals_maps(
        original_residuals=original_residuals,
        swapped_residuals=swapped_residuals,
        lat=lat,
        lon=lon,
    )

    # ========================================================
    # SAVE FIGURES
    # ========================================================

    if WHOLE_DATASET:
        dataset_str = "whole_dataset"
    else:
        dataset_str = "test_set"
        
    
    save_figure(fig0, f"scores_{dataset_str}")
    save_figure(fig1, f"salinity_distributions_{dataset_str}")
    save_figure(fig2, f"residual_distributions_{dataset_str}")
    save_figure(fig3, f"residuals_vs_true_{dataset_str}")
    save_figure(fig4, f"residuals_vs_salinity_{dataset_str}")

    save_figure(fig5a, f"delta_salinity_map_{dataset_str}")
    save_figure(fig5b, f"delta_prediction_map_{dataset_str}")
    save_figure(fig5c, f"delta_residual_map_{dataset_str}")
    
    save_figure(fig6a, f"glodap_salinity_map_{dataset_str}")
    save_figure(fig6b, f"swapped_salinity_map_{dataset_str}")


    save_figure(fig7a, f"glodap_prediction_map_{dataset_str}")
    save_figure(fig7b, f"swapped_prediction_map_{dataset_str}")


    save_figure(fig8a, f"glodap_residual_map_{dataset_str}")
    save_figure(fig8b, f"swapped_residual_map_{dataset_str}")


    print(f"Figures saved to:\n{OUTPUT_DIR}")


# ============================================================
# SCORE TABLE
# ============================================================


def plot_scores(original_df, original_pred, swapped_df, swapped_pred):

    from datetime import datetime

    swapped_scores = scoring(
        y_true=swapped_df["talk"],
        y_pred=swapped_pred,
    ).rename(f"{salinity_plot_dict[SWAPPED_SALINITY_NAME]['product_name']}")

    original_scores = scoring(
        y_true=original_df["talk"],
        y_pred=original_pred,
    ).rename(f"{salinity_plot_dict['original']['product_name']}")

    metrics_dict = {
        "r2_score": r"$R^2$",
        "mean_absolute_error": r"MAE ($\mu mol\ kg^{-1}$)",
        "root_mean_squared_error": r"RMSE ($\mu mol\ kg^{-1}$)",
        "median_absolute_error": r"MedAE ($\mu mol\ kg^{-1}$)",
        "huber_loss": "Huber",
        "mean_bias": r"Mean bias ($\mu mol\ kg^{-1}$)",
        "median_bias": r"Median bias ($\mu mol\ kg^{-1}$)",
    }

    swapped_scores = swapped_scores.rename(index=metrics_dict)
    original_scores = original_scores.rename(index=metrics_dict)

    swapped_scores = (
        swapped_scores.reset_index(drop=False)
        .rename(columns={"index": "Metric"})
        .set_index("Metric")
    )

    original_scores = (
        original_scores.reset_index(drop=False)
        .rename(columns={"index": "Metric"})
        .set_index("Metric")
    )

    scores = pd.concat([original_scores, swapped_scores], axis=1)

    # ------------------------------------------------------------------
    # Metadata rows (important for comparing runs later)
    # ------------------------------------------------------------------

    evaluation_set = "Whole dataset" if WHOLE_DATASET else "Test set"

    metadata = pd.DataFrame(
        {
            original_scores.columns[0]: [
                len(original_df),
                evaluation_set,
            ],
            swapped_scores.columns[0]: [
                len(swapped_df),
                evaluation_set,
            ],
        },
        index=[
            "N samples",
            "Evaluation set",
        ],
    )

    scores_display = pd.concat([metadata, scores])

    # ------------------------------------------------------------------
    # Save human-readable spreadsheet
    # ------------------------------------------------------------------

    excel_name = (
        "scores_table_whole_dataset.xlsx"
        if WHOLE_DATASET
        else "scores_table_test_set.xlsx"
    )

    scores_display.to_excel(OUTPUT_DIR / excel_name)

    # ------------------------------------------------------------------
    # Save machine-readable run history for experiment comparison
    # ------------------------------------------------------------------

    run_summary = {
        "evaluation_set": evaluation_set,
        "n_samples": len(swapped_df),
        "salinity_product": salinity_plot_dict[SWAPPED_SALINITY_NAME]["product_name"],
    }

    for metric, value in original_scores.iloc[:, 0].items():
        run_summary[f"original_{metric}"] = value

    for metric, value in swapped_scores.iloc[:, 0].items():
        run_summary[f"swapped_{metric}"] = value

    history_path = ROOT / f"outputs/salinity_swaps/ score_history.csv"

    run_df = pd.DataFrame([run_summary])

    if history_path.exists():

        history = pd.read_csv(history_path)

        # remove previous run with same configuration
        history = history[
            ~(
                (history["evaluation_set"] == run_summary["evaluation_set"])
                & (
                    history["salinity_product"]
                    == run_summary["salinity_product"]
                )
            )
        ]

        history = pd.concat([history, run_df], ignore_index=True)

    else:
        history = run_df

    history.to_csv(history_path, index=False)

    # ------------------------------------------------------------------
    # Save publication-quality LaTeX table
    # ------------------------------------------------------------------

    latex_scores = scores.copy().astype(object)

    for idx in latex_scores.index:

        if idx == r"$R^2$":
            latex_scores.loc[idx] = latex_scores.loc[idx].map(lambda x: f"{x:.3f}")
            best_col = scores.loc[idx].idxmax()
        else:
            latex_scores.loc[idx] = latex_scores.loc[idx].map(lambda x: f"{x:.2f}")

            if "bias" in idx.lower():
                best_col = scores.loc[idx].abs().idxmin()
            else:
                best_col = scores.loc[idx].idxmin()

        latex_scores.loc[idx, best_col] = (
            r"\textbf{" + latex_scores.loc[idx, best_col] + "}"
        )

    caption = (
            f"Performance metrics for the GLODAP and {salinity_plot_dict[SWAPPED_SALINITY_NAME]['product_name']} salinity products. Metrics are computed on the "
            f" (GLODAP {evaluation_set.lower()} $\\cap$ {salinity_plot_dict[SWAPPED_SALINITY_NAME]['product_name']}) points ($N={len(swapped_df):,}$ samples)."
        )
        

    latex_table = latex_scores.to_latex(
        escape=False,
        column_format="lcc",
        caption=caption,
        label=f"tab:performance_metrics_{salinity_plot_dict[SWAPPED_SALINITY_NAME]['product_name'].replace(' ', '_').lower()}_{evaluation_set.lower().replace(' ', '_')}",
    )

    latex_name = (
        "performance_metrics_whole_dataset.tex"
        if WHOLE_DATASET
        else "performance_metrics_test_set.tex"
    )

    with open(OUTPUT_DIR / latex_name, "w") as f:
        f.write(latex_table)

    # ------------------------------------------------------------------
    # Quick-look PNG table
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.axis("off")

    table = ax.table(
        cellText=scores_display.round(3).astype(str).values,
        rowLabels=scores_display.index,
        colLabels=scores_display.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    title = (
        rf"Performance Metrics ({evaluation_set}, "
        rf"$N={len(swapped_df):,}$)"
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    fig.tight_layout()

    png_name = (
        "performance_metrics_whole_dataset.png"
        if WHOLE_DATASET
        else "performance_metrics_test_set.png"
    )

    fig.savefig(
        OUTPUT_DIR / png_name,
        dpi=300,
        bbox_inches="tight",
    )

    return fig

# def plot_scores(original_df, original_pred, swapped_df, swapped_pred):

#     swapped_scores = scoring(
#         y_true=swapped_df["talk"],
#         y_pred=swapped_pred,
#     ).rename(f"{salinity_plot_dict[SWAPPED_SALINITY_NAME]['product_name']}")

#     original_scores = scoring(
#         y_true=original_df["talk"],
#         y_pred=original_pred,
#     ).rename(f"{salinity_plot_dict['original']['product_name']}")
    
    
#     metrics_dict = {
#         "r2_score": "R²",
#         "mean_absolute_error": "MAE",
#         "root_mean_squared_error": "RMSE",
#         "median_absolute_error": "MedAE",
#         "huber_loss": "Huber",
#         "mean_bias": "Mean bias",
#         "median_bias": "Median bias",
#     }
        
#     swapped_scores = swapped_scores.rename(index=metrics_dict)
#     swapped_scores = swapped_scores.reset_index(drop=False).rename(columns={"index": "Metric"}).set_index("Metric")
#     original_scores = original_scores.rename(index=metrics_dict)
#     original_scores = original_scores.reset_index(drop=False).rename(columns={"index": "Metric"}).set_index("Metric")

#     scores = pd.concat([original_scores, swapped_scores], axis=1)
    
#     if WHOLE_DATASET:
#         scores.to_excel(OUTPUT_DIR / "scores_table_whole_dataset.xlsx")
#     else:
#         scores.to_excel(OUTPUT_DIR / "scores_table_test_set.xlsx")

#     fig, ax = plt.subplots(figsize=(8, 3))

#     ax.axis("off")
#     ax.table(scores.round(3), loc="center", cellLoc="center", index=True)
#     if WHOLE_DATASET:
#         ax.set_title(rf"Performance Metrics (GLODAP $\bigcap$ {salinity_plot_dict[SWAPPED_SALINITY_NAME]['product_name']})", fontsize=14, fontweight="bold")
#     else:
#         ax.set_title(rf"Performance Metrics (GLODAP test set $\bigcap$ {salinity_plot_dict[SWAPPED_SALINITY_NAME]['product_name']})", fontsize=14, fontweight="bold")

#     fig.tight_layout()

#     return fig


# ============================================================
# DISTRIBUTIONS
# ============================================================

def plot_salinities_distributions(original_df, swapped_df, delta_sal):

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # --------------------------------------------------------

    ax[0].hist(
        original_df["salinity"],
        bins=50,
        density=True,
        alpha=0.4,
        label=salinity_plot_dict['original']['product_name'],
        color="blue",
    )

    ax[0].hist(
        swapped_df["salinity"],
        bins=50,
        density=True,
        alpha=0.4,
        label=salinity_plot_dict[SWAPPED_SALINITY_NAME]['product_name'],
        color="red",
    )

    ax[0].set_title("Salinity Distribution")
    ax[0].set_xlabel("Salinity (ppt)")
    ax[0].grid(True)
    ax[0].legend()

    # --------------------------------------------------------

    ax[1].hist(delta_sal, bins=50, density=True, alpha=0.6, color="green")

    mean_delta_sal = np.mean(delta_sal)
    med_delta_sal = np.median(delta_sal)

    ax[1].axvline(mean_delta_sal, linewidth=2, label=f"mean={mean_delta_sal:.2f}")
    ax[1].axvline(med_delta_sal, linestyle="--", linewidth=2, label=f"median={med_delta_sal:.2f}")

    ax[1].set_title("Mismatch Distribution")
    ax[1].set_xlabel(r"$\Delta S$ (ppt)")
    ax[1].grid(True)
    ax[1].legend()

    fig.tight_layout()

    return fig


def plot_residuals_distributions(original_residuals, swapped_residuals, delta_residuals):

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # --------------------------------------------------------

    ax[0].hist(original_residuals, bins=50, density=True, alpha=0.4, label="GLODAP", color="blue")

    ax[0].hist(
        swapped_residuals,
        bins=50,
        density=True,
        alpha=0.4,
        label=salinity_plot_dict[SWAPPED_SALINITY_NAME]['product_name'],
        color="red",
    )

    ax[0].set_title("Residual Distribution")
    ax[0].set_xlabel("Residuals (µmol/kg)")
    ax[0].grid(True)
    ax[0].legend()

    # --------------------------------------------------------

    ax[1].hist(delta_residuals, bins=50, density=True, alpha=0.6, color="green")

    mean_delta_res = np.mean(delta_residuals)
    med_delta_res = np.median(delta_residuals)

    ax[1].axvline(mean_delta_res, linewidth=2, label=f"mean={mean_delta_res:.2f}")
    ax[1].axvline(med_delta_res, linestyle="--", linewidth=2, label=f"median={med_delta_res:.2f}")

    ax[1].set_title("Residual Difference Distribution")
    ax[1].set_xlabel(r"$\Delta residuals$ (µmol/kg)")
    ax[1].grid(True)
    ax[1].legend()

    fig.tight_layout()

    return fig


# ============================================================
# SCATTER PLOTS
# ============================================================

def scatter_residuals_vs_true(original_df, original_pred, swapped_df, swapped_pred):

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True, constrained_layout=True)

    # --------------------------------------------------------

    plot_residuals_y(original_df["talk"], original_pred, ax=axs[0], color="blue")

    axs[0].set_title("GLODAP Residuals")
    axs[0].set_xlabel("Observed Total Alkalinity (µmol/kg)")
    axs[0].set_ylabel("Residual (µmol/kg)")

    # --------------------------------------------------------

    plot_residuals_y(swapped_df["talk"], swapped_pred, ax=axs[1], color="red")

    axs[1].set_title(f"{salinity_plot_dict[SWAPPED_SALINITY_NAME]['product_name']} Residuals")
    axs[1].set_xlabel("Observed Total Alkalinity (µmol/kg)")

    fig.tight_layout()

    return fig


def scatter_residuals_vs_sal(original_df, original_residuals, swapped_df, swapped_residuals):

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True, constrained_layout=True)

    # --------------------------------------------------------

    scatter_residuals_vs_feature(
        residuals=original_residuals,
        feature=original_df["salinity"],
        ax=axs[0],
        color="blue",
    )

    axs[0].set_title("GLODAP Residuals vs Salinity")
    axs[0].set_xlabel("GLODAP Salinity (ppt)")
    axs[0].set_ylabel("Residual (µmol/kg)")

    # --------------------------------------------------------

    scatter_residuals_vs_feature(
        residuals=swapped_residuals,
        feature=swapped_df["salinity"],
        ax=axs[1],
        color="red",
    )

    axs[1].set_title(f"{salinity_plot_dict[SWAPPED_SALINITY_NAME]['product_name']} Residuals vs Salinity")
    axs[1].set_xlabel(f"{salinity_plot_dict[SWAPPED_SALINITY_NAME]['sal_label']} (ppt)")

    fig.tight_layout()

    return fig


# ============================================================
# COLORMAP UTILITIES
# ============================================================

def symmetric_limits(data, upper=HIGH_CBAR_LIMIT):
    vmax = np.nanpercentile(np.abs(data), upper)
    return -vmax, vmax

from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs


# ============================================================
# MAPS
# ============================================================

def compare_deltas_maps(delta_sal, delta_pred, delta_residuals):

    corr_delta_res = np.corrcoef(delta_sal, delta_residuals)[0, 1]
    corr_delta_pred = np.corrcoef(delta_sal, delta_pred)[0, 1]

    lat = delta_sal.index.get_level_values("lat")
    lon = delta_sal.index.get_level_values("lon")

    delta_sal_vmin, delta_sal_vmax = symmetric_limits(delta_sal)
    delta_pred_vmin, delta_pred_vmax = symmetric_limits(delta_pred)
    delta_res_vmin, delta_res_vmax = symmetric_limits(delta_residuals)

    delta_sal_kwargs = dict(vmin=delta_sal_vmin, vmax=delta_sal_vmax, cmap=cmo.balance)
    delta_pred_kwargs = dict(vmin=delta_pred_vmin, vmax=delta_pred_vmax, cmap=cmo.balance)
    delta_res_kwargs = dict(vmin=delta_res_vmin, vmax=delta_res_vmax, cmap=cmo.balance)

    fig1, ax1 = plt.subplots(
        figsize=(10, 5.5),
        constrained_layout=True,
        subplot_kw={"projection": crs.PlateCarree()},
    )

    plot_residuals_map(
        delta_sal,
        lat=lat,
        lon=lon,
        ax=ax1,
        title=r" $\Delta$ Salinity",
        cbar_label=r"$\Delta$ Salinity [PSU]",
        **delta_sal_kwargs,
    )

    ax1.coastlines(linewidth=0.6)

    gl = ax1.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    fig2, ax2 = plt.subplots(
        figsize=(10, 5.5),
        constrained_layout=True,
        subplot_kw={"projection": crs.PlateCarree()},
    )

    plot_residuals_map(
        delta_pred,
        lat=lat,
        lon=lon,
        ax=ax2,
        title=r" $\Delta$ Predictions",
        cbar_label=r"$\Delta$TA [$\mu mol\ kg^{-1}$]",
        **delta_pred_kwargs,
    )

    ax2.coastlines(linewidth=0.6)

    gl = ax2.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    ax2.text(
        0.94,
        0.02,
        rf"Pearson $\mathbf{{r={corr_delta_pred:.2f}}}$",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig3, ax3 = plt.subplots(
        figsize=(10, 5.5),
        constrained_layout=True,
        subplot_kw={"projection": crs.PlateCarree()},
    )

    plot_residuals_map(
        delta_residuals,
        lat=lat,
        lon=lon,
        ax=ax3,
        title=r" $\Delta$ Residuals",
        cbar_label=r"$\Delta$ Res [$\mu mol\ kg^{-1}$]",
        **delta_res_kwargs,
    )

    ax3.coastlines(linewidth=0.6)

    gl = ax3.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    ax3.text(
        0.94,
        0.02,
        rf"Pearson $\mathbf{{r={corr_delta_res:.2f}}}$",
        transform=ax3.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    return fig1, fig2, fig3


def compare_salinity_maps(original_df, swapped_df):

    lat = swapped_df.index.get_level_values("lat")
    lon = swapped_df.index.get_level_values("lon")

    sal_kwargs = dict(vmin=20, vmax=40, cmap=cmo.haline)

    fig1, ax1 = plt.subplots(
        figsize=(10, 5.5),
        constrained_layout=True,
        subplot_kw={"projection": crs.PlateCarree()},
    )

    plot_residuals_map(
        original_df["salinity"],
        lat=lat,
        lon=lon,
        ax=ax1,
        title=" GLODAP Salinity",
        cbar_label="Salinity [ppt]",
        **sal_kwargs,
    )

    ax1.coastlines(linewidth=0.6)

    gl = ax1.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    fig2, ax2 = plt.subplots(
        figsize=(10, 5.5),
        constrained_layout=True,
        subplot_kw={"projection": crs.PlateCarree()},
    )

    plot_residuals_map(
        swapped_df["salinity"],
        lat=lat,
        lon=lon,
        ax=ax2,
        title=f" {salinity_plot_dict[SWAPPED_SALINITY_NAME]['product_name']}",
        cbar_label="Salinity [ppt]",
        **sal_kwargs,
    )

    ax2.coastlines(linewidth=0.6)

    gl = ax2.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    return fig1, fig2


def compare_predictions_maps(original_pred, swapped_pred, lat, lon):

    pred_kwargs = dict(vmin=2200, vmax=2500, cmap=cmo.thermal)

    fig1, ax1 = plt.subplots(
        figsize=(10, 5.5),
        constrained_layout=True,
        subplot_kw={"projection": crs.PlateCarree()},
    )

    plot_residuals_map(
        original_pred,
        lat=lat,
        lon=lon,
        ax=ax1,
        title="GLODAP Predictions",
        cbar_label=r"TA [$\mu mol\ kg^{-1}$]",
        **pred_kwargs,
    )

    ax1.coastlines(linewidth=0.6)

    gl = ax1.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    fig2, ax2 = plt.subplots(
        figsize=(10, 5.5),
        constrained_layout=True,
        subplot_kw={"projection": crs.PlateCarree()},
    )

    plot_residuals_map(
        swapped_pred,
        lat=lat,
        lon=lon,
        ax=ax2,
        title=f"{salinity_plot_dict[SWAPPED_SALINITY_NAME]['product_name']} Predictions",
        cbar_label=r"TA [$\mu mol\ kg^{-1}$]",
        **pred_kwargs,
    )

    ax2.coastlines(linewidth=0.6)

    gl = ax2.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    return fig1, fig2


def compare_residuals_maps(original_residuals, swapped_residuals, lat, lon):

    residual_vmin = min(
        np.percentile(original_residuals, LOW_CBAR_LIMIT),
        np.percentile(swapped_residuals, LOW_CBAR_LIMIT),
    )

    residual_vmax = max(
        np.percentile(original_residuals, HIGH_CBAR_LIMIT),
        np.percentile(swapped_residuals, HIGH_CBAR_LIMIT),
    )

    res_kwargs = dict(vmin=residual_vmin, vmax=residual_vmax, cmap=cmo.balance)

    fig1, ax1 = plt.subplots(
        figsize=(10, 5.5),
        constrained_layout=True,
        subplot_kw={"projection": crs.PlateCarree()},
    )

    plot_residuals_map(
        original_residuals,
        lat=lat,
        lon=lon,
        ax=ax1,
        title="GLODAP Residuals",
        cbar_label=r"$\Delta$ TA [$\mu mol\ kg^{-1}$]",
        **res_kwargs,
    )

    ax1.coastlines(linewidth=0.6)

    gl = ax1.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    fig2, ax2 = plt.subplots(
        figsize=(10, 5.5),
        constrained_layout=True,
        subplot_kw={"projection": crs.PlateCarree()},
    )

    plot_residuals_map(
        swapped_residuals,
        lat=lat,
        lon=lon,
        ax=ax2,
        title=f" {salinity_plot_dict[SWAPPED_SALINITY_NAME]['product_name']} Residuals",
        cbar_label=r"$\Delta$ TA [$\mu mol\ kg^{-1}$]",
        **res_kwargs,
    )

    ax2.coastlines(linewidth=0.6)

    gl = ax2.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    return fig1, fig2


if __name__ == "__main__":
    main()


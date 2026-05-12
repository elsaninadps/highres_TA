# adapt from chatgpt


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
from sklearn import metrics
import optuna

from model_selection import (
    ModelSelectionConfig,
    get_splits_by_expocode_salinity_bin_based,
    load_config,
    load_data,
    preprocess_data,
)

from scipy.special import huber
from highres_ta import BaggingCatBoostResidualRegressor
from highres_ta import estimators as models

ROOT = pathlib.Path(dotenv.find_dotenv("pyproject.toml")).parent

YNAME_TARGET = 'talk'

FIXED_PARAMS= dict(
    n_jobs=8,
    loss_function="MAE",
)

DEFAULT_PARAMS = dict(
    n_estimators=3, #24
    iterations= 850,
    max_samples=0.66,
    random_strength=1,
    min_data_in_leaf=40,
)

LINEAR_FEATURES = ["salinity"] #, "temperature"]

ALL_FEATURES = [
    "salinity",
    "temperature",
    "bottomdepth",
    "depth",
    "ssh_adt",
    "ssh_sla",
    "chl_globcolour",
    "nitrate",
    "silicate",
    "phosphate",
    "ncoord_a",
    "ncoord_b",
    "ncoord_c"
]


STRUCTURAL_PARAMS = dict(
    polynomial_degree=1,
    linear_features=LINEAR_FEATURES,
)

RUN_NAME = "example_optuna"

# =========================
# MAIN PIPELINE
# =========================
def main():

    # Load data and config
    config = load_config(ROOT / "scripts/cv_example_config.yaml")
    df = prepare_data(config)
    
    
    # Train-test split based on expocode and salinity bins
    train_x, train_y, test_x, test_y = train_test_split(df, config.xname_features, config.yname_target)

    #optuna runs
    best_params, selected_features = run_structural_optuna(df, config, n_trials=500)
    #best_params, selected_features = run_finetuning_optuna(df, config, n_trials=50)


    # keep only best features combination
    config.xname_features = selected_features 
    train_x = train_x[selected_features]
    test_x = test_x[selected_features]

    figs = []

    # train with best parameters
    fig0, fig1, fig2, fig3 = train(
        train_x.copy(), train_y,
        test_x.copy(), test_y,
        best_params
    )
    
    fig4 = table_parameters(best_params, selected_features)

    figs.extend([fig0, fig1, fig2, fig3, fig4])

    bagged_model = models.BaggingCatBoostResidualRegressor.load(
        ROOT / f"models/bagged_catboost_residual_model_{RUN_NAME}.pkl"
    )
    yhat_test = bagged_model.predict(test_x)

    ds = inference(bagged_model, train_x.copy())

    fig4, axs = plot_predictions(ds)
    plot_scores_map(test_x, test_y, yhat_test, ax=axs[4], vmin=-40, vmax=40)
    figs.append(fig4)

    save_figs_to_pdf(figs, filename=f"{ROOT}/outputs/optuna_runs/training_results_{RUN_NAME}.pdf")
    
    
def prepare_data(config):
    
    df_raw = load_data()

    # Preprocess data
    df_raw["lon"] = (df_raw["lon"] - 180) % 360 - 180

    coast_mask = get_coastal_mask()
    selector = df_raw[["lat", "lon"]].reset_index(drop=True).to_xarray()
    df_raw["is_coastal"] = coast_mask.sel(selector, method="nearest", tolerance=0.6)

    df = add_n_coords(df_raw)
    df = preprocess_data(df, config)
    df = filter_outliers(df)
    
    return df
    
    
    
# =========================
# OPTUNA OBJECTIVE
# =========================
def make_finetuning_objective(df, config):
    splits = get_splits_by_expocode_salinity_bin_based(df, n_folds=3)

    def objective(trial):

        # =============================
        # HYPERPARAMETERS
        # =============================
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 3, 20),
            "iterations": trial.suggest_int("iterations", 700, 1000),
            "max_samples": trial.suggest_float("max_samples", 0.5, 0.9),
            "random_strength": trial.suggest_float("random_strength", 0.5, 1.5),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 40, 80),

        }

        params |= STRUCTURAL_PARAMS

        selected_features = config.xname_features.copy()

        scores = cross_validation_optuna(
            trial=trial,
            df = df,
            splits= splits,
            selected_features= selected_features,
            params = params,
            config= config
        )
        
        trial.set_user_attr("selected_features", selected_features)
        trial.set_user_attr("linear_features", params["linear_features"])
        trial.set_user_attr("polynomial_degree", params["polynomial_degree"])
        trial.set_user_attr("params", params)
        trial.set_user_attr("nfeatures", len(selected_features))

        return float(np.mean(scores))
    


    return objective

def plot_optuna_study(study):
    from optuna.visualization.matplotlib import plot_optimization_history
    from optuna.visualization.matplotlib import plot_slice
    from optuna.visualization import plot_parallel_coordinate
    from optuna.visualization import plot_contour
    from optuna.visualization import plot_intermediate_values
    from optuna.visualization import plot_edf
    from optuna.visualization import plot_rank
    from PIL import Image
    import io
    import plotly.graph_objects
    import matplotlib.axes 


    df_trials = study.trials_dataframe()
    df_trials.to_csv(f"{ROOT}/outputs/optuna_runs/optuna_trials_{RUN_NAME}.csv", index=False)
    
    # Trials convergence
    
    fig0, ax0 = plt.subplots(figsize=(12, 8))
    ax0.plot(df_trials.number, df_trials.value)
    ax0.set_ylabel("MAE")
    ax0.set_xlabel("trial number")
    ax0.set_title("Complexity vs MAE")
    
    
    # Complexity plot
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.scatter(x= df_trials.user_attrs_nfeatures, y= df_trials.value)
    ax1.set_ylabel("MAE")
    ax1.set_xlabel("number of features")
    ax1.set_title("Complexity vs MAE")
    
    fig2 = plot_binary_classification_map(df_trials)
    
    
    figs = [fig0, fig1, fig2]

    for plot_fn in [
        plot_contour,
        plot_parallel_coordinate,
    ]:
        
        logger.info(f"Plot {plot_fn}")
        pfig = plot_fn(study)  # Plotly figure
         
        if isinstance(pfig, plotly.graph_objects.Figure):
            # Convert Plotly figure -> Matplotlib figure
            img_bytes = pfig.to_image(format="png", scale=2)

            img = Image.open(io.BytesIO(img_bytes))

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{plot_fn}")
            
            figs.append(fig)


    save_figs_to_pdf(
        figs,
        filename=f"{ROOT}/outputs/optuna_runs/optuna_study_{RUN_NAME}.pdf"
    )





import numpy as np
import matplotlib.pyplot as plt


def plot_binary_classification_map(df):
    """
    Visualize binary feature combinations using separating segments.
    """

    # ------------------------------------------------------------
    # Extract binary feature columns
    # ------------------------------------------------------------
    
    
    score_col="value"
    feature_prefix="params_"
    point_size=80

    
    feature_cols = [
        c for c in df.columns
        if c.startswith(feature_prefix)
    ]

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found")

    # ------------------------------------------------------------
    # Convert binary values to +/-1
    # ------------------------------------------------------------
    X = (
        df[feature_cols]
        .replace({True: 1, False: -1})
        .fillna(-1)
        .astype(float)
    )

    # ------------------------------------------------------------
    # Create one direction vector per feature
    # ------------------------------------------------------------
    n_features = len(feature_cols)

    angles = np.linspace(
        0,
        2 * np.pi,
        n_features,
        endpoint=False,
    )

    vectors = np.column_stack([
        np.cos(angles),
        np.sin(angles),
    ])

    # ------------------------------------------------------------
    # Compute point positions
    # ------------------------------------------------------------
    embedding = X.values @ vectors

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))

    # ------------------------------------------------------------
    # Draw separating segments
    # ------------------------------------------------------------
    segment_length = np.max(np.abs(embedding)) * 1.3

    for feature, angle in zip(feature_cols, angles):

        # Direction vector
        dx = np.cos(angle)
        dy = np.sin(angle)

        # Perpendicular vector defines separating segment
        px = -dy
        py = dx

        x1 = -px * segment_length
        y1 = -py * segment_length

        x2 = px * segment_length
        y2 = py * segment_length

        ax.plot(
            [x1, x2],
            [y1, y2],
            linestyle="--",
            alpha=0.4,
        )

        # Labels
        clean_name = feature.replace(feature_prefix, "")

        ax.text(
            dx * segment_length * 1.1,
            dy * segment_length * 1.1,
            f"{clean_name}=True",
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
        )

        ax.text(
            -dx * segment_length * 1.1,
            -dy * segment_length * 1.1,
            f"{clean_name}=False",
            ha="center",
            va="center",
            fontsize=9,
            alpha=0.6,
        )

    # ------------------------------------------------------------
    # Plot points
    # ------------------------------------------------------------
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=df[score_col],
        cmap="coolwarm",
        s=point_size,
        edgecolors="black",
        alpha=0.85,
    )

    # ------------------------------------------------------------
    # Cosmetics
    # ------------------------------------------------------------
    ax.axhline(0, linewidth=0.5, alpha=0.3)
    ax.axvline(0, linewidth=0.5, alpha=0.3)

    ax.set_aspect("equal")

    fig.colorbar(scatter, label=score_col)

    ax.set_title("Binary Classification Map")

    fig.tight_layout()

    return fig


def plot_params_umap(study):

    import umap

    # -----------------------------------
    # Load Optuna trials dataframe
    # -----------------------------------
    df_trials = study.trials_dataframe()

    # -----------------------------------
    # Keep only parameter columns
    # (Optuna names them like params_xxx)
    # -----------------------------------
    feature_cols = [c for c in df_trials.columns if c.startswith("params_")]

    # -----------------------------------
    # Convert everything to binary
    # True  -> 1
    # False -> 0
    # -----------------------------------
    X = (
        df_trials[feature_cols]
        .replace({True: 1, False: 0})
        .fillna(0)
        .astype(float)
    )

    # -----------------------------------
    # Score column
    # (Optuna objective value)
    # -----------------------------------
    scores = df_trials["value"]

    # -----------------------------------
    # Compute 2D embedding
    # -----------------------------------
    embedding = umap.UMAP(
        n_neighbors=10,
        min_dist=0.2,
        random_state=42
    ).fit_transform(X)

    # -----------------------------------
    # Plot
    # -----------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=scores,
        cmap="coolwarm",
        s=80,
        alpha=0.8
    )

    fig.colorbar(scatter, label="Score")

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Optuna Trials Grouped by Binary Features")

    ax.grid(alpha=0.2)
    fig.tight_layout()
    
    return fig

def cross_validation_optuna(trial, df, splits, selected_features, params, config):
    
    # =============================
    # CROSS-VALIDATION
    # =============================
    scores = []

    for fold_idx, (itrain, itest) in enumerate(splits):

        train_df = df.iloc[itrain]
        test_df = df.iloc[itest]

        train_x = train_df[selected_features].copy()
        train_y = train_df[config.yname_target]

        test_x = test_df[selected_features].copy()
        test_y = test_df[config.yname_target]

        model = models.BaggingCatBoostResidualRegressor(**params)
        logger.info(f"Trial {trial.number}, fold {fold_idx}")
        model.fit(train_x, train_y)

        yhat = model.predict(test_x)

        score = metrics.mean_absolute_error(test_y, yhat)

        scores.append(score)

        # pruning
        trial.report(np.mean(scores), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return scores

def make_structural_objective(df, config):
    splits = get_splits_by_expocode_salinity_bin_based(df, n_folds=3)

    def objective(trial):

        params = DEFAULT_PARAMS # default hyperparams
        params |= FIXED_PARAMS #fixed njobs and MAE loss
        


        # =============================
        # FEATURE SELECTION

        base_features = config.xname_features
        selected_features = ["salinity", "temperature"] # always include salinity and temp
        #, "ncoord_a", "ncoord_b", "phosphate", "ssh_adt"]

        for f in base_features:
            if f in selected_features:
                continue

            use_f = trial.suggest_categorical(f"use_{f}", [True, False])
            if use_f:
                selected_features.append(f)

        # =============================
        # LINEAR FEATURE SELECTION
        
        linear_features = ["salinity"] # salinity ALWAYS included

        # optionally include temperature (if present)
        use_temp_linear = trial.suggest_categorical("use_temp_linear", [True, False])
        if use_temp_linear:
            linear_features.append("temperature")
        params["linear_features"] = linear_features

        # =============================
        # POLYNOMIAL DEGREE SELECTION

        params["polynomial_degree"] = trial.suggest_categorical("polynomial_degree", [1, 2])

        # to retrieve them afterwards
        trial.set_user_attr("selected_features", selected_features)
        trial.set_user_attr("linear_features", linear_features)
        trial.set_user_attr("polynomial_degree", params["polynomial_degree"])
        trial.set_user_attr("params", params)
        trial.set_user_attr("nfeatures", len(selected_features))


        
        scores = cross_validation_optuna(
            trial=trial,
            df = df,
            splits= splits,
            selected_features= selected_features,
            params = params,
            config= config
        )

        return float(np.mean(scores))

    return objective



def run_structural_optuna(df, config, n_trials=30):
    study = optuna.create_study(
        study_name= f"{RUN_NAME}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )

    objective = make_structural_objective(df, config)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_trial = study.best_trial

    df_trials = study.trials_dataframe()
    df_trials.to_csv(f"{ROOT}/outputs/optuna_runs/optuna_trials_{RUN_NAME}.csv", index=False)

    params = DEFAULT_PARAMS.copy()
    params |= FIXED_PARAMS
    
    # recover linear features
    linear_features = best_trial.user_attrs["linear_features"]
    params["linear_features"] = linear_features
    
    # recover polynomial degree
    params["polynomial_degree"] = best_trial.params["polynomial_degree"]
    
    # recover selected features
    selected_features = best_trial.user_attrs["selected_features"]
    
    plot_optuna_study(study)
    
    logger.success(f"Finished optuna run: Best score: {study.best_value} with structural params: {study.best_params} ")

    return params, selected_features


def run_finetuning_optuna(df, config, n_trials=30):
    
    study = optuna.create_study(
        study_name=f"{RUN_NAME}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )

    objective = make_finetuning_objective(df, config)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_trial = study.best_trial


    params = study.best_params
    params |= STRUCTURAL_PARAMS
    params |= FIXED_PARAMS
    
    selected_features = config.xname_features
    
    logger.info(f"Best score: {study.best_value}")
    logger.info(f"Best params: {study.best_params}")

    return params, selected_features

def train(train_x, train_y, test_x, test_y, params):

    tr_coast = train_x.index.get_level_values("is_coastal").astype(bool)
    te_coast = test_x.index.get_level_values("is_coastal").astype(bool)

    bagged_model = models.BaggingCatBoostResidualRegressor(**params)

    logger.info(f"Fitting model with columns: {train_x.columns.tolist()}")
    logger.info(f"Fitting model with parameters: {params}")
    
    bagged_model.fit(train_x, train_y)
    bagged_model.save(ROOT / f"models/bagged_catboost_residual_model_{RUN_NAME}.pkl")

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


# =========================
# REMAINING FUNCTIONS (UNCHANGED)
# =========================


def table_parameters(best_params: dict, selected_features: list):

    # -----------------------------
    # Convert to DataFrames
    # -----------------------------
    params_df = pd.DataFrame(
        list(best_params.items()),
        columns=["Parameter", "Value"]
    )

    features_df = pd.DataFrame(
        selected_features,
        columns=["Selected Features"]
    )

    # -----------------------------
    # Create Figure
    # -----------------------------
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 6)
    )

    # Remove axes
    for ax in axes:
        ax.axis("off")

    # -----------------------------
    # Left table: Best Parameters
    # -----------------------------
    table1 = axes[0].table(
        cellText=params_df.values,
        colLabels=params_df.columns,
        loc="center",
        cellLoc="center"
    )

    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1.2, 1.5)

    axes[0].set_title(
        "Best Hyperparameters",
        fontsize=14,
        fontweight="bold"
    )

    # -----------------------------
    # Right table: Selected Features
    # -----------------------------
    table2 = axes[1].table(
        cellText=features_df.values,
        colLabels=features_df.columns,
        loc="center",
        cellLoc="center"
    )

    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 1.5)

    axes[1].set_title(
        "Selected Features",
        fontsize=14,
        fontweight="bold"
    )
    
    fig.tight_layout()

    return fig


def save_figs_to_pdf(figs: list[plt.Figure], filename=f"training_results_{RUN_NAME}.pdf", **props):
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(filename) as pdf:
        for fig in figs:
            props = dict(dpi=300, bbox_inches="tight") | props
            pdf.savefig(fig, **props)

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
    df: pd.DataFrame, xname_features, yname_target
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    itrain, itest = get_splits_by_expocode_salinity_bin_based(df, n_folds=7)[1]
    train_x = df.iloc[itrain][xname_features]
    train_y = df.iloc[itrain][yname_target]
    test_x = df.iloc[itest][xname_features]
    test_y = df.iloc[itest][yname_target]
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
    main()
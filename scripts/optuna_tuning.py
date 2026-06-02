import pathlib
from functools import lru_cache
import dotenv
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from sklearn import metrics
import optuna


from highres_ta import estimators as models
from highres_ta.inference import inference
from highres_ta.utils import save_figs_to_pdf
from highres_ta.evaluation import scoring
from bagged_regressors_experiments import (
    plot_predictions,
    plot_scores_map,
    train,
)
from highres_ta.preprocessing import (
    load_config, 
    load_data, 
    preprocess_data, 
    train_test_split, 
    add_coastal_flag, 
    get_splits_by_expocode_salinity_bin_based
)

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

# =========================
# MAIN PIPELINE
# =========================
def main():
    
    config = load_config(fname_config_yaml="example_config.yaml")
    global RUN_NAME
    RUN_NAME = config.run_name
    
    logger.info(f"Starting run: {RUN_NAME}")
    
    df = load_data()
    df = preprocess_data(df, config)
    df = add_coastal_flag(df)
    train_x, train_y, test_x, test_y = train_test_split(df, config)
    

    #optuna runs
    best_params, selected_features = run_structural_optuna(df, config, n_trials=5)
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
        model_params=best_params,
        run_name = RUN_NAME
    )
    
    fig4 = table_parameters(best_params, selected_features)

    figs.extend([fig0, fig1, fig2, fig3, fig4])

    bagged_model = models.BaggingCatBoostResidualRegressor.load(
        ROOT / f"models/{RUN_NAME}.pkl"
    )
    yhat_test = bagged_model.predict(test_x)

    ds = inference(bagged_model, train_x.copy())

    fig4, axs = plot_predictions(ds)
    plot_scores_map(test_x, test_y, yhat_test, ax=axs[4], vmin=-40, vmax=40)
    figs.append(fig4)

    save_figs_to_pdf(figs, filename=f"{ROOT}/outputs/optuna_runs/training_results_{RUN_NAME}.pdf")
    
    
    
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

# =========================
# PLOTTINING FUNCTIONS 
# =========================


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

if __name__ == "__main__":
    main()
    
    
#TODO: use built in optuna visualization tools, maybe transform this to a notebook
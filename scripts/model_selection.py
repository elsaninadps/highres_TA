import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Type
import yaml

import catboost
import dotenv
import joblib
import lightgbm
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
import matplotlib.pyplot as plt

ModelSklearnAPI = (
    Type[RandomForestRegressor] | Type[catboost.CatBoostRegressor] | Type[lightgbm.LGBMRegressor] | Type[LinearRegression]
)

CONFIG_FNAME = "cv_config_coarse_search.yaml"

LOGGING_LEVEL = "INFO"
CV_VERBOSITY = 3
N_CPUS = -1  # -1 = all available CPUs

ROOT = Path(dotenv.find_dotenv("pyproject.toml")).parent
DATA_PATH = ROOT / "data/training/GLODAPv2023-raw_collocated-{y}.pq"
SAVE_PATH = ROOT/ "outputs/model_selection"


CV_SCORING_METRICS = [
    "neg_root_mean_squared_error",
    "neg_median_absolute_error",
    "neg_mean_absolute_error",
    "r2",
]

NFOLDS = 0  # will be set later according to the config

METRICS_PLOT_LABELS = {
    "neg_root_mean_squared_error": "Neg RMSE",  
    "neg_median_absolute_error": "Neg MedAE",
    "neg_mean_absolute_error": "Neg MAE",
    "r2": "R²",
    "fit_time": "Fit Time (s)",
    "score_time": "Score Time (s)"
}

DF_INDEX_COLUMNS = (
    "expocode",
    "time",
    "lat",
    "lon",
)

COMPULSORY_COLUMNS = {
    "talk",
    "salinity",
} | set(DF_INDEX_COLUMNS)

SALINITY_BIN_EDGES = (
    0,
    32,
    34,
    36,
    np.inf,
)

logger.remove()
logger.add(sys.stderr, level=LOGGING_LEVEL)

ESTIMATOR_NAMES = Literal[ "RandomForest", "CatBoost", "LightGBM", "LinearRegression" ]
ESTIMATORS = {
    "RandomForest": RandomForestRegressor,
    "CatBoost": catboost.CatBoostRegressor,
    "LightGBM": lightgbm.LGBMRegressor,
    "LinearRegression": LinearRegression
}

ESTIMATORS_COLORS = {
    "RandomForestRegressor": 'blue', 
    "CatBoostRegressor": 'orange',
    "LGBMRegressor": 'green',
    "LinearRegression": 'red'
}

@dataclass
class ModelCVParams:
    model_name: ESTIMATOR_NAMES
    param_grid: list[dict[str, object]]
    model: ModelSklearnAPI
    default_kwargs: dict[str, object] = field(default_factory=dict)


@dataclass
class ModelSelectionConfig:
    fname_data_parquet: str | Path
    yname_target: Literal["talk", "talk_normalized"]
    xname_features: list[str]
    num_cv_folds: int
    num_best_models_cohort: int
    save_path: str | Path
    params: list[ModelCVParams]
    salinity_bins: tuple[float, ...] = SALINITY_BIN_EDGES
    salinity_name: str = "salinity"
    salinity_norm_value: float = 34.5
    
    
def main():
    
    config_path = ROOT / f"scripts/{CONFIG_FNAME}"
    config = load_config(config_path)
    data_raw = load_data()
    
    # IDEA: consider removing outliers or computing a weighting for these outliers
    data = preprocess_data(data_raw, config)

    # NOTE: This is a bit messy and could be neater in a function, but for now, OK
    
    train_df, test_df = get_train_test_split_by_expocode_salinity_bin_based(data)
    cv_splitter = get_splits_by_expocode_salinity_bin_based(train_df) 
    
    train_x = train_df.drop(columns=[config.yname_target])
    train_y = train_df[config.yname_target]
    
    test_x = test_df.drop(columns=[config.yname_target])
    test_y = test_df[config.yname_target]

    cv_models = ()
    cv_results = ()
    best_cv_results = ()
    models_labels = [model_cv_params.model_name for model_cv_params in config.params]
    logger.info(f"Models to train: {models_labels}")
    
    for model_cv_params in config.params:
        
        cv_model = train_model(
            train_x,
            train_y,
            splitter=cv_splitter,  # convert to list, so can be saved with pickle later
            model_cv_params=model_cv_params,
            n_jobs=N_CPUS,
            verbose=CV_VERBOSITY,
        )


        save_cv_model(cv_model, model_cv_params.model_name)
        cv_result = extract_cv_results(cv_model)
        
        cv_models += (cv_model,)
        cv_results += (cv_result,)
        best_cv_results += (extract_best_model_cohort(cv_result, ranks_to_select=config.num_best_models_cohort),)
        
        fit_best_estimator_on_test(cv_model, test_x, test_y)
        
    boxplot_scores_distribution(cv_results, best_cv_results) 

    cv_results_combined = combine_cv_results(cv_results)
    best_cv_results_combined = combine_best_cv_results(best_cv_results)
    
    extract_best_params_tables(best_cv_results_combined)
    best_scores_stability_plot(best_cv_results_combined)   
    publish_best_cohort_mean_scores(best_cv_results_combined)
    publish_best_params_tables(best_cv_results_combined)
    
    # what do we do with the results
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html


def load_config(fname_config_yaml: str | Path) -> ModelSelectionConfig:
    global SAVE_PATH, NFOLDS

    with open(fname_config_yaml, "r") as f:
        config_dict = yaml.safe_load(f)

    config = ModelSelectionConfig(**config_dict)

    if not config.params:
        raise ValueError("No model parameters provided in the config file")
    else:
        for i, model_cv_params in enumerate(config.params):
            model_cv_params["model"] = ESTIMATORS[model_cv_params["model_name"]]  # type: ignore
            config.params[i] = ModelCVParams(**model_cv_params)  # type: ignore

    SAVE_PATH = ROOT / config.save_path
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    NFOLDS = config.num_cv_folds
    
    return config


def load_data(compulsory_columns: set[str] = COMPULSORY_COLUMNS) -> pd.DataFrame:
    
    """
    Loads the training data from a parquet file

    Contains all the columns and rows of the training data
    No engineering or preprocessing is done here, just loading
    the data into a pandas dataframe

    Parameters
    ----------
    fname_data_parquet : str | Path
        The path to the parquet file containing the training data
    Returns
    -------
    pd.DataFrame
        The training data as a pandas dataframe
    """
    
    data_path = str(DATA_PATH)

    logger.info(f"Loading data from {data_path.format(y='YYYY')} for years 1982-2021")
    data = pd.concat([pd.read_parquet(data_path.format(y=y)) for y in range(1982, 2022)])

    # Check that the compulsory columns are present in the data
    columns = data.columns.intersection(compulsory_columns)

    if len(columns) < len(compulsory_columns):
        missing_cols = compulsory_columns - set(columns)
        raise ValueError(f"Missing columns in the data: {missing_cols}")

    logger.debug(f"data.head() = \n{data.head().T.head(50)}")
    return data


def preprocess_data(df: pd.DataFrame, config: ModelSelectionConfig) -> pd.DataFrame:
    
    """
    Select columns, engineer features, bin salinity

    Parameters
    ----------
    df : pd.DataFrame
        The raw training df as a pandas dataframe
    config : ModelSelectionConfig
        The configuration for the model selection process, containing any parameters needed for preprocessing

    Returns
    -------
    pd.DataFrame
        The preprocessed training df
    """

    # salinity binning
    salinity_bins = config.salinity_bins
    salinity = df[config.salinity_name]
    df["salinity_bin"] = salinity_binning(salinity, bins=salinity_bins)
    
    # alkalinity normalization
    salt_norm_value = config.salinity_norm_value
    df["talk_normalized"] = normalize_alkalinity(df["talk"], salinity, salt_norm_value)
    
    # filter outliers
    df = filter_outliers(df, column="salinity", lower_abs=20, upper_abs=40)

    # set quadruple index 
    index_columns = DF_INDEX_COLUMNS
    index_columns = list(DF_INDEX_COLUMNS + ("salinity_bin",))
    df = df.set_index(index_columns)

    # select columns and drop rows with missing values in the selected columns
    keep_cols = set(config.xname_features + [config.yname_target]).union(COMPULSORY_COLUMNS)
    valid_columns = list(keep_cols - set(index_columns))
    df = df[valid_columns].dropna()

    logger.debug(f"Preprocessed data head: \n{df.head()}")

    return df


def filter_outliers(df: pd.DataFrame, column: str, lower_abs: float, upper_abs: float) -> pd.DataFrame:
    if lower_abs is not None:
        df = df[df[column] >= lower_abs]
    if upper_abs is not None:
        df = df[df[column] <= upper_abs]

    return df


def salinity_binning(
    salinity: pd.Series, bins: tuple[float, ...], bin_labels: None | list[str | float] = None
) -> pd.Series:
    n_bins = len(bins)
    bin_label = bin_labels or range(1, n_bins)
    return pd.cut(salinity, bins=bins, labels=bin_label)


def normalize_alkalinity(
    alkalinity: pd.Series, salinity: pd.Series, norm_value: float
) -> pd.Series:
    return norm_value * alkalinity / salinity


def get_splits_by_expocode_salinity_bin_based(
    data: pd.DataFrame, random_state: int = 42, n_folds: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    
    index = data.index.to_frame()

    grouper = index["expocode"]
    stratifier = index["salinity_bin"]

    #TODO:verify how the shuffle affects stratification and grouping
    # PROBLEM: default random_state is 42, so shuffle will always be True.
    
    shuffle = False if random_state is None else True  # if random state provided, then True
    #splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    splitter = StratifiedGroupKFold(n_splits=n_folds)

    splits = splitter.split(data, y=stratifier, groups=grouper)

    # return a list so that we can pickle the CV splitter later
    return list(splits)

def get_train_test_split_by_expocode_salinity_bin_based(
    data: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Generate a 5-fold StratifiedGroupKFold, yielding a 20% validation and 80% train distribution in each of the 5 folds
    train_test_splitter = get_splits_by_expocode_salinity_bin_based(data, random_state=random_state)
    
    # only take the first fold instance to generate our train-test split according to its train-validation (80:20) distribution 
    idx_train, idx_test = train_test_splitter[0]
    
    # generate train and test dfs according to the generated indices
    train = data.iloc[idx_train]
    test = data.iloc[idx_test]

    return train, test


def train_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    splitter: Iterable[tuple[np.ndarray, np.ndarray]],
    model_cv_params: ModelCVParams,
    **kwargs,
) -> GridSearchCV:
    
    logger.info(f"Training model with parameters: \n{model_cv_params}")
    logger.debug(f"Training data shapes: train_x = {train_x.shape}, train_y = {train_y.shape}")
    logger.debug(f"Splitter: \n{splitter}")

    props = {
        "scoring": CV_SCORING_METRICS,
        "refit": CV_SCORING_METRICS[0],
    } | kwargs
    # verbose is int otherwise, LightGBM fail
    #estimator = model_cv_params.model(verbose=0, **model_cv_params.default_kwargs)
    default_params = model_cv_params.default_kwargs | {"verbose": 0}
    if model_cv_params.model_name == "LinearRegression":
        default_params.pop("verbose")
    estimator = model_cv_params.model(**default_params)

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=model_cv_params.param_grid,
        cv=splitter,
        **props,
    )
    
    grid_search.fit(train_x, train_y)

    logger.success(f"Finished training model {model_cv_params.model_name}")

    return grid_search


def save_cv_model(cv_model: GridSearchCV, model_name: str):

    #save_models_path = ROOT / f"models/
    
    save_models_path = SAVE_PATH / f"models/cv_{model_name}.joblib"
    Path(save_models_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(cv_model, save_models_path, compress=1)

    logger.success(f"Saved CV model to {save_models_path}")


def extract_cv_results(cv_model: GridSearchCV) -> pd.DataFrame:
    results = pd.DataFrame(cv_model.cv_results_)
    model_name = cv_model.estimator.__class__.__name__    
    results["model_name"] = model_name
    logger.debug(f"Extracted CV results: \n{results.head(15).T}")
    return results

def extract_best_model_cohort(model_cv_result, ranks_to_select: int = 1) -> pd.DataFrame:
    primary_metric = CV_SCORING_METRICS[0]

    # Sort by primary metric rank (ascending, smaller rank is better), then by mean_score_time (ascending, smaller is better),
    # then by mean_fit_time (ascending, smaller is better) to break ties
    sorted_results = model_cv_result.sort_values(
        by=[f"rank_test_{primary_metric}", "mean_score_time", "mean_fit_time"],
        ascending=[True, True, True]
    )

    best_indexes = sorted_results.head(ranks_to_select).index
    best_cohort = model_cv_result.loc[best_indexes].reset_index(drop=True)
    
    best_cohort['submodel_name'] = best_cohort['model_name'] + "_" + best_cohort[f'rank_test_{primary_metric}'].astype(str)
    best_cohort = best_cohort.set_index("submodel_name") 
    best_cohort['rank_in_cohort'] = best_cohort[f'rank_test_{primary_metric}'] 

    logger.debug(f"Selected {ranks_to_select} best CV cohort for model {model_cv_result['model_name'].iloc[0]}: \n{best_cohort.T.to_markdown()}")

    return best_cohort

def combine_cv_results(cv_results: Iterable[pd.DataFrame]) -> pd.DataFrame:
    combined_results = pd.concat(cv_results, ignore_index=True)
    logger.debug(f"Combined CV results: \n{combined_results.head(15).T}")
    return combined_results
    
def combine_best_cv_results(best_cv_results: Iterable[pd.DataFrame]) -> pd.DataFrame:
    combined_best_results = pd.concat(best_cv_results, ignore_index=False)
    logger.info(f"Combined best CV results: \n{combined_best_results.T.to_markdown()}")
    return combined_best_results

def failsafe_checks():

    estimator_names = set(ESTIMATOR_NAMES.__args__)
    estimator_keys = set(ESTIMATORS.keys())

    assert estimator_names == estimator_keys, (
        f"ESTIMATOR_NAMES and ESTIMATORS keys must match, but got {estimator_names} and {estimator_keys}"
    )

def boxplot_scores_distribution(cv_results, best_cohort_results, title = 'CV results boxplot comparison'):
    
    nfolds = NFOLDS
    nplots = len(CV_SCORING_METRICS)
    fig, axes = plt.subplots(nplots, 1, figsize=(20, 4*nplots))
    
    for i, metric in enumerate(CV_SCORING_METRICS):   
        
        boxplot_list = []
        labels_list = []
        
        for model_results, best_cohort_result in zip(cv_results, best_cohort_results):
        
            scores_per_metric = concatenate_all_folds_scores_per_metric(model_results, metric) 
            model_label= model_results['model_name'].iloc[0]
            best_scores_per_metric =concatenate_all_folds_scores_per_metric(best_cohort_result, metric)
            best_cohort_label = f"Best {best_cohort_result['model_name'].iloc[0]} cohort"
            
            labels_list.append(model_label)
            labels_list.append(best_cohort_label)
            
            boxplot_list.append(scores_per_metric)
            boxplot_list.append(best_scores_per_metric)
            
        axes[i].boxplot(boxplot_list, labels=labels_list)
        axes[i].set_ylabel(metric)
    
    plt.tight_layout()
    fig.savefig(SAVE_PATH / f"{title.replace(' ', '_')}.png")
    
def best_scores_stability_plot(best_cv_results, title = 'Best models cohort CV results stability'):
    
    nfolds = NFOLDS
    nplots = len(CV_SCORING_METRICS)
    fig, axes = plt.subplots(nplots, 1, figsize=(10, 3*nplots))
    axes = axes.flatten()

    folds = np.linspace(0, nfolds-1, nfolds).astype(int)
    
    rank_marker = {'1': 's', '2': 'o', '3': 'D', '4': 'x', '5': '*'}  # different marker for each rank in cohort
    selected_ranks = best_cv_results.rank_in_cohort.unique()

    for model in list(best_cv_results.model_name.unique()):
        
        color = ESTIMATORS_COLORS[model]
            
        for kth_rank in selected_ranks:
            
            kth_best_cv_result = best_cv_results[(best_cv_results.model_name == model) & (best_cv_results.rank_in_cohort == kth_rank)]
            marker_type = rank_marker[str(kth_rank)]
    
            for i, metric in enumerate(CV_SCORING_METRICS):
                
                logger.debug(f"{kth_best_cv_result.T.to_markdown()}")  # debug print the selected best CV cohort for this model and rank
                kth_scores_per_metric = [kth_best_cv_result[f"split{y}_test_{metric}"] for y in range(0, nfolds)]
                kth_scores_per_metric = pd.DataFrame(kth_scores_per_metric, index=folds)  
                kth_scores_per_metric.index.name = 'fold'
                
                
                kth_scores_per_metric.plot(marker=marker_type, color=color, linestyle=':', ax=axes[i], legend=False)
                #axes[i].scatter(x = folds, y = kth_scores_per_metric, marker=marker_type, label = label, color=color)
                #axes[i].axhline(y = best_cv_results.loc[model, f"mean_test_{metric}"], linestyle='--', color =color,label = f"{model} mean")
                axes[i].set_ylabel(metric)
                axes[i].set_xlabel('fold')
                axes[i].legend()
        
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(SAVE_PATH / f"{title.replace(' ', '_')}.png")
            
            
def concatenate_all_folds_scores_per_metric(model_cv_result, metric):
    nfolds = NFOLDS
    return pd.concat([model_cv_result[f"split{y}_test_{metric}"] for y in range(0, nfolds)], ignore_index=True)


def extract_best_cohort_mean_scores(best_cv_results_combined):
    
    keep = [f"mean_test_{metric}" for metric in CV_SCORING_METRICS]
    [keep.append(f"std_test_{metric}")  for metric in CV_SCORING_METRICS]
    keep = keep + ['mean_fit_time','std_fit_time', 'mean_score_time', 'std_score_time']
    

    best_cohort_mean_scores = best_cv_results_combined[keep]

    for metric in CV_SCORING_METRICS:
        metric_label = METRICS_PLOT_LABELS.get(metric, metric)
        best_cohort_mean_scores[metric_label] = (best_cohort_mean_scores['mean_test_' + metric].round(4).astype(str) 
                                                 + " ± " + best_cohort_mean_scores['std_test_' + metric].round(4).astype(str)
        )
    
    for time_metric in ['fit_time', 'score_time']:
        
        metric_label = METRICS_PLOT_LABELS.get(time_metric, time_metric)
        best_cohort_mean_scores[metric_label] = (
            best_cohort_mean_scores['mean_' + time_metric].round(2).astype(str) 
            + " ± " + best_cohort_mean_scores['std_' + time_metric].round(2).astype(str)
        )
    
    best_cohort_mean_scores = best_cohort_mean_scores.drop(columns = keep)  # drop the original mean and std columns, keep only the formatted ones
    
    logger.info(f"Best cohort mean scores: \n{best_cohort_mean_scores.T.to_markdown()}")
    
    return best_cohort_mean_scores

def publish_best_cohort_mean_scores(best_cv_results_combined):
    
    best_cohort_mean_scores = extract_best_cohort_mean_scores(best_cv_results_combined)
    
    # publish a table plot of the best cohort mean scores with metrics as columns and models as rows and save it as a png
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=best_cohort_mean_scores.values, colLabels=best_cohort_mean_scores.columns, rowLabels=best_cohort_mean_scores.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title("Best Cohort CV Mean Scores", fontsize=14)
    plt.savefig(SAVE_PATH / f"best_cohort_mean_scores.png", bbox_inches='tight')
    
    logger.success(f"Saved best cohort mean scores to {SAVE_PATH} / best_cohort_mean_scores.png")

def extract_best_params_tables(best_cv_result):
    
        params_cols = [col for col in best_cv_result.columns if col.startswith('param_')]
        params_table = best_cv_result[params_cols]
        params_table = params_table.T.dropna()
        
        logger.info(f"Best CV cohort parameters: \n{params_table.to_markdown()}")
        
        return params_table


def publish_best_params_tables(combined_best_cv_results):
    
    for model in combined_best_cv_results.model_name.unique():
        
        model_best_cv_result = combined_best_cv_results[combined_best_cv_results.model_name == model]
        params_table = extract_best_params_tables(model_best_cv_result)
        params_table = params_table.rename(index=lambda x: x.replace('param_', ''))

        fig, ax = plt.subplots(figsize=(10, 0.5*len(params_table )))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=params_table.values, colLabels=params_table.columns, rowLabels=params_table.index, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        plt.title(f"Best CV Cohort Parameters for {model}", fontsize=14)
        plt.savefig(SAVE_PATH / f"best_cv_cohort_params_{model}.png", bbox_inches='tight')
    
    

def fit_best_estimator_on_test(cv_model, test_x, test_y):
    
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

    best_estimator = cv_model.best_estimator_
    test_score = cv_model.score(test_x, test_y)
    
    y_pred = best_estimator.predict(test_x)
    
    # dataframe with scores on metrics from CV scoring metrics
    
    scores = pd.DataFrame({
        "root_mean_squared_error": [root_mean_squared_error(test_y, y_pred)],
        "median_absolute_error": [median_absolute_error(test_y, y_pred)],
        "mean_absolute_error": [mean_absolute_error(test_y, y_pred)],
        "r2": [r2_score(test_y, y_pred)],
    })
    
    publish_test_scores(scores, cv_model.estimator.__class__.__name__)
    
    logger.info(f"Test score for model {cv_model.estimator.__class__.__name__}: {scores.T.to_markdown()}")
    #logger.info(f"Test score per sample for model {cv_model.estimator.__class__.__name__}: {score_sample}")
    
def publish_test_scores(scores, model_name: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=scores.values, colLabels=scores.columns, rowLabels=scores.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title("Test Scores of Best Estimator on Test Set", fontsize=14)
    plt.savefig(SAVE_PATH / f"{model_name}_test_scores.png", bbox_inches='tight')
    
    logger.success(f"Saved test scores to {SAVE_PATH} / {model_name}_test_scores.png")
    
       
        
# # TODO: do cv on best estimator and on test set, and do visualization on those ones.

if __name__ == "__main__":
    failsafe_checks()
    main()

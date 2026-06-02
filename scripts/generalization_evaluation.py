
from highres_ta.evaluation import TestSet
import dotenv
import pathlib
from model_selection import (
    ModelSelectionConfig,
    get_splits_by_expocode_salinity_bin_based,
    load_config,
    load_data,
    preprocess_data,
    salinity_binning,
)
from scipy.special import huber
from highres_ta import BaggingCatBoostResidualRegressor
from highres_ta import estimators as models
from highres_ta.inference import train_test_split
from highres_ta.evaluation import make_prediction_df, scoring_from_df, TestSet, get_set_props, add_some_index_levels
from optuna_inference import prepare_data, save_figs_to_pdf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from loguru import logger
import matplotlib.pyplot as plt
from test_set_sal_sensitivity import (compare_metrics_timeseries, 
                                      compare_residuals_maps, 
                                      compare_residuals_target_scatterplots, 
                                      compare_joint_kde_distribution, 
                                      compare_residuals_scatterplots, 
                                      noise_impact_analysis,
                                      compare_scores)

ROOT = pathlib.Path(dotenv.find_dotenv("pyproject.toml")).parent
MODEL_FILE = "bagged_catboost_residual_modelstructural_optuna_4.pkl"     
OUTPUTS_FOLDER = ROOT/f"outputs/test_set_evaluation"

def main():
    
    plt.close("all")

    # Loads and config
    trained_model =   models.BaggingCatBoostResidualRegressor.load(
        ROOT / f"models/{MODEL_FILE}"
    )
    config = load_config(ROOT / "scripts/cv_example_config.yaml")
    config.xname_features = trained_model.feature_names_in_.tolist()
    df = prepare_data(config)
    train_x, train_y, test_x, test_y = train_test_split(df, config)
    test_x = add_some_index_levels(test_x)
    
    global ORIGINAL_TEST_X, ORIGINAL_TEST_Y, TRAINED_MODEL, FEATURES
    ORIGINAL_TEST_Y = test_y
    ORIGINAL_TEST_X = test_x
    TRAINED_MODEL = trained_model
    FEATURES = trained_model.feature_names_in_.tolist()
    
    # Original test set predictions and scores
    yhat_test   = trained_model.predict(test_x)    
    predictions_df = make_prediction_df(y_pred = yhat_test, y_true = test_y, test_x=test_x)
    scores = scoring_from_df(predictions_df)
    nsamples = len(predictions_df['y_pred'])
    
    # bundle the original test set into a dataclass
    original_test_dict = {
        'test_x' : ORIGINAL_TEST_X.copy(),
        'predictions_df' : predictions_df, # to be filled after prediction
        'scores' : scores,
        'noise' : None,
        'nsamples': nsamples,
        'label' : 'all' 
    }
    props = get_set_props(0)
    original_test_dict.update(props)
    original_test_set = TestSet(**original_test_dict)

    # # COASTAL SUBSETTING
    
    # coastal_mask = test_x.index.get_level_values("is_coastal") == True
    # non_coastal_mask = ~coastal_mask
    
    # coastal_subset = make_test_subset(coastal_mask, 'coastal', 1)
    # non_coastal_subset = make_test_subset(non_coastal_mask, 'non_coastal', 2)
    
    # test_set_list = [original_test_set, coastal_subset, non_coastal_subset]
    
    # TODO: subset on non boolean indexes
    
    # SALINITY BIN SUBSETING
    
    bin_edges_dict = {
    1:"sal bin [<32]",
    2:"sal bin [32,34]",
    3:"sal bin [34,36]",
    4:"sal bin [>36]"
    }
    
    test_set_list = [original_test_set]
    sal_bins_labels = test_x.index.get_level_values("salinity_bin").unique().to_list()
    for i, sal_bin in enumerate(sal_bins_labels):
        mask = test_x.index.get_level_values("salinity_bin") == sal_bin
        subset = make_test_subset(mask, subset_label= bin_edges_dict[sal_bin], subset_nbr=i+1)
        test_set_list.append(subset)

    
    general_test_assessment(test_set_list)
    # global scores
    # coastal/noncoastal scores
    # salinity bin scores?
    # residual vs target
    # residual vs features
    # residual maps 
    # timeseries metric
    # -- yearly, is_coastal/non_coastal, salinity_bin, lat_lon
    # -- seasonal diagnostics
    # 
    
    
    
def make_test_subset(subsetting_idx, subset_label, subset_nbr):
    
    trained_model = TRAINED_MODEL
    test_x = ORIGINAL_TEST_X.copy()
    test_y = ORIGINAL_TEST_Y.copy()
    
    assert test_y.shape == test_x.index.get_level_values('time').shape # still the same size
    logger.info(f"{test_y.shape =},{test_x.index.get_level_values('time').shape=}")
    
    test_x = test_x[subsetting_idx]
    test_y = test_y[subsetting_idx]
    
    yhat_test= trained_model.predict(test_x)    
    predictions_df = make_prediction_df(y_pred = yhat_test, y_true = test_y, test_x=test_x, add_levels=False)
    scores = scoring_from_df(predictions_df)
    nsamples = len(predictions_df['y_pred'])
    
    # bundle the original test set into a dataclass
    test_dict = {
        'test_x' : test_x,
        'predictions_df' : predictions_df, # to be filled after prediction
        'scores' : scores,
        'nsamples': nsamples,
        'label' : subset_label 
    }
    props = get_set_props(subset_nbr)
    test_dict.update(props)
    subset_test_set = TestSet(**test_dict)
    
    return subset_test_set


def plot_sal_and_ta_distributions(test_set: TestSet, fig = None):

    if fig is None:
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10*2, 4*2))
    else:
        ax = fig.axes
        ax = np.array(fig.axes).reshape(2, 2)

    test_x = test_set.test_x
    color = test_set.color
    label = test_set.label
    linestyle = test_set.linestyle
    nsamples = test_set.nsamples
    
    
    # --- Left: true TA distributions ---
    ax[0,0].hist(
        test_set.predictions_df['y_true'],
        bins=50,
        density=True,
        alpha=0.4,
        label=f"{label} (n={nsamples})",
        color = color
    )

    test_set.predictions_df['y_true'].plot(kind="kde", ax=ax[0,0], linewidth=1, label=f"{label} KDE", color= color, linestyle=linestyle)

    ax[0,0].set_xlabel("Observed Total Alkalinity (µmol/kg)")
    ax[0,0].grid(True)
    ax[0,0].legend()
    
    # --- Left: predicted TA distributions ---
    ax[0,1].hist(
        test_set.predictions_df['y_pred'],
        bins=50,
        density=True,
        alpha=0.4,
        label=f"{label} (n={nsamples})",
        color = color
    )

    test_set.predictions_df['y_pred'].plot(kind="kde", ax=ax[0,1], linewidth=1, label=f"{label} KDE", color= color, linestyle=linestyle)

    ax[0,1].set_xlabel("Predictedd Total Alkalinity (µmol/kg)")
    ax[0,1].grid(True)
    ax[0,1].legend()
    
    
    # --- Resiudals distributions ---
    ax[1,0].hist(
        test_set.predictions_df['residuals'],
        bins=50,
        density=True,
        alpha=0.4,
        label=f"{label} (n={nsamples})",
        color = color
    )

    test_set.predictions_df['residuals'].plot(kind="kde", ax=ax[1,0], linewidth=1, label=f"{label} KDE", color= color, linestyle=linestyle)

    ax[1,0].set_xlabel("Residuals (µmol/kg)")
    ax[1,0].grid(True)
    ax[1,0].legend()
    
    
    # --- Left: salinity distributions ---
    ax[1,1].hist(
        test_x["salinity"],
        bins=50,
        density=True,
        alpha=0.4,
        label=f"{label} (n={nsamples})",
        color = color
    )

    test_x["salinity"].plot(kind="kde", ax=ax[1,1], linewidth=1, label=f"{label} KDE", color= color, linestyle=linestyle)


    ax[1,1].set_title("Salinity Distribution")
    ax[1,1].set_xlabel("Salinity")
    ax[1,1].grid(True)
    ax[1,1].legend()


    fig.tight_layout()

    return fig


def compare_sal_and_ta_distribution(test_set_list : list[TestSet]):
    
    #from highres_ta.evaluation import plot_noise_distribution
    
    fig = plot_sal_and_ta_distributions(test_set = test_set_list[0])
    for test_set in test_set_list[1:]:
        fig = plot_sal_and_ta_distributions(test_set=test_set, fig=fig)
        
    return fig


def general_test_assessment(test_set_list: list[TestSet], time_mesh = 'year', study_name = 'example_subsetting_eval.pdf'):
    
    features_list = FEATURES
    
    for test_set in test_set_list:
        logger.info(f"Scores for {test_set.label}:\n{test_set.scores.to_markdown(floatfmt='.3f')}")
    
    fig0 = compare_scores(test_set_list)
    fig1 = compare_metrics_timeseries(test_set_list=test_set_list, time_grouping_level=time_mesh)
    fig2 = compare_sal_and_ta_distribution(test_set_list)
    fig3 = compare_residuals_maps(test_set_list)
    fig4 = compare_residuals_target_scatterplots(test_set_list)
    fig5 = compare_joint_kde_distribution(test_set_list, feature="salinity")
    
    figs = [fig0, fig1, fig2, fig3, fig4, fig5]
    
    for feature in features_list:
        fig = compare_residuals_scatterplots(test_set_list, feature=feature)
        figs.append(fig)

    save_figs_to_pdf(figs, filename=f"{OUTPUTS_FOLDER}/{study_name}")
    
if __name__ == "__main__":
    main()
    
    
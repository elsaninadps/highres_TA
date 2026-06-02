
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
from highres_ta.evaluation import make_prediction_df, scoring_from_df, TestSet, get_set_props
from optuna_inference import prepare_data, save_figs_to_pdf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from loguru import logger
import matplotlib.pyplot as plt


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
    config.xname_features = trained_model.feature_names_in_.tolist() #TODO: clarify how config is used here
    
    df = prepare_data(config)
    train_x, train_y, test_x, test_y = train_test_split(df, config)
    
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
        'test_x' : test_x,
        'predictions_df' : predictions_df, # to be filled after prediction
        'scores' : scores,
        'noise' : None,
        'label' : 'original',
        'nsamples': nsamples 
    }
    props = get_set_props(0)
    original_test_dict.update(props)
    OriginalTestSet = TestSet(**original_test_dict)

    
    # Noise analysis
    
    
    # noise_impact_analysis(random_noise_test_set_list, 
    #                       study_name= "random_noise_impact_analysis.pdf"
    #                       )
    

    # noise_impact_analysis(uniform_noise_test_set_list, 
    #                       features_list= FEATURES,
    #                       study_name="uniform_0.5_noise_impact_analysis.pdf"
    #                       )
    # noise_impact_analysis(prop_dev_test_set_list, features_list = FEATURES,    
    #                       study_name= "prop_sal_dev_noise_impact_analysis.pdf"
    #                       )
    
    # noise_impact_analysis(inv_prop_dev_test_set_list, features_list = FEATURES
    #                       study_name= "inv_prop_sal_dev_noise_impact_analysis.pdf"
    #                       )


    
def make_uniform_noise_test_set(OriginalTestSet, noise_dict):   
    from highres_ta.evaluation import get_set_props
    # UNIFORM NOISE---------------
    
    test_x = OriginalTestSet.test_x
    
    i=1
    for label, noise in noise_dict.items:
        

        
        def make_testset_list(noise_dict_list, original_test_set):
    
    

    test_set_list = [original_test_set]
    for i, noise_dict in enumerate(noise_dict_list):
        
        test_set_dict = run_noisy_test(**noise_dict)
        plot_props = get_set_props(i+1) #zero always for oringinal test set
        test_set_dict.update(plot_props)
        
        test_set = TestSet(**test_set_dict)
        test_set_list.append(test_set)
        
    return test_set_list
        
        
    
    uniform_plus_noise = test_x['salinity']*0 + 0.5
    uniform_plus_dict ={
        'noise': uniform_plus_noise,
        'label': "Uniform +0.5 Noise"
    }
    
    uniform_minus_noise = test_x['salinity']*0 -0.5
    uniform_minus_dict ={
        'noise': uniform_minus_noise,
        'label': "Uniform -0.5 Noise"
    }
    
    uniform_noise_test_set_list = make_testset_list(
        noise_dict_list= [uniform_plus_dict, uniform_minus_dict],
        original_test_set=OriginalTestSet
    )
        
    return test_set_list


def make_proportional_noise_test_set():
    
        # -------------------------
    # PROPORTIONAL TO SAL DEVIATION NOISE
    # -------------------------

    small_prop_noise = (test_x["salinity"] - test_x["salinity"].median()) * 0.3
    small_prop_dict = {
        "noise": small_prop_noise,
        "label": "Proportional Noise (sal deviation * 0.3)",
    }

    large_prop_noise = (test_x["salinity"] - test_x["salinity"].median()) * 0.7
    large_prop_dict = {
        "noise": large_prop_noise,
        "label": "Proportional Noise (sal deviation * 0.7)",
    } 
    
    prop_dev_test_set_list = make_testset_list(
        noise_dict_list=[large_prop_dict, small_prop_dict],
        original_test_set=OriginalTestSet,
    )


def make_inv_prop_noise_test_set():
    
    
    # -------------------------
    # SAL DEVIATION INV PROP NOISE
    # -------------------------

    eps = 1e-3

    small_invprop_noise = 1 / (np.abs(test_x["salinity"] - test_x["salinity"].median()) + eps) * 0.3
    small_invprop_dict = {
        "noise": small_invprop_noise,
        "label": "Inv Prop Noise (1 / sal deviation * 0.3)",
    }

    large_invprop_noise = 1 / (np.abs(test_x["salinity"] - test_x["salinity"].median()) + eps) * 0.7
    large_invprop_dict = {
        "noise": large_invprop_noise,
        "label": "Inv Prop Noise (1 / sal deviation * 0.7)",
    }
    
    inv_prop_dev_test_set_list = make_testset_list(
        noise_dict_list=[large_invprop_dict, small_invprop_dict],
        original_test_set=OriginalTestSet,
    )


def make_random_noise_test_set():
    
    
     # -------------------------
    # RANDOM NOISE
    # -------------------------

    random_noise = np.random.rand(len(test_y)) - np.random.rand(len(test_y))
    random_noise_dict = {
        "noise": random_noise,
        "label": "Random Noise",
    }


    random_noise_test_set_list = make_testset_list(
        noise_dict_list=[random_noise_dict],
        original_test_set=OriginalTestSet,
    )


        
        
def run_noisy_test(label, noise):
    
    trained_model = TRAINED_MODEL
    original_test_y = ORIGINAL_TEST_Y
    noisy_test_x = ORIGINAL_TEST_X.copy()
    
    noisy_test_x['salinity'] = noisy_test_x['salinity'] + noise
    
    #noisy_test_x["noisy_salinity_bin"] = salinity_binning(noisy_salinity, bins=salinity_bins)

    noisy_yhat_test  = trained_model.predict(noisy_test_x)
    logger.info(f"{len(noisy_yhat_test)=}, {len(ORIGINAL_TEST_Y)=}, {len(noisy_test_x)=}")
    noisy_predictions_df = make_prediction_df(y_pred = noisy_yhat_test, y_true = original_test_y, test_x=noisy_test_x)
    noisy_scores = scoring_from_df(noisy_predictions_df)
    nsamples = len(noisy_predictions_df['y_pred'])
    
    
    testset_dict = {
        'test_x' : noisy_test_x,
        'noise' : noise,
        'predictions_df': noisy_predictions_df,
        'scores': noisy_scores,
        'label': label,
        'nsamples': nsamples
    }
    
    return testset_dict



def noise_impact_analysis(test_set_list, features_list, study_name = "example_salinity_imapct_analysis.pdf"):

    
    for test_set in test_set_list:
        logger.info(f"Scores for {test_set.label}:\n{test_set.scores.to_markdown(floatfmt='.3f')}")
    
    # initialize figures with first test set (original)
    
    fig0 = compare_noise_distribution(test_set_list)
    fig1 = compare_metrics_timeseries(test_set_list=test_set_list)
    fig3 = compare_scores(test_set_list)
    fig4 = compare_residuals_distributions(test_set_list)
    fig5 = compare_residuals_maps(test_set_list)
    fig6 = compare_residuals_target_scatterplots(test_set_list)
    fig7 = compare_joint_kde_distribution(test_set_list, feature="salinity")
    
    figs = [fig0, fig1, fig3, fig4, fig5, fig6, fig7]
    
    for feature in features_list:
        fig = compare_residuals_scatterplots(test_set_list, feature=feature)
        figs.append(fig)

    save_figs_to_pdf(figs, filename=f"{OUTPUTS_FOLDER}/{study_name}")
    
def plot_noise_distribution(test_set: TestSet, fig = None):

    if fig is None:
        fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
    else:
        ax = fig.axes

    test_x = test_set.test_x
    noise = test_set.noise
    color = test_set.color
    label = test_set.label
    linestyle = test_set.linestyle
    nsamples = test_set.nsamples
    # --- Left: salinity distributions ---
    ax[0].hist(
        test_x["salinity"],
        bins=50,
        density=True,
        alpha=0.4,
        label=f"{label} (n={nsamples})",
        color = color
    )

    test_x["salinity"].plot(kind="kde", ax=ax[0], linewidth=1, label=f"{label} KDE", color= color, linestyle=linestyle)


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
            label = f"{label} (n={nsamples})",
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

    
def compare_noise_distribution(test_set_list : list[TestSet]):
    
    #from highres_ta.evaluation import plot_noise_distribution
    
    fig = plot_noise_distribution(test_set = test_set_list[0])
    for test_set in test_set_list[1:]:
        fig = plot_noise_distribution(test_set=test_set, fig=fig)
        
    return fig
        
    
def compare_metrics_timeseries(test_set_list: list[TestSet], 
                               time_grouping_level: str = 'year', 
                               secondary_grouping_level: str | None = None, ):
    
    """Compare timeseries of metrics accross different test sets. 
    time_grouping_level : str =  year, month, 8D_bin, time
    secondary_grouping_level : str = salinity_bin, lat_bin, is_coastal"""
    
    from highres_ta.evaluation import plot_metrics_timeseries
    
    fig = plot_metrics_timeseries(test_set_list[0], 
                                  time_grouping_level=time_grouping_level, 
                                  secondary_grouping_level=secondary_grouping_level)
    
    for test_set in test_set_list[1:]:
        fig = plot_metrics_timeseries(test_set, 
                                      time_grouping_level= time_grouping_level, 
                                      secondary_grouping_level= secondary_grouping_level, 
                                      fig=fig)
        
    return fig
    
def compare_residuals_distributions(test_set_list: list[TestSet]):
    
    from highres_ta.evaluation import plot_residuals_distribution
    
    xmin = test_set_list[0].predictions_df["residuals"].min()
    xmax = test_set_list[0].predictions_df["residuals"].max()
    x_range = (xmin, xmax)
    
    fig = plot_residuals_distribution(test_set=test_set_list[0])
    
    for test_set in test_set_list[1:]:
        
        # adapt plot range if necessary
        xmin = min(xmin, test_set.predictions_df["residuals"].min())
        xmax = max(xmax, test_set.predictions_df["residuals"].max())
        x_range = (xmin, xmax)
        
        fig = plot_residuals_distribution(test_set=test_set, fig=fig, x_range=x_range)
        
    return fig
        
        
def compare_residuals_scatterplots(test_set_list: list[TestSet], feature):
    from highres_ta.evaluation import scatter_residuals_vs_feature
            
    fig = scatter_residuals_vs_feature(test_set=test_set_list[0], feature=feature)
    for test_set in test_set_list[1:]:
        fig = scatter_residuals_vs_feature(test_set=test_set, feature=feature, fig=fig)

    return fig


def compare_residuals_target_scatterplots(test_set_list: list[TestSet]):
    from highres_ta.evaluation import scatter_residuals_vs_target
            
    fig = scatter_residuals_vs_target(test_set=test_set_list[0])
    for test_set in test_set_list[1:]:
        fig = scatter_residuals_vs_target(test_set=test_set, fig=fig)

    return fig


# def test_set_annalysis(test_set:TestSet, filename="example_salinity_analysis.pdf"):
#     predictions_df = test_set.predictions_df
    
    
#     fig0 = plot_metrics_timeseries(test_set, secondary_grouping_level="is_coastal")
#     fig1 = plot_metrics_timeseries(test_set, time_grouping_level="month", secondary_grouping_level="is_coastal")
#     fig1 = plot_metrics_timeseries(test_set, secondary_grouping_level="salinity_bin")
#     fig2 = plot_metrics_timeseries(test_set, secondary_grouping_level="lat_bin")

#     fig3 = plot_residuals_distribution(predictions_df)
#     fig4 = plot_residuals_feature_swarm(predictions_df, feature="salinity")
#     fig5 = plot_residuals_feature_kde(predictions_df, feature="salinity")
    
#     figs = [fig0, fig1, fig2, fig3, fig4, fig5]
#     for feature in test_set.test_x.columns:
#         fig = scatter_residuals_vs_feature(test_set, feature)
#         figs.append(fig)

#     if test_set.noise is not None:
#         fig6 = plot_noise_distribution(test_set)
#         figs.append(fig6)
    
#     save_figs_to_pdf(figs, filename=filename)



def compare_scores(test_set_list):
    from highres_ta.evaluation import plot_scores_table
    
    all_scores = pd.concat([test_set.scores.rename(test_set.label) for test_set in test_set_list], axis=1)
    
    fig = plot_scores_table(all_scores, label="Overall Scores")
    fig.tight_layout()
    
    return fig


def compare_residuals_maps(test_set_list):
    from cartopy.crs import PlateCarree
    from highres_ta.evaluation import plot_residuals_map
    
    nplots = len(test_set_list)
    ncols = 2
    nrows = (nplots + ncols - 1) // ncols  # ceil division

    fig = plt.figure(figsize=(12, 5 * nrows))

    for i, test_set in enumerate(test_set_list):
        
        ax = fig.add_subplot(
            nrows, ncols, i + 1,
            projection=PlateCarree(205)
        )
        plot_residuals_map(
            test_set.predictions_df,
            ax=ax,
        )

        ax.set_title(f"{test_set.label} Residuals (Predicted - True)", fontsize=14, fontweight="bold")
        ax.coastlines(lw=0.5)

    axes = fig.get_axes()
    cbar = plt.colorbar(
        axes[0].collections[0],
        ax=axes[-1],
        location="right",
        label="Residual (µmol kg$^{-1}$)",
        shrink=0.6,
    )
    cbar.set_ticks(range(-20, 22, 4))

    return fig


def compare_joint_kde_distribution(test_set_list, feature):
    
    from highres_ta.evaluation import plot_residuals_feature_kde
        
    nplots = len(test_set_list)
    ncols = 2
    nrows = (nplots + ncols - 1) // ncols  # ceil division

    fig = plt.figure(figsize=(12, 5 * nrows))

    for i, test_set in enumerate(test_set_list):
        
        ax = fig.add_subplot(nrows, ncols, i + 1)
        fig = plot_residuals_feature_kde(test_set, feature, fig)
        
    return fig


    
    # # OriginalTestSet = TestSet(
    # #     test_x = test_x,
    # #     predictions_df = predictions_df, # to be filled after prediction
    # #     scores = scores,
    # #     noise = None,
    # #     color = "blue",
    # #     marker = "o",
    # #     label = "Original",
    # #     linestyle = "-",
    # # )
    
    # # Noisy test set creation and predictions
    # #noise = (test_x['salinity'] - test_x['salinity'].median() ) * 0.5
    
    # noise = np.zeros(len(test_x['salinity']))+ 0.5
    # noisy_test_x, noisy_predictions_df, noisy_scores = make_noisy_test(noise, test_x, test_y, trained_model)
    # Uniform_Positive_Noise_TestSet = TestSet(
    #     test_x = noisy_test_x,
    #     predictions_df = noisy_predictions_df,
    #     scores = noisy_scores,
    #     noise = noise,
    #     color = "orange",
    #     marker = "x",
    #     label = "Uniform +0.5 Noise",
    #     linestyle = "--",   
    # )
    
    # noise = np.zeros(len(test_x['salinity']))-0.5
    # noisy_test_x, noisy_predictions_df, noisy_scores = make_noisy_test(noise, test_x, test_y, trained_model)
    # Uniform_Negative_Noise_TestSet = TestSet(
    #     test_x = noisy_test_x,
    #     predictions_df = noisy_predictions_df,
    #     scores = noisy_scores,
    #     noise = noise,
    #     color = "green",
    #     marker = "s",
    #     label = "Uniform -0.5 Noise",
    #     linestyle = ":",   
    # )
    
    # uniform_test_set_list = [OriginalTestSet, Uniform_Positive_Noise_TestSet, Uniform_Negative_Noise_TestSet]
    
    
    # # -------------------------
    # # SAL DEVIATION PROPORTIONAL NOISE
    # # -----------------------
    
    # noise = (test_x['salinity'] - test_x['salinity'].median() ) * 0.3
    # noisy_test_x, noisy_predictions_df, noisy_scores = make_noisy_test(noise, test_x, test_y, trained_model)
    # small_prop_Noise_TestSet = TestSet(
    #     test_x = noisy_test_x,
    #     predictions_df = noisy_predictions_df,
    #     scores = noisy_scores,
    #     noise = noise,
    #     color = "orange",
    #     marker = "x",
    #     label = "Proportional Noise (sal deviation*0.3)",
    #     linestyle = "--",     
    # )
    
    # noise = (test_x['salinity'] - test_x['salinity'].median() ) * 0.7
    # noisy_test_x, noisy_predictions_df, noisy_scores = make_noisy_test(noise, test_x, test_y, trained_model)
    # large_prop_Noise_TestSet = TestSet(
    #     test_x = noisy_test_x,
    #     predictions_df = noisy_predictions_df,
    #     scores = noisy_scores,
    #     noise = noise,
    #     color = "green",
    #     marker = "s",
    #     label = "Proportional Noise (sal deviation*0.7)",
    #     linestyle = ":",   
    # )
    
    # prop_dev_test_set_list = [OriginalTestSet, large_prop_Noise_TestSet, small_prop_Noise_TestSet]

    
    # # -------------------------
    # # SAL DEVIATION INV PROP NOISE
    # # -----------------------
    
    # #noise = 1/(test_x['salinity'] - test_x['salinity'].median() ) * 0.3
    # eps = 1e-3  # or domain-relevant scale
    # noise = 1 / (np.abs(test_x['salinity'] - test_x['salinity'].median()) + eps) * 0.7
    
    # noisy_test_x, noisy_predictions_df, noisy_scores = make_noisy_test(noise, test_x, test_y, trained_model)
    # small_invprop_Noise_TestSet = TestSet(
    #     test_x = noisy_test_x,
    #     predictions_df = noisy_predictions_df,
    #     scores = noisy_scores,
    #     noise = noise,
    #     color = "orange",
    #     marker = "x",
    #     label = "Inv Prop Noise (sal deviation*0.3)",
    #     linestyle = "--",      
    # )
    
    # #noise = 1/(test_x['salinity'] - test_x['salinity'].median() ) * 0.7
    # noise = 1 / (np.abs(test_x['salinity'] - test_x['salinity'].median()) + eps) * 0.7
    
    # noisy_test_x, noisy_predictions_df, noisy_scores = make_noisy_test(noise, test_x, test_y, trained_model)
    # large_invprop_Noise_TestSet = TestSet(
    #     test_x = noisy_test_x,
    #     predictions_df = noisy_predictions_df,
    #     scores = noisy_scores,
    #     noise = noise,
    #     color = "green",
    #     marker = "s",
    #     label = "Inv Prop Noise (1/sal deviation*0.7)",
    #     linestyle = ":",  
    # )
    
    # inv_prop_dev_test_set_list = [OriginalTestSet, large_invprop_Noise_TestSet, small_invprop_Noise_TestSet]

    
    # #-------------------
    # # RANDOM NOISE
    # #-------------------
    
    
    # noise = np.random.rand(len(test_y))
    # noise -= np.random.rand(len(test_y))
    
    # noisy_test_x, noisy_predictions_df, noisy_scores = make_noisy_test(noise, test_x, test_y, trained_model)
    # Random_Noise_TestSet = TestSet(
    #     test_x = noisy_test_x,
    #     predictions_df = noisy_predictions_df,
    #     scores = noisy_scores,
    #     noise = noise,
    #     color = "orange",
    #     marker = "x",
    #     label = "Random Noise",
    #     linestyle = "--",      
    # )
    
    # random_noise_test_set_list = [OriginalTestSet, Random_Noise_TestSet]
    
    # noise_impact_analysis(random_noise_test_set_list, 
    #                       config.xname_features, 
    #                       study_name= "random_noise_impact_analysis.pdf"
    #                       )
    

    # #test_set_annalysis(predictions_df, test_x, filename="example_salinity_analysis.pdf")
    # #test_set_annalysis(noisy_predictions_df, noisy_test_x, filename="example_noisy_salinity_analysis.pdf", additional_figs=[noise_distribution_fig])

    # # noise_impact_analysis(uniform_test_set_list, 
    # #                       features_list = config.xname_features, 
    # #                       study_name="uniform_0.5_noise_impact_analysis.pdf"
    # #                       )
    # # noise_impact_analysis(prop_dev_test_set_list, 
    # #                       config.xname_features, 
    # #                       study_name= "prop_sal_dev_noise_impact_analysis.pdf"
    # #                       )
    
    # # noise_impact_analysis(inv_prop_dev_test_set_list, 
    # #                       config.xname_features, 
    # #                       study_name= "inv_prop_sal_dev_noise_impact_analysis.pdf"
    # #                       )

    # # #ds = inference(bagged_model, train_x.copy())
     
        





    
if __name__ == "__main__":
    main()
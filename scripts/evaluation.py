
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
from inference import train_test_split
from highres_ta.evaluation import make_prediction_df, scoring_from_df, TestSet, get_plot_props
from optuna_inference import prepare_data, save_figs_to_pdf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from loguru import logger
import matplotlib.pyplot as plt


ROOT = pathlib.Path(dotenv.find_dotenv("pyproject.toml")).parent
MODEL_FILE = "bagged_catboost_residual_modelstructural_optuna_4.pkl"     
OUTPUTS_FOLDER = ROOT/f"outputs/salinity_analysis"

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
    
    global ORIGINAL_TEST_X, ORIGINAL_TEST_Y, TRAINED_MODEL
    ORIGINAL_TEST_Y = test_y
    ORIGINAL_TEST_X = test_x
    TRAINED_MODEL = trained_model



    
if __name__ == "__main__":
    main()
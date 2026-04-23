from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
import json
from sklearn.metrics import median_absolute_error, root_mean_squared_error, r2_score, mean_absolute_error
import joblib
import catboost as cb
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import copy
import scipy.stats as stats
import seaborn as sns
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def root_median_squared_error(y_true, y_pred):
    # from copilot
    squared_errors = (y_true - y_pred) ** 2
    median_sq_error = np.median(squared_errors)
    return np.sqrt(median_sq_error)

# Huber loss without heavy pytorch logic
def huber_loss(y_true, y_pred, delta=1.0):
    #from chatgpt
    
    error = y_true - y_pred
    
    is_small = np.abs(error) <= delta
    
    squared_loss = 0.5 * error**2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    
    return np.mean(np.where(is_small, squared_loss, linear_loss))


def blank_plot(n_subplots, ncols=2, base_size=(5, 5)):
    
    n_rows = n_subplots // ncols + n_subplots % ncols
    fig, axes = plt.subplots(n_rows, ncols, figsize=(base_size[0]*ncols, base_size[1]*n_rows))
    axes = axes.flatten()
    return fig, axes

def blank_submaps(n_subplots, n_cols, lon_res=1.0, lat_res=1.0, base_size=(12, 12)):
    
    n_rows = n_subplots // n_cols + n_subplots % n_cols

    projection = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(base_size[0]*n_cols, base_size[1]*n_rows),
        subplot_kw={"projection": projection}
    )

    # Flatten axes for easy iteration (works for 1D/2D cases)
    axes = np.atleast_1d(axes).flatten()

    # Define tick locations once
    lon_ticks = np.arange(-180, 181, lon_res)
    lat_ticks = np.arange(-90, 91, lat_res)

    for i, ax in enumerate(axes):

        if i >= n_subplots:
            ax.set_visible(False)
            continue

        ax.set_global()
        ax.coastlines()

        gl = ax.gridlines(
            crs=projection,
            draw_labels=False,
            linewidth=0.5,
            color='gray',
            alpha=0.5,
            linestyle='--'
        )

        gl.xlocator = plt.FixedLocator(lon_ticks)
        gl.ylocator = plt.FixedLocator(lat_ticks)

        ax.add_feature(cfeature.LAND, zorder=10, facecolor='white')
        ax.add_feature(cfeature.OCEAN, facecolor='white')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)

    return fig, axes


def blank_map(projection = ccrs.PlateCarree(), lon_res = 1.0, lat_res = 1.0 ):

    # Set grid resolution (change this)

    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=projection)

    ax.set_global()
    ax.coastlines()

    # Define tick locations
    lon_ticks = np.arange(-180, 181, lon_res)
    lat_ticks = np.arange(-90, 91, lat_res)

    gl = ax.gridlines(
        crs=projection,
        draw_labels=False,
        linewidth=0.5,
        color='gray',
        alpha=0.5,
        linestyle='--'
    )

    gl.xlocator = plt.FixedLocator(lon_ticks)
    gl.ylocator = plt.FixedLocator(lat_ticks)


    ax.add_feature(cfeature.LAND, zorder = 10, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)

    
    return fig, ax


def delete_axes(fig, axes):
    for ax in axes:
        if not ax.has_data():
            fig.delaxes(ax)

def get_feature_importance(model, X, y=None, scoring=None, n_repeats=10, random_state=42):
    
    # from chatgpt
    
    feature_names = getattr(model, "feature_names_in_", X.columns)

    # 1. Tree models (RF, GBM etc.)
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

    # 2. Linear models
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_)

    # 3. CatBoost models
    elif hasattr(model, "get_feature_importance"):
        importance = model.get_feature_importance()

    # 4. Fully model-agnostic fallback
    else:
        if y is None:
            raise ValueError("y must be provided for permutation importance")

        perm = permutation_importance(
            model,
            X,
            y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state
        )

        importance = perm.importances_mean
        
    feat_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })
    
    feat_importance_df= feat_importance_df.set_index('feature', drop=True)
    
    return feat_importance_df


def plot_actual_vs_predicted(y_true, y_pred, ax = None, fig = None, title = 'Actual vs predicted', label = ''):

    single_figure = False
    if ax == None:
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(14, 10))
        single_figure = True

    ax.scatter(y_true, y_pred, alpha=0.6, label = label)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    if single_figure == True:
        ax.set_title(f"{title} - {label}")
        plt.show();


def plot_residuals_vs_predicted(residuals, y_pred, ax = None, fig = None, title = 'Residuals vs Predicted', label = ''):

    single_figure = False
    
    if ax == None:
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(14, 10))
        single_figure = True

    ax.scatter(y_pred, residuals, alpha=0.6, label = label)
    ax.axhline(0, linestyle="--")

    ax.set_xlabel("Residuals")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    if single_figure == True:
        ax.set_title(f"{title} - {label}")
        plt.show();   
        
def plot_residuals_distribution(residuals, ax = None, fig = None, title = 'Residuals distribution', label = ''):
    
    single_figure = False
    if ax == None:
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(14, 10))
        single_figure = True

    sns.histplot(residuals, kde=True, ax= ax, alpha = 0.6, label = label)
    ax.axhline(0, linestyle="--")

    ax.set_xlabel("Residuals")
    ax.set_title(title)

    if single_figure == True:
        ax.set_title(f"{title} - {label}")
        plt.show();   


def plot_qq_residuals(residuals, ax=None, fig=None, title='Q-Q Plot of Residuals', label=None):
    
    single_figure = False

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14,10))
        single_figure = True

    # Compute theoretical quantiles and ordered residuals
    (theoretical_q, ordered_residuals), _ = stats.probplot(residuals, dist="norm")

    # Plot dataset (matplotlib will auto-cycle colors)
    ax.scatter(theoretical_q, ordered_residuals, alpha=0.6, label=label)

    # Draw reference line only once
    if len(ax.lines) == 0:
        min_q = theoretical_q.min()
        max_q = theoretical_q.max()
        ax.plot([min_q, max_q], [min_q, max_q], linestyle="--", color="black")

    ax.set_title(title)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")

    if single_figure == True:
        ax.set_title(f"{title} - {label}")
        plt.show();  

def plot_residual_vs_predictor(residuals, x_eval, feature_name, ax = None, fig = None, title = 'Residual vs predictor', label = ''):

    single_figure = False
    if ax == None:
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(14, 10))
        single_figure = True
        
    predictor = x_eval[feature_name]

    ax.scatter(predictor, residuals, alpha=0.6, label = label)

    # min_val = min(predictor.min(), residuals.min())
    # max_val = max(predictor.max(), residuals.max())

    # ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    ax.set_xlabel("Residual")
    ax.set_ylabel(f"{feature_name}")
    ax.set_title(title)

    if single_figure == True:
        ax.set_title(f"{title} - {label}")
        plt.show();


def plot_residuals_on_map(residuals, validation_lon, validation_lat, title = '',  ax = None, fig = None):

    single_figure = False
    if ax is None:
        fig, ax = blank_map()
        single_figure = True
    
    scatter = ax.scatter(validation_lon, validation_lat, c=residuals, marker = 's', cmap='viridis', alpha=0.6, vmin=residuals.min(), vmax=residuals.max())
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label='residuals')
    
    if single_figure == True:
        plt.colorbar(scatter, ax=ax, label='residuals')
        plt.show();

def plot_residuals_vs_time(residuals, validation_time, title = '', ax = None, fig = None):

    single_figure = False
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 10))
        single_figure = True

    ax.scatter(validation_time, residuals, alpha=0.6)
    ax.axhline(0, linestyle="--")

    ax.set_xlabel("Time")
    ax.set_ylabel("Residuals")
    ax.set_title(title)

    if single_figure == True:
        plt.show();
    


import copy
import shutil
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# Assume get_feature_importance and plotting functions already exist


# =========================
# SUBMODEL
# =========================

class subModel:
    
    def __init__(self, base_regressor, submodel_name, parent_dir):
        
        self.regressor = copy.deepcopy(base_regressor)
        
        self.name = submodel_name
        self.parent_dir = parent_dir
        self.regressor_type = type(base_regressor)
        
        self.filepath = parent_dir / submodel_name
        
        self.eval_scores = pd.DataFrame()
        self.features_importance = pd.DataFrame()
        self.predictions = pd.DataFrame()
        self.features = []
        self.is_trained = False

    def prepare_directory(self, mode="overwrite"):
        
        if self.filepath.exists():
            if mode == "overwrite":
                shutil.rmtree(self.filepath)
                self.filepath.mkdir(parents=True)
            elif mode == "skip":
                return
            elif mode == "error":
                raise FileExistsError(f"{self.filepath} already exists")
        else:
            self.filepath.mkdir(parents=True)

    def simple_training(self, x_train, y_train, x_eval, y_eval, overwrite=False, export=False):
        
        if self.is_trained and not overwrite:
            raise RuntimeError(f"{self.name} already trained. Use overwrite=True to retrain.")
        
        if export:
            self.prepare_directory(mode="overwrite" if overwrite else "error")
        
        self.regressor.fit(x_train, y_train)

        y_pred = self.regressor.predict(x_eval)

        if y_pred.ndim == 2:
            y_pred = y_pred[:, 0]

        self.predictions = pd.DataFrame({
            "y_pred": y_pred,
            "y_eval": y_eval,
            "residuals": y_pred - y_eval
        })

        self.eval_scores = pd.DataFrame({
            "MAE": [mean_absolute_error(y_eval, y_pred)],
            "MdAE": [median_absolute_error(y_eval, y_pred)],
            "RMSE": [root_mean_squared_error(y_eval, y_pred)],
            "RMdSE": [root_median_squared_error(y_eval, y_pred)],
            "HuberLoss": [huber_loss(y_eval, y_pred, delta=1.0)],
            "R2": [r2_score(y_eval, y_pred)],
            "Bias": [(y_pred - y_eval).mean()]
        })
        self.eval_scores.index = [self.name]
        self.eval_scores = self.eval_scores.T
        
        self.features_importance = get_feature_importance(self.regressor, x_train, y_train)
        self.features = x_train.columns.tolist()
        
        self.is_trained = True
        
        if export:
            self.export_submodel()

    def export_submodel(self):
        
        self.prepare_directory(mode="overwrite")
        
        joblib.dump(self.regressor, self.filepath / f"{self.name}_trained_regressor.joblib")

        self.predictions.to_csv(self.filepath / f"{self.name}_predictions.csv", index=False)
        self.features_importance.to_csv(self.filepath / f"{self.name}_features_importance.csv", index=False)
        self.eval_scores.to_csv(self.filepath / f"{self.name}_eval_scores.csv", index=True)
        
        metadata = {
            "model_type": self.regressor_type.__name__,
            "params": self.regressor.get_params(),
        }
        
        with open(self.filepath / f"{self.name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def plot_actual_vs_predicted(self, title= 'Actual vs Predicted', ax = None, fig = None):
        plot_actual_vs_predicted(self.predictions['y_eval'], self.predictions['y_pred'], title=title, label = self.name, ax = ax, fig = fig)

    
    def plot_residuals_vs_predicted(self, title= 'Residuals vs Predicted', ax = None, fig = None):
        plot_residuals_vs_predicted(self.predictions['y_eval'], self.predictions['y_pred'], title=title, label = self.name, ax = ax, fig = fig)

        
    def plot_residuals_distribution(self, title= 'Residuals Distribution', ax = None, fig = None):
        plot_residuals_distribution(self.predictions['residuals'], title=title, label = self.name, ax = ax, fig = fig)

        
    def plot_qq_residuals(self, title= 'QQ Plot of Residuals', ax = None, fig = None):
        plot_qq_residuals(self.predictions['residuals'], title=title, label = self.name, ax = ax, fig = fig)

    def plot_residual_vs_predictor(self, x_eval, feature_name, title = 'Residuals vs predictor', ax = None, fig = None):
        plot_residual_vs_predictor(self.predictions['residuals'], x_eval, feature_name, title=title, label = self.name, ax = ax, fig = fig)
    
    def plot_residuals_on_map(self, validation_lon, validation_lat, ax = None, fig = None):
        plot_residuals_on_map(self.predictions['residuals'], validation_lon, validation_lat, title = self.name, ax = ax, fig = fig)
                
    def plot_residuals_vs_time(self, validation_time, title = '', ax = None, fig = None):
        plot_residuals_vs_time(self.predictions['residuals'], validation_time, title = title, ax = ax, fig = fig)
    
    
    def plot_residual_vs_all_predictors(self, x_eval):
        
        n_features = len(self.features)
        fig, axes = blank_plot(n_subplots=n_features, ncols=3, base_size=(5, 5))
        
        for i, feature_name in enumerate(self.features):
            self.plot_residual_vs_predictor(self.predictions['residuals'], x_eval, feature_name, title=feature_name, ax = axes[i], fig = fig)
            fig.suptitle(f"{self.name}", fontsize=16)
        
        plt.tight_layout()
        plt.show();
        
        
    def plot_all(self, x_eval):
        
        # main 4 diagnostic plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        self.plot_actual_vs_predicted( ax=axes[0], fig=fig)
        self.plot_residuals_vs_predicted( ax=axes[1], fig=fig)
        self.plot_residuals_distribution( ax=axes[2], fig=fig)
        self.plot_qq_residuals( ax=axes[3], fig=fig)
        
        fig.suptitle(f"{self.name}", fontsize=16)
        
        plt.tight_layout()
        plt.show();
        
        # residual vs predictor for all predictors
        self.plot_residual_vs_all_predictors(x_eval)

# =========================
# MODEL WRAPPER
# =========================

class ModelWrapper:

    def __init__(self, regressor, model_name, parent_dir):
        
        self.untrained_regressor = regressor
        self.name = model_name
        self.regressor_type = type(regressor)
        
        self.filepath = parent_dir / model_name
        
        self.single_training_model = None
        
        self.cv_eval_scores = pd.DataFrame()
        self.cv_features_importance = pd.DataFrame()
        self.cv_models_dict: dict[str, subModel] = {}
        self.features = []

    def prepare_directory(self, mode="overwrite"):
        
        if self.filepath.exists():
            if mode == "overwrite":
                shutil.rmtree(self.filepath)
                self.filepath.mkdir(parents=True)
            elif mode == "skip":
                return
            elif mode == "error":
                raise FileExistsError(f"{self.filepath} already exists")
        else:
            self.filepath.mkdir(parents=True)

    def cross_validation_run(self, cv_folds, overwrite=False, export=False):
        
        if export:
            self.prepare_directory(mode="overwrite" if overwrite else "error")
        
        self.cv_models_dict = {}
        self.cv_eval_scores = pd.DataFrame()
        self.cv_features_importance = pd.DataFrame()
        
        for fold in cv_folds:
            
            fold_model = subModel(self.untrained_regressor, fold, self.filepath)
            
            x_train_fold = cv_folds[fold]['x_train']
            y_train_fold = cv_folds[fold]['y_train']
            x_eval_fold = cv_folds[fold]['x_validation']
            y_eval_fold = cv_folds[fold]['y_validation']
        
            fold_model.simple_training(
                x_train_fold, y_train_fold,
                x_eval_fold, y_eval_fold,
                overwrite=True,
                export=export
            )
            
            self.cv_models_dict[fold] = fold_model
            
            self.cv_eval_scores = pd.concat([self.cv_eval_scores, fold_model.eval_scores], axis=1)
            
            fold_feat_importance_df = fold_model.features_importance.rename(columns={'importance': fold})
            self.cv_features_importance = pd.concat([self.cv_features_importance, fold_feat_importance_df], axis=1)
            
        self.features = self.cv_features_importance.index.tolist()
        
        if export:
            self.export_cv_results()

    def single_training(self, x_train, y_train, X_eval, y_eval, overwrite=False, export=False):
        
        self.single_training_model = subModel(self.untrained_regressor, self.name, self.filepath)
        
        self.single_training_model.simple_training(
            x_train, y_train, X_eval, y_eval,
            overwrite=overwrite,
            export=export
        )
        
        self.features = self.single_training_model.features

    def export_cv_results(self):
        
        for fold in self.cv_models_dict:
            self.cv_models_dict[fold].export_submodel()
        
        self.cv_eval_scores.to_csv(self.filepath / f"{self.name}_cv_eval_scores.csv")
        self.cv_features_importance.to_csv(self.filepath / f"{self.name}_cv_features_importance.csv")

    def export_model_wrapper_info(self):
        
        metadata = {
            "model_name": self.name,
            "regressor_params": self.untrained_regressor.get_params(),
            "regressor_class": self.regressor_type.__name__,
        }
        
        with open(self.filepath / f"{self.name}_model_wrapper_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def plot_cv_actual_vs_predicted(self, title = '', ax = None, fig = None):
        
        single_figure = False
        
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 10))
            single_figure = True
        
        for fold in self.cv_models_dict.keys():
            self.cv_models_dict[fold].plot_actual_vs_predicted( ax = ax, fig = fig)
        
        ax.legend()
        
        plt.show() if single_figure == True else None
        
    def plot_cv_residuals_vs_predicted(self, title = '', ax = None, fig = None):
        
        single_figure = False
        
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 10))
            single_figure = True
            
        for fold in self.cv_models_dict.keys():
            self.cv_models_dict[fold].plot_residuals_vs_predicted( ax = ax, fig = fig)
        
        ax.legend()
        
        plt.show() if single_figure == True else None
        
    def plot_cv_residuals_distribution(self, title = '', ax = None, fig = None):
        
        single_figure = False
        
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 10))
            single_figure = True

        for fold in self.cv_models_dict.keys():
            self.cv_models_dict[fold].plot_residuals_distribution( ax = ax, fig = fig)
        
        ax.legend()
        
        plt.show() if single_figure == True else None
        
    def plot_cv_qq_residuals(self, title = '', ax = None, fig = None):
        
        single_figure = False
        
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 10))
            single_figure = True
        
        for fold in self.cv_models_dict.keys():
            fold_model = self.cv_models_dict[fold]
            fold_model.plot_qq_residuals(ax = ax, fig = fig)
        
        ax.legend()
        plt.show() if single_figure == True else None
        
    def plot_cv_residual_vs_predictor(self, cv_fold_dict, feature_name, title = '', ax = None, fig = None):
        
        single_figure = False
        
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 10))
            single_figure = True
        
        for fold in self.cv_models_dict.keys():
            fold_model = self.cv_models_dict[fold]
            x_eval = cv_fold_dict[fold]['x_validation']
            fold_model.plot_residual_vs_predictor(x_eval, feature_name, title=title, ax = ax, fig = fig)
        
        ax.legend()
        plt.show() if single_figure == True else None
        
    def plot_cv_residual_vs_all_predictors(self, cv_folds_dict):
        
        n_features = len(self.features)
        fig, axes = blank_plot(n_subplots=n_features, ncols=3, base_size=(5, 5))
        
        for i, feature_name in enumerate(self.features):
            for fold in self.cv_models_dict.keys():
                fold_model = self.cv_models_dict[fold]
                x_eval = cv_folds_dict[fold]['x_validation']
                fold_model.plot_residual_vs_predictor(x_eval = x_eval, feature_name = feature_name, ax = axes[i], fig = fig)
        
        fig.suptitle(f"{self.name}", fontsize=16)
        delete_axes(fig, axes)
        plt.tight_layout()
        plt.show();
        
    def plot_cv_residuals_on_map(self, cv_folds_dict):
        
        n_subplots = len(self.cv_models_dict)
        
        fig, axes = blank_submaps(n_subplots=n_subplots, n_cols=3)
        
        for i, fold in enumerate(self.cv_models_dict.keys()):
            
            fold_model = self.cv_models_dict[fold]
            validation_lon = cv_folds_dict[fold]['validation_lon']
            validation_lat = cv_folds_dict[fold]['validation_lat']
            fold_model.plot_residuals_on_map(validation_lon, validation_lat, ax = axes[i], fig = fig)
        
        
        fig.suptitle(f"{self.name}", fontsize=16)
        delete_axes(fig, axes)
        plt.show();
    
    def plot_cv_residuals_vs_time(self, cv_folds_dict):
        
        n_subplots = len(self.cv_models_dict)
        
        fig, axes = blank_plot(n_subplots=n_subplots, ncols=3, base_size=(5, 5))
        
        for i, fold in enumerate(self.cv_models_dict.keys()):
            
            fold_model = self.cv_models_dict[fold]
            validation_time = cv_folds_dict[fold]['validation_time']
            fold_model.plot_residuals_vs_time(validation_time, title = f"Fold: {fold}", ax = axes[i], fig = fig)
        
        
        fig.suptitle(f"{self.name}", fontsize=16)
        delete_axes(fig, axes)
        plt.tight_layout()
        plt.show();
            
    def plot_cv_all(self, cv_folds_dict):
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        self.plot_cv_actual_vs_predicted(ax = axes[0], fig = fig)
        self.plot_cv_residuals_vs_predicted(ax = axes[1], fig = fig)
        self.plot_cv_residuals_distribution(ax = axes[2], fig = fig)
        self.plot_cv_qq_residuals(ax = axes[3], fig = fig)
        
        fig.suptitle(f"{self.name}", fontsize=16)
        plt.tight_layout()
        plt.show();
        
        self.plot_cv_residual_vs_all_predictors(cv_folds_dict)


# =========================
# EXPERIMENT
# =========================

class Experiment():
    
    def __init__(self, cv_folds_dict, parent_dir , experiment_name):
        
        self.name = f"Experiment_{experiment_name}"
        
        self.filepath = parent_dir / self.name
        self.cv_folds = cv_folds_dict
        
        self.average_scores = pd.DataFrame()
        self.average_features_importance = pd.DataFrame()
        
        self.models_dict: dict[str, ModelWrapper] = {}

    def prepare_directory(self, mode="overwrite"):
        
        if self.filepath.exists():
            if mode == "overwrite":
                shutil.rmtree(self.filepath)
                self.filepath.mkdir(parents=True)
            elif mode == "skip":
                return
            elif mode == "error":
                raise FileExistsError(f"{self.filepath} already exists")
        else:
            self.filepath.mkdir(parents=True)
            
    def add_a_new_model(self, regressor, model_name:str):
        
        ModelWrapper_instance = ModelWrapper(regressor, model_name, self.filepath)
        self.models_dict[model_name] = ModelWrapper_instance
    
    def add_an_existing_model(self, trained_model_wrapper: ModelWrapper):
        
        if trained_model_wrapper.name in self.models_dict:
            raise ValueError(f"Model with name {trained_model_wrapper.name} already exists in the experiment.")
        
        self.models_dict[trained_model_wrapper.name] = trained_model_wrapper
        
        trained_model_wrapper.filepath = self.filepath / trained_model_wrapper.name
        trained_model_wrapper.export_model_wrapper_info()
    
    def add_many_new_models(self, dict_of_models: dict[str, object]):
        
        """dict_of_models should be in format: {model_name: regressor_instance}"""
        
        for regressor, model_name in dict_of_models.items():
            self.add_a_new_model(regressor=regressor, model_name=model_name)
    
    def add_many_existing_models(self, list_of_trained_model_wrappers: list[ModelWrapper]):
        for a_trained_model_wrapper in list_of_trained_model_wrappers:
            self.add_an_existing_model(a_trained_model_wrapper)
            
    def run_cv_for_all_models(self, overwrite=False, export=False):

        self.prepare_directory(mode="overwrite" if overwrite else "error")

        for model in self.models_dict.values():
            model.cross_validation_run(self.cv_folds, overwrite=overwrite, export=export)

    def update_summary_tables(self):
        
        for model_name in self.models_dict:
            model = self.models_dict[model_name]
            self.average_scores[model_name] = model.cv_eval_scores.mean(axis=1)
            self.average_features_importance[model_name] = model.cv_features_importance.mean(axis=1)

        self.average_scores.to_csv(self.filepath / f"{self.name}_average_cv_eval_scores.csv")
        self.average_features_importance.to_csv(self.filepath / f"{self.name}_average_cv_features_importance.csv")

    def export_experiment_results(self):
        
        for model in self.models_dict.values():
            model.export_model_wrapper_info()
            model.export_cv_results()
        
        self.average_scores.to_csv(self.filepath / f"{self.name}_average_cv_eval_scores.csv")
        self.average_features_importance.to_csv(self.filepath / f"{self.name}_average_cv_features_importance.csv")
        
    def compare_actual_vs_predicted(self, ncols = 2):

        fig, axes = blank_plot(n_subplots=len(self.models_dict), ncols=ncols)
        
        for i, model_name in enumerate(self.models_dict.keys()):
            model = self.models_dict[model_name]
            model.plot_cv_actual_vs_predicted(ax=axes[i], fig=fig)
            axes[i].set_title(model_name)
        
        delete_axes(fig, axes)   
        plt.tight_layout()
        plt.show();

    
    def compare_residuals_vs_predicted(self, ncols = 2):
        
        fig, axes = blank_plot(n_subplots=len(self.models_dict), ncols=ncols)
        
        for i, model_name in enumerate(self.models_dict.keys()):
            model = self.models_dict[model_name]
            model.plot_cv_residuals_vs_predicted(ax= axes[i], fig=fig)
            axes[i].set_title(model_name)
        
        delete_axes(fig, axes)    
        plt.tight_layout()
        plt.show();
        
    
    def compare_residuals_distribution(self, ncols = 2):
        
        fig, axes = blank_plot(n_subplots=len(self.models_dict), ncols=ncols)
        
        for i, model_name in enumerate(self.models_dict.keys()):
            model = self.models_dict[model_name]
            model.plot_cv_residuals_distribution(ax=axes[i], fig=fig)
            axes[i].set_title(model_name)
        
        delete_axes(fig, axes)   
        plt.tight_layout()
        plt.show();
        
    def compare_qq_residuals(self, ncols = 2):
        
        fig, axes = blank_plot(n_subplots=len(self.models_dict), ncols=ncols)
        
        for i, model_name in enumerate(self.models_dict.keys()):
            model = self.models_dict[model_name]
            model.plot_cv_qq_residuals(ax=axes[i], fig=fig)
            axes[i].set_title(model_name)
        
        delete_axes(fig, axes)
        plt.tight_layout()
        plt.show();
        
    def compare_cv_eval_scores_boxplot(self, ncols = 2):
    
        metrics = list(self.average_scores.index)
        
        fig, axes = blank_plot(n_subplots=len(metrics), ncols=ncols, base_size=(8, 8))
        models_labels = list(self.models_dict.keys())

        for i, metric in enumerate(metrics):
            
            models_scores= [self.models_dict[model_name].cv_eval_scores.loc[metric] for model_name in self.models_dict.keys()]
                
            axes[i].boxplot(models_scores, tick_labels = models_labels)
            axes[i].set_ylabel(metric)

        delete_axes(fig, axes)
        plt.tight_layout()
        plt.show()
        
    def compare_cv_residual_vs_predictor(self, feature_name, ncols = 2):
        
        fig, axes = blank_plot(n_subplots=len(self.models_dict), ncols=ncols)
        
        for i, model_name in enumerate(self.models_dict.keys()):
            model = self.models_dict[model_name]
            model.plot_cv_residual_vs_predictor(cv_fold_dict=self.cv_folds, feature_name=feature_name, ax=axes[i], fig=fig)
            axes[i].set_title(model_name)
        
        delete_axes(fig, axes)
        plt.tight_layout()
        plt.show();
        fig.save()
        
    def compare_cv_residual_vs_all_predictors(self, ncols = 2):
        
        for model_name in self.models_dict.keys():
            model = self.models_dict[model_name]
            model.plot_cv_residual_vs_all_predictors(cv_folds_dict=self.cv_folds)
            
            
    def plot_all_models_residuals_on_map(self):
        
        for model_name in self.models_dict.keys():
            model = self.models_dict[model_name]
            model.plot_cv_residuals_on_map(cv_folds_dict=self.cv_folds)
            
    def plot_all_models_residuals_vs_time(self):
        
        for model_name in self.models_dict.keys():
            model = self.models_dict[model_name]
            model.plot_cv_residuals_vs_time(cv_folds_dict=self.cv_folds)
        
    def run_full_cv_assessment(self, overwrite = True, export = True, ncols = 2):
        
        self.run_cv_for_all_models(overwrite=overwrite, export=export)
        self.update_summary_tables()
        self.export_experiment_results()
        
        self.compare_cv_eval_scores_boxplot(ncols=ncols)
        self.compare_actual_vs_predicted(ncols=ncols)
        self.compare_residuals_vs_predicted(ncols=ncols)   
        self.compare_residuals_distribution(ncols=ncols)
        self.compare_qq_residuals(ncols=ncols)
        self.compare_cv_residual_vs_all_predictors()
        self.plot_all_models_residuals_on_map()
        self.plot_all_models_residuals_vs_time()

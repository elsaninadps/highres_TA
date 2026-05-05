import sklearn.metrics as metrics
from scipy.special import huber
import pandas as pd
from loguru import logger
import numpy as np
from typing import Literal
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from cartopy.crs import PlateCarree

INDEXES_LITERAL = Literal["expocode", "time", "lat", "lon", "salinity_bin", 'is_coastal']


@dataclass
class TestSet:
        test_x: pd.DataFrame
        predictions_df : pd.DataFrame
        scores: pd.DataFrame
        label: str 
        color : str 
        marker: str
        linestyle: str
        nsamples: int
        noise: np.ndarray | None | pd.Series =None
        
        
def get_set_props(i):
    
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = [m for m in Line2D.markers if isinstance(m, str) and len(m) == 1]
    linestyles = list(Line2D.lineStyles.keys())

    return {
        "color": colors[i % len(colors)],
        "marker": markers[i % len(markers)],
        "linestyle": linestyles[i % len(linestyles)],
    }
        

def scoring_from_df(prediction_df:pd.DataFrame):
    
    y_true = prediction_df.y_true
    y_pred = prediction_df.y_pred
  
    scores = scoring(y_true, y_pred)
    return scores

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
            "nsample":len(y_true),
        },
        name="Scores",
    ).to_frame()
    scores.index.name = "Metric"

    logger.debug(f"\n{scores.to_markdown(floatfmt='.3f')}")

    return scores.Scores
    

def make_prediction_df(y_pred: np.ndarray, y_true: pd.Series, test_x, add_levels = True) -> pd.DataFrame:
    
    assert y_true.shape == y_pred.shape 

    # --- Base dataframe ---
    predictions_df = pd.DataFrame({
        "y_pred": y_pred,
        "y_true": y_true,
    }, index=y_true.index)
    

    predictions_df["residuals"] = y_pred - y_true
    
    predictions_df = predictions_df.join(test_x)
    
    if add_levels:
        predictions_df = add_some_index_levels(predictions_df)

    return predictions_df


def add_some_index_levels(df):
    
        # --- Extract time index ---
    time_idx = df.index.get_level_values("time")

    # --- Derived temporal features ---
    year = time_idx.year
    month = time_idx.month

    # day of year (1–365/366)
    doy = time_idx.dayofyear

    # 8-day bin (1-based)
    bin_8d = ((doy - 1) // 8) + 1
    
    
    lat = df.index.get_level_values("lat")

    lat_bin = np.where(
        lat < -30,
        "south_extratropics",
        np.where(
            lat <= 30,
            "tropics",
            "north_extratropics"
        )
    )

    # --- Build new MultiIndex ---
    # Keep existing index levels
    index_df = df.index.to_frame(index=False)

    index_df["lat_bin"] = lat_bin
    index_df["year"] = year.values
    index_df["month"] = month.values
    index_df["8D_bin"] = bin_8d.values

    new_index = pd.MultiIndex.from_frame(index_df)

    df.index = new_index
    
    return df

def make_multilevel_groups(df, time_grouping_level="year", secondary_grouping_level=None):
    
    if secondary_grouping_level is None:
        grouped_df = df.groupby(
            df.index.get_level_values(time_grouping_level)
        )
    else:
        grouped_df = df.groupby([
            df.index.get_level_values(time_grouping_level),
            df.index.get_level_values(secondary_grouping_level)
        ])
        
    return grouped_df

def plot_salinity_distribution(test_set, fig = None):
    
    if fig == None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        ax = fig.get_axes()[-1]
    
    sal = test_set.test_x['salinity']
    color = test_set.color
    label = test_set.label
    linestyle = test_set.linestyle
    nsamples = test_set.nsamples
    
    # --- Left: salinity distributions ---
    ax.hist(
        sal,
        bins=50,
        density=True,
        alpha=0.4,
        label=f"{label} (n={nsamples})",
        color = color
    )

    sal.plot(kind="kde", ax=ax, linewidth=1, label=f"{label} KDE", color= color, linestyle=linestyle)


    ax.set_title("Salinity Distribution")
    ax.set_xlabel("Salinity")
    ax.grid(True)
    ax.legend()
    
    return fig

def plot_metrics_timeseries(
    test_set: TestSet,
    time_grouping_level="year",
    secondary_grouping_level: None | str = None,
    fig=None
):
    """
    Groups predictions_df and plots metrics over time with optional secondary grouping.
    Colors are fixed per group for consistency.
    """

    predictions_df = test_set.predictions_df
    linestyle = test_set.linestyle
    marker = test_set.marker
    label = test_set.label
    

    grouped_df = make_multilevel_groups(predictions_df, time_grouping_level, secondary_grouping_level)
    groups = predictions_df.index.get_level_values(secondary_grouping_level).unique() if secondary_grouping_level else [None]
    grouped_scores = grouped_df.apply(lambda df: scoring_from_df(df))
    
    metrics = grouped_scores.columns
    n_metrics = len(metrics)

    # --- COLOR MAP (FIXED ACROSS RUNS) ---
    cmap = plt.get_cmap("tab10")
    color_map = {
        group: cmap(i % 10)
        for i, group in enumerate(sorted(groups) if secondary_grouping_level else groups)
    }

    # --- FIGURE SETUP ---
    if fig is None:
        fig, axes = plt.subplots(
            n_metrics, 1,
            figsize=(10, 4 * n_metrics),
            sharex=True
        )
    else:
        axes = fig.get_axes()
        

    # --- PLOTTING ---
    for i, metric in enumerate(metrics):
        ax = axes[i]

        if secondary_grouping_level is None:
            ax.plot(
                grouped_scores.index,
                grouped_scores[metric],
                linestyle=linestyle,
                marker = marker,
                color="black",
                label = label
            )
        else:
            for group in groups:
                scores = grouped_scores.xs(group, level=secondary_grouping_level)
                ax.plot(
                    scores.index.get_level_values(0),
                    scores[metric],
                    linestyle=linestyle,
                    color=color_map[group],
                    label= f"{label}, {secondary_grouping_level}={group}"
                )

        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel(time_grouping_level)    
    if time_grouping_level == "month":
        import calendar
        axes[i].set_xticks(range(1, 13))
        axes[i].set_xticklabels(calendar.month_abbr[1:])
        
    fig.tight_layout()

    return fig

def scatter_residuals_vs_target(test_set, fig = None):
    
    
    logger.info(f"calling on scatter with feature = 'y_true")
    fig = scatter_residuals_vs_feature(test_set, feature = 'y_true', fig = fig)
    
    fig.suptitle("Test Residuals (Predicted - True)", fontsize=14, fontweight="bold")
    fig.get_axes()[0].set_xlabel("Observed Total Alkalinity (µmol/kg)")

    return fig

    
def plot_residuals_map(predictions_df, ax=None, **kwargs):

    cmap = plt.get_cmap("RdBu_r", lut=11)

    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1, projection=PlateCarree(205))
        ax.coastlines(lw=0.5)
    else:
        fig = ax.get_figure()

    coords = predictions_df.index.to_frame()
    props = dict(cmap=cmap, s=10, vmin=-22, vmax=22, transform=PlateCarree()) | kwargs
    ax.scatter(
        coords["lon"],
        coords["lat"],
        c=predictions_df["residuals"] ,
        **props,
    )
    
    #TODO: add nsamples on map

    return fig, ax

def plot_residuals_feature_kde(test_set, feature, fig = None):
    
    # --- FIGURE SETUP ---
    if fig is None:
        fig, axes = plt.subplots(figsize=(10, 5))
    else:
        ax = fig.get_axes()[-1] #last added ax
        
    data = test_set.predictions_df[[feature, "residuals"]].dropna()
    
    # --- KDE plot ---
    kde = sns.kdeplot(
        data=data,
        x=feature,
        y="residuals",
        fill=True,
        levels=30,
        cmap="viridis",
        ax=ax
    )

    # --- colorbar (use last collection) ---
    mappable = ax.collections[-1]
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label("Density")

    ax.axhline(0, color="white", linestyle="--", linewidth=1)

    ax.set_title(f"{test_set.label} (n={test_set.nsamples})- Joint Density: {feature} vs Residuals")
    ax.set_xlabel(feature)
    ax.set_ylabel("Residuals")

    return fig


def plot_scores_table(scores, label, fig = None):
    # plot scores as table in figure
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 3))

    ax.axis("off")
    ax.table(scores.round(3), loc="center", cellLoc="center")
    ax.set_title(f"{label}", fontsize=14, fontweight="bold")

    return fig


def scatter_residuals_vs_feature(test_set, feature, fig=None):
    
    linestyle = test_set.linestyle
    label = test_set.label
    color = test_set.color
    df = test_set.predictions_df
    
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        ax = fig.get_axes()[0]

    logger.info(f"plotting residuals vs {feature}")
    ax.scatter(df[feature], df["residuals"], alpha=0.3, s=10, label =f"{label} (n={test_set.nsamples})", color = color)

    # optional smoothing (binned mean)
    bins = np.linspace(df[feature].min(), df[feature].max(), 50)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    digitized = np.digitize(df[feature], bins)

    means = [
        df["residuals"][digitized == i].mean()
        for i in range(1, len(bins))
    ]

    ax.plot(bin_centers, means, color="black",  linewidth=1, linestyle = linestyle, label = f"{label} trend")


    ax.set_xlabel(feature)
    ax.set_ylabel("Residual (µmol/kg)")
    ax.set_title(f"Residuals vs {feature}")
    ax.grid(True)
    ax.legend()

    return fig


def plot_residuals_distribution(test_set, fig=None, x_range=None):
    
    residuals = test_set.predictions_df["residuals"]
    color = test_set.color
    label = test_set.label

    if fig is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        ax = fig.get_axes()[0]

    
    ax.hist(
        residuals,
        bins=50,
        density=True,
        alpha=0.5,
        color=color,
        label=f"{label} (n={test_set.nsamples})",
        range=x_range
    )

    residuals.plot(
        kind="kde",
        ax=ax,
        color=color,
        linewidth=1,
        label=f"{label} KDE"
    )

    if x_range is not None:
        ax.set_xlim(x_range)  # <-- critical
    else:
        
        xmin = residuals.min()
        xmax = residuals.max()
        x_range = (xmin, xmax)
        ax.set_xlim(x_range)
        

    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual")
    ax.grid(True)
    ax.legend()

    return fig


# def plot_residuals_feature_swarm(df, feature, sample=10000):

#     data = df[[feature, "residuals"]].copy()

#     # downsample only for visualization stability
#     if len(data) > sample:
#         data = data.sample(sample, random_state=0)

#     fig, ax = plt.subplots(figsize=(9, 5))

#     sns.scatterplot(
#         data=data,
#         x=feature,
#         y="residuals",
#         alpha=0.25,
#         s=10,
#         ax=ax
#     )

#     ax.axhline(0, color="black", linestyle="--", linewidth=1)

#     ax.set_title(f"Residual Structure vs {feature}")
#     ax.set_xlabel(feature)
#     ax.set_ylabel("Residuals")
#     ax.grid(True)

#     return fig


# def plot_residuals_violin(df, feature, n_bins=10):

#     # bin continuous feature
#     df = df.copy()
#     df["feature_bin"] = pd.qcut(df[feature], q=n_bins, duplicates="drop")

#     fig, ax = plt.subplots(figsize=(10, 5))

#     sns.violinplot(
#         data=df,
#         x="feature_bin",
#         y="residuals",
#         ax=ax,
#         inner="quartile",
#         cut=0
#     )

#     ax.set_title(f"Residual Distribution vs {feature}")
#     ax.set_xlabel(feature + " (binned)")
#     ax.set_ylabel("Residuals")
#     ax.tick_params(axis='x', rotation=45)
#     ax.grid(True)

#     return fig


# def plot_residual_swarm(df, feature, n_bins=8, sample=5000):

#     data = df[[feature, "residuals"]].dropna().copy()

#     # optional downsampling (important for swarm plots)
#     if len(data) > sample:
#         data = data.sample(sample, random_state=0)

#     data["feature_bin"] = pd.qcut(data[feature], q=n_bins, duplicates="drop")

#     fig, ax = plt.subplots(figsize=(10, 4))

#     sns.swarmplot(
#         data=data,
#         x="feature_bin",
#         y="residuals",
#         ax=ax,
#         size=2
#     )

#     ax.axhline(0, color="black", linestyle="--", linewidth=1)

#     ax.set_title(f"Residual swarm vs {feature} (binned)")
#     ax.set_xlabel(feature)
#     ax.set_ylabel("Residual")

#     plt.xticks(rotation=45)
#     ax.grid(True)

#     return fig


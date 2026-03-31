import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn import model_selection


def read_yaml(fname: str):

    with open(fname) as fobj:
        config = yaml.safe_load(fobj)

    # config = munch.munchify(config)

    return config


def columns_list(all_columns, prefix):

    all_time_cols = [i for i in all_columns if "time" in i]

    all_lat_cols = [i for i in all_columns if "lat" in i]

    all_lon_cols = [i for i in all_columns if "lon" in i]

    # all_coord_cols = [i for i in all_columns if ('lat' in i or 'lon' in i or 'time' in i)]


def offline_GLODAP_import(parent_path):

    # import from local file
    df = pd.read_csv(f"{parent_path}/data_source/matched_GLODAP2023.csv")

    # Naming convention and drop sampledepth and index column
    renaming_dict = read_yaml(f"{parent_path}/data_source/renaming_dict.yaml")

    df = df.drop(columns=["maxsampdepth", df.columns[0]]).rename(columns=renaming_dict)

    # column names
    all_cols = df.columns.to_list()
    all_time_cols = [i for i in all_cols if "time" in i]

    # reattribute correct datetime types
    df = df.astype({col: "datetime64[ns]" for col in all_time_cols}).astype(
        {"expocode_gp": "string"}
    )

    # move alk column to the front
    col = df.pop("talk_gp")
    df.insert(0, "talk_gp", col)

    # print_info
    print(df.info())

    return df


def online_GLODAP_import(parent_path):

    url = "https://data.up.ethz.ch/shared/OceanSODA-ETHZv2/total_alkalinity/GLODAPv2023/GLODAPv2023-raw_collocated-{y}.pq"
    df = pd.concat([pd.read_parquet(url.format(y=y)) for y in range(1982, 2022)])

    # renaming_dict = read_yaml(f"{parent_path}/data_source/renaming_dict.yaml")
    # df = df.rename(columns = renaming_dict)

    # move talk column to the front
    col = df.pop("talk_gp")
    df.insert(0, "talk_gp", col)

    # # store column names
    # all_cols = df.columns.to_list()

    # all_time_cols = [i for i in all_cols if 'time' in i]
    # all_coord_cols = [i for i in all_cols if ('lat' in i or 'lon' in i or 'time' in i)]

    # print info
    print(df.info())

    return df


def keep_predictors(df):

    drop_cols = (
        [
            "talk_gp",
            "talk_gp_normalized",
            "year_gp",
            "expocode_gp",
            "salinity_bin",
            "tco2_gp",
            "fco2_gp",
            "silicate_gp",
            "phtsinsitutp_gp",
            "oxygen_gp",
            "nitrate_gp",
            "nitrite_gp",
            "phosphate_gp",
            "aou_gp",
            "salinity_soda",
            "temp_soda",
        ]
        + df.filter(regex="time").columns.tolist()
        + df.filter(regex="lat|lon").columns.tolist()
        + df.filter(regex="uncert|error").columns.tolist()
        + df.filter(regex="depth").columns.tolist()
        + df.filter(regex="sss").columns.tolist()
        + df.filter(regex="flag").columns.tolist()
        + df.filter(regex="qc").columns.tolist()
    )

    x_training_df = df.drop(columns=drop_cols)

    return x_training_df


def plot_split(
    source_df,
    train_idx,
    test_idx,
    label_train="train",
    label_eval="test",
    train_color="C0",
    eval_color="C1",
    title="GLODAPv2",
):

    train_expo = set(source_df.expocode_gp.iloc[train_idx].unique())
    test_expo = set(source_df.expocode_gp.iloc[test_idx].unique())
    assert len(train_expo & test_expo) == 0, (
        "can't have overlapping expocodes in train and test sets"
    )

    source_df.iloc[train_idx].expocode_gp.value_counts().plot(
        kind="hist", bins=50, alpha=0.5, label=label_train
    )
    source_df.iloc[test_idx].expocode_gp.value_counts().plot(
        kind="hist", bins=50, alpha=0.5, label=label_eval
    )
    plt.legend()

    salinity_bin_labels = list(source_df.salinity_bin.unique())

    split_stats = pd.concat(
        [
            source_df["salinity_bin"].iloc[train_idx].value_counts()[salinity_bin_labels],
            source_df["salinity_bin"].iloc[test_idx].value_counts()[salinity_bin_labels],
        ],
        axis=1,
        keys=[label_train, label_eval],
    )

    fig, ax = plt.subplots(figsize=[12, 5])
    source_df.iloc[train_idx].plot(
        x="lon_gp", y="lat_gp", c=train_color, kind="scatter", alpha=0.5, label=label_train, ax=ax
    )
    source_df.iloc[test_idx].plot(
        x="lon_gp", y="lat_gp", c=eval_color, kind="scatter", alpha=0.5, label=label_eval, ax=ax
    )
    ax.legend()

    fig.suptitle(title)
    plt.tight_layout()

    split_stats

    plt.show()


def traintest_salinity_expocode_split(source_df):

    splitter = model_selection.StratifiedGroupKFold(n_splits=5)
    splits = splitter.split(
        X=source_df, y=source_df["salinity_bin"], groups=source_df["expocode_gp"]
    )

    train_idx, test_idx = next(splits)

    train_split_df = source_df.iloc[train_idx]
    test_split_df = source_df.iloc[test_idx]

    plot_split(source_df=source_df, train_idx=train_idx, test_idx=test_idx)

    return train_split_df, test_split_df


def cv_salinity_expocode_split(
    source_train_df: pd.DataFrame,
    n_splits=5,
    normalized_y=False,
    random_seed=42,
    plot_cv_splits=True,
):
    """Cross validation folds using StratifiedGroupKFold, based on salinity bin (class stratification) and expocode (group)

    source_train_df : original df with all variables, minus the heldout test set

    Returns:
        dict: contains each train/validation splits per fold (key)
    """

    # TODO: make possible random shuffling

    cv_splitter = model_selection.StratifiedGroupKFold(n_splits=n_splits)
    cv_splits = list(
        cv_splitter.split(
            source_train_df, source_train_df["salinity_bin"], groups=source_train_df["expocode_gp"]
        )
    )

    # Extract the first validation split from cv_splits
    cv_folds = dict()

    x_training = keep_predictors(source_train_df)
    y_training = (
        source_train_df["talk_gp_normalized"] if normalized_y else source_train_df["talk_gp"]
    )

    for i, (train_folds_idx, validation_fold_idx) in enumerate(cv_splits):
        cv_folds[f"fold{i}"] = {
            "x_train": x_training.iloc[train_folds_idx],
            "y_train": y_training.iloc[train_folds_idx],
            "x_validation": x_training.iloc[validation_fold_idx],
            "y_validation": y_training.iloc[validation_fold_idx],
            "train_idx": train_folds_idx,
            "validation_idx": validation_fold_idx,
            "train_lat": source_train_df["lat_gp"].iloc[train_folds_idx],
            "train_lon": source_train_df["lon_gp"].iloc[train_folds_idx],
            "train_time": source_train_df["time_gp"].iloc[train_folds_idx],
            "train_expocode": source_train_df["expocode_gp"].iloc[train_folds_idx],
            "validation_lat": source_train_df["lat_gp"].iloc[validation_fold_idx],
            "validation_lon": source_train_df["lon_gp"].iloc[validation_fold_idx],
            "validation_time": source_train_df["time_gp"].iloc[validation_fold_idx],
            "validation_expocode": source_train_df["expocode_gp"].iloc[validation_fold_idx],
        }

        if plot_cv_splits == True:
            plot_split(
                source_train_df,
                train_folds_idx,
                validation_fold_idx,
                label_train="train (4/5)",
                label_eval="validation (1/5)",
                eval_color="C2",
                title=f"fold {i}",
            )

    return cv_folds

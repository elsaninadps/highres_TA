"""Tune and fit a quantile ensemble over data splits and HPO replicates.

The prepared data is partitioned once into ``n_parts`` stratified, cruise-grouped
blocks using a fixed ``split_random_state``. Block 0 is the test set for every
ensemble member and is never trained on, so the assembled ensemble keeps an
honest held-out score.

The ensemble is a nested two-factor design. Split ``k`` (1..n_parts-1) reserves
block ``k`` for validation and trains on the remaining blocks. Within each split,
``n_hpo_replicates`` independent Optuna studies each tune and fit one member
(seed ``100 * k + r``). Hyperparameters are tuned by leave-one-block-out
cross-validation inside the training blocks only, with folds fixed per split, so
replicates differ only through the optimisation path and model randomness. The
test-set variance decomposes into a data-split term and an HPO term.

Examples
--------
Smoke-test one member::

    uv run python scripts/train_quantile_ensemble.py --stage train --split 1 \
        --hpo-replicate 0 --n-trials 2 --optuna-jobs 1

Check the splits without training anything::

    uv run python scripts/train_quantile_ensemble.py --dry-run

Run or resume every member and assemble the ensemble::

    uv run python scripts/train_quantile_ensemble.py --stage all
"""

import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Annotated, Any

import dotenv
import numpy as np
import optuna
import pandas as pd
import typer
import yaml
from loguru import logger

import highres_ta as ta

ROOT = Path(dotenv.find_dotenv("pyproject.toml")).parent
DEFAULT_CONFIG = ROOT / "scripts" / "quantile_ensemble_config.yaml"
SPLIT_NAMES = ("train", "validation", "test")
app = typer.Typer()


@dataclass(frozen=True)
class Member:
    split: int
    replicate: int

    @property
    def seed(self) -> int:
        return 100 * self.split + self.replicate

    @property
    def name(self) -> str:
        return f"split_{self.split}_hpo_{self.replicate}"


class Stage(str, Enum):
    TRAIN = "train"
    ASSEMBLE = "assemble"
    ALL = "all"


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration in {path} must be a mapping.")
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def prepare_data(config: dict[str, Any]) -> pd.DataFrame:
    data_config = config["data"]
    coordinate_columns = data_config["coordinate_columns"]
    feature_names = data_config["feature_names"]
    engineered_feature_names = data_config["engineered_feature_names"]
    raw_feature_names = [name for name in feature_names if name not in engineered_feature_names]
    target_name = data_config["target_name"]
    quality_columns = data_config["quality_columns"]
    required_columns = list(
        dict.fromkeys(coordinate_columns + raw_feature_names + [target_name] + quality_columns)
    )

    data = (
        ta
        .load_data(str(resolve_path(data_config["parquet_glob"])))[required_columns]
        .set_index(coordinate_columns, drop=False)
        .pipe(
            ta.drop_extreme_salinities,
            min=data_config["minimum_salinity"],
            max=data_config["maximum_salinity"],
        )
        .pipe(
            ta.add_talk_adjustment,
            fname=str(resolve_path(data_config["talk_adjustment_csv"])),
        )
        .pipe(ta.drop_bad_quality_talk)
        .dropna(subset=data_config["non_null_columns"])
        .drop_duplicates(subset=coordinate_columns, keep="first")
        .select_dtypes(include=[np.number])
        .pipe(ta.add_cyclical_dayofyear)
        .pipe(ta.add_spherical_coords)
        .loc[:, feature_names + [target_name]]
    )
    logger.info("Prepared {:,} observations with {} features", len(data), len(feature_names))
    return data


def data_fingerprint(data: pd.DataFrame) -> str:
    row_hashes = pd.util.hash_pandas_object(data, index=True).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def quantiles(config: dict[str, Any]) -> np.ndarray:
    return np.asarray(config["model"]["quantiles"], dtype=float)


def interval_config(config: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return {
        str(label): (float(bounds[0]), float(bounds[1]))
        for label, bounds in config["evaluation"]["intervals"].items()
    }


def fixed_catboost_params(config: dict[str, Any], seed: int) -> dict[str, Any]:
    model_config = config["model"]
    alpha_string = ",".join(f"{alpha:g}" for alpha in quantiles(config))
    return {
        "loss_function": f"MultiQuantile:alpha={alpha_string}",
        "iterations": model_config["maximum_iterations"],
        "random_seed": seed,
        "allow_writing_files": False,
        "thread_count": model_config["trial_thread_count"],
        "verbose": False,
        "early_stopping_rounds": model_config["early_stopping_rounds"],
    }


def suggest_parameters(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 100.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "depth": trial.suggest_int("depth", 4, 8),
        "random_strength": trial.suggest_float("random_strength", 1.0, 10.0),
    }


def make_model(
    config: dict[str, Any], seed: int, catboost_params: dict[str, Any]
) -> ta.CatBoostResidualRegressor:
    return ta.CatBoostResidualRegressor(
        linear_features=config["model"]["linear_features"],
        feature_names=config["data"]["feature_names"],
        polynomial_degree=config["model"]["polynomial_degree"],
        **catboost_params,
    )


def validation_loss(model: ta.CatBoostResidualRegressor) -> float:
    scores = model.boosting_model_.get_best_score()["validation"]
    loss_name = model.loss_function
    if loss_name in scores:
        return float(scores[loss_name])
    if len(scores) == 1:
        return float(next(iter(scores.values())))
    raise KeyError(f"Could not identify {loss_name!r} in validation scores: {scores}")


def objective_factory(
    data: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    seed: int,
):
    feature_names = config["data"]["feature_names"]
    target_name = config["data"]["target_name"]
    alpha = quantiles(config)
    intervals = interval_config(config)
    fixed_params = fixed_catboost_params(config, seed)

    def objective(trial: optuna.Trial) -> float:
        params = fixed_params | suggest_parameters(trial)
        fold_losses: list[float] = []
        fold_scores: list[dict[str, float]] = []
        fold_best_iterations: list[int] = []

        for fold_index, (train_index, validation_index) in enumerate(folds):
            train = data.iloc[train_index]
            validation = data.iloc[validation_index]
            model = make_model(config, seed, params)
            model.fit(
                train.loc[:, feature_names],
                train.loc[:, target_name],
                eval_set=(
                    validation.loc[:, feature_names],
                    validation.loc[:, target_name],
                ),
            )

            prediction = model.predict_quantiles(validation.loc[:, feature_names])
            fold_losses.append(validation_loss(model))
            fold_scores.append(
                ta.score_quantile_predictions(
                    validation.loc[:, target_name], prediction, alpha, intervals
                )
            )
            fold_best_iterations.append(model.boosting_model_.get_best_iteration() + 1)
            trial.report(float(np.median(fold_losses)), step=fold_index)

        trial.set_user_attr("fold_multi_quantile_loss", fold_losses)
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("fold_best_iterations", fold_best_iterations)
        trial.set_user_attr("refit_iterations", int(np.median(fold_best_iterations)))
        return float(np.median(fold_losses))

    return objective


def optimize(
    data: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    member: Member,
    storage_path: Path,
    study_name: str,
    target_completed_trials: int,
    optuna_jobs: int,
) -> optuna.Study:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=member.seed, n_startup_trials=20),
        storage=f"sqlite:///{storage_path}",
        study_name=study_name,
        load_if_exists=True,
    )
    fold_digest = hashlib.sha256()
    for train_index, validation_index in folds:
        fold_digest.update(np.asarray(train_index, dtype=np.int64).tobytes())
        fold_digest.update(np.asarray(validation_index, dtype=np.int64).tobytes())
    signature_payload = {
        "data_fingerprint": data_fingerprint(data),
        "fold_digest": fold_digest.hexdigest(),
        "features": config["data"]["feature_names"],
        "target": config["data"]["target_name"],
        "model": config["model"],
        "split": member.split,
        "replicate": member.replicate,
        "seed": member.seed,
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    existing_signature = study.user_attrs.get("study_signature")
    if existing_signature is not None and existing_signature != signature:
        raise ValueError(
            f"Study {study_name!r} was created for different data, folds, or model "
            "settings. Use a new output directory or remove the mismatched study."
        )
    study.set_user_attr("study_signature", signature)
    study.set_metric_names(["multi_quantile_loss"])
    completed = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    remaining = max(0, target_completed_trials - completed)
    logger.info(
        "Study {} has {}/{} completed trials; running {}",
        study_name,
        completed,
        target_completed_trials,
        remaining,
    )
    if remaining:
        study.optimize(
            objective_factory(data, folds, config, member.seed),
            n_trials=remaining,
            n_jobs=optuna_jobs,
            show_progress_bar=optuna_jobs == 1,
        )
    return study


def selected_params(
    study: optuna.Study, config: dict[str, Any], seed: int, final_fit: bool
) -> dict[str, Any]:
    params = fixed_catboost_params(config, seed) | study.best_trial.params
    params["iterations"] = study.best_trial.user_attrs["refit_iterations"]
    params.pop("early_stopping_rounds", None)
    if final_fit:
        params["thread_count"] = config["model"]["final_thread_count"]
    return params


def score_subsets(
    data: pd.DataFrame, prediction: np.ndarray, config: dict[str, Any]
) -> dict[str, dict[str, float]]:
    target_name = config["data"]["target_name"]
    alpha = quantiles(config)
    intervals = interval_config(config)
    scores = {"all": ta.score_quantile_predictions(data[target_name], prediction, alpha, intervals)}
    minimum_bottomdepth = config["evaluation"].get("minimum_bottomdepth")
    if minimum_bottomdepth is not None:
        mask = data["bottomdepth"].to_numpy() > minimum_bottomdepth
        if np.any(mask):
            scores["deep"] = ta.score_quantile_predictions(
                data.loc[mask, target_name], prediction[mask], alpha, intervals
            )
    return scores


def prediction_frame(
    data: pd.DataFrame,
    row_ids: np.ndarray,
    prediction: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    target_name = config["data"]["target_name"]
    frame = pd.DataFrame({
        "row_id": row_ids,
        "observed": data[target_name].to_numpy(),
        "bottomdepth": data["bottomdepth"].to_numpy(),
    })
    for index, alpha in enumerate(quantiles(config)):
        frame[f"q_{alpha:g}"] = prediction[:, index]
    return frame


def prediction_columns(config: dict[str, Any]) -> list[str]:
    return [f"q_{alpha:g}" for alpha in quantiles(config)]


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}.")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")
    temporary_path.replace(path)


def members(
    config: dict[str, Any], split: int | None = None, replicate: int | None = None
) -> list[Member]:
    n_splits = int(config["cross_validation"]["n_parts"]) - 1
    n_replicates = int(config["ensemble"]["n_hpo_replicates"])
    if split is not None and not 1 <= split <= n_splits:
        raise ValueError(f"Split {split} is outside 1..{n_splits}.")
    if replicate is not None and not 0 <= replicate < n_replicates:
        raise ValueError(f"HPO replicate {replicate} is outside 0..{n_replicates - 1}.")
    return [
        Member(k, r)
        for k in range(1, n_splits + 1)
        for r in range(n_replicates)
        if split in (None, k) and replicate in (None, r)
    ]


def outer_blocks(data: pd.DataFrame, config: dict[str, Any]) -> list[np.ndarray]:
    """Return the shared partition. Block 0 always tests, so no member trains on it."""
    cv_config = config["cross_validation"]
    n_parts = int(cv_config["n_parts"])
    if n_parts < 3:
        raise ValueError("cross_validation.n_parts must be at least 3.")
    return [
        test_index
        for _, test_index in ta.make_train_test_folds(
            data,
            n_splits=n_parts,
            shuffle=True,
            random_state=int(cv_config["split_random_state"]),
        )
    ]


def split_indices(blocks: Sequence[np.ndarray], split: int) -> dict[str, Any]:
    """Return train/validation/test positional indices when block ``split`` validates."""
    if not 1 <= split < len(blocks):
        raise ValueError(f"Split {split} is outside 1..{len(blocks) - 1}.")
    train_blocks = [
        np.sort(block) for index, block in enumerate(blocks) if index not in (0, split)
    ]
    return {
        "train": np.sort(np.concatenate(train_blocks)),
        "validation": np.sort(blocks[split]),
        "test": np.sort(blocks[0]),
        "train_blocks": train_blocks,
    }


def tuning_folds(split: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-block-out folds as positions within ``split["train"]``."""
    train = split["train"]
    blocks = split["train_blocks"]
    folds = []
    for held_out, block in enumerate(blocks):
        fit = np.concatenate([other for index, other in enumerate(blocks) if index != held_out])
        folds.append((np.searchsorted(train, np.sort(fit)), np.searchsorted(train, block)))
    return folds


def validate_split(data: pd.DataFrame, split: dict[str, Any], index: int) -> None:
    covered = np.concatenate([split[name] for name in SPLIT_NAMES])
    if len(covered) != len(data) or len(np.unique(covered)) != len(data):
        raise ValueError(f"Split {index} does not cover every row exactly once.")
    groups = data.index.get_level_values("expocode").to_numpy()
    for left, right in combinations(SPLIT_NAMES, 2):
        overlap = set(groups[split[left]]).intersection(groups[split[right]])
        if overlap:
            raise ValueError(f"Split {index} leaks cruises between {left} and {right}: {overlap}")
    logger.info(
        "Split {}: {:,} train / {:,} validation / {:,} test observations",
        index,
        len(split["train"]),
        len(split["validation"]),
        len(split["test"]),
    )


def write_split(output_dir: Path, data: pd.DataFrame, split: dict[str, Any], index: int):
    frame = pd.DataFrame({
        "row_id": np.concatenate([split[name] for name in SPLIT_NAMES]),
        "split": np.repeat(SPLIT_NAMES, [len(split[name]) for name in SPLIT_NAMES]),
    }).sort_values("row_id", ignore_index=True)
    frame["expocode"] = data.index.get_level_values("expocode").to_numpy()[frame["row_id"]]
    path = output_dir / "splits" / f"split_{index}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def train_member(
    member: Member,
    data: pd.DataFrame,
    config: dict[str, Any],
    blocks: Sequence[np.ndarray],
    target_trials: int,
    optuna_jobs: int,
    fingerprint: str,
) -> Path:
    output_dir = resolve_path(config["output_dir"])
    feature_names = config["data"]["feature_names"]
    target_name = config["data"]["target_name"]

    split = split_indices(blocks, member.split)
    train = data.iloc[split["train"]]

    study = optimize(
        train,
        tuning_folds(split),
        config,
        member,
        output_dir / "studies" / f"{member.name}.sqlite3",
        f"catboost_quantile_{member.name}",
        target_trials,
        optuna_jobs,
    )
    model = make_model(
        config, member.seed, selected_params(study, config, member.seed, final_fit=True)
    )
    model.fit(train[feature_names], train[target_name])

    scores = {}
    for name in ("validation", "test"):
        subset = data.iloc[split[name]]
        prediction = model.predict_quantiles(subset[feature_names])
        scores[name] = score_subsets(subset, prediction, config)
        path = output_dir / "predictions" / f"{member.name}_{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        prediction_frame(subset, split[name], prediction, config).to_parquet(path, index=False)

    model.training_metadata_ = {
        "split": member.split,
        "replicate": member.replicate,
        "seed": member.seed,
        "data_fingerprint": fingerprint,
        "observation_counts": {name: len(split[name]) for name in SPLIT_NAMES},
        "cruise_counts": {
            name: len(set(data.index.get_level_values("expocode").to_numpy()[split[name]]))
            for name in SPLIT_NAMES
        },
        "best_trial": study.best_trial.number,
        "best_inner_cv_loss": study.best_value,
        "best_params": study.best_trial.params,
        "refit_iterations": study.best_trial.user_attrs["refit_iterations"],
        "scores": scores,
    }

    model_path = output_dir / "models" / f"quantile_{member.name}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    write_json(output_dir / "models" / f"quantile_{member.name}.json", model.training_metadata_)
    logger.success("Saved member {} to {}", member.name, model_path)
    return model_path


def summarize_ensemble_test(
    data: pd.DataFrame,
    config: dict[str, Any],
    members: Sequence[Member],
    fingerprint: str,
) -> None:
    output_dir = resolve_path(config["output_dir"])
    paths = [output_dir / "predictions" / f"{member.name}_test.parquet" for member in members]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot summarize the held-out test set; missing member predictions: "
            + ", ".join(str(path) for path in missing)
        )

    frames = [pd.read_parquet(path).sort_values("row_id", ignore_index=True) for path in paths]
    row_ids = frames[0]["row_id"].to_numpy()
    if any(not np.array_equal(frame["row_id"].to_numpy(), row_ids) for frame in frames):
        raise ValueError("Ensemble members were evaluated on different test rows.")

    columns = prediction_columns(config)
    splits = sorted({member.split for member in members})
    replicates = sorted({member.replicate for member in members})
    if len(members) != len(splits) * len(replicates):
        raise ValueError("The ensemble must be a balanced splits x replicates design.")
    member_predictions = np.stack([frame[columns].to_numpy() for frame in frames])
    predictions = member_predictions.reshape(
        len(splits), len(replicates), *member_predictions.shape[1:]
    )
    decomposition = ta.decompose_variance(predictions)
    prediction = decomposition["mean"]
    test = data.iloc[row_ids]

    frame = prediction_frame(test, row_ids, prediction, config)
    variance_means = {}
    for index, alpha in enumerate(quantiles(config)):
        frame[f"split_var_q_{alpha:g}"] = decomposition["split_var"][:, index]
        frame[f"hpo_var_q_{alpha:g}"] = decomposition["hpo_var"][:, index]
        variance_means[f"q_{alpha:g}"] = {
            "split_var": float(decomposition["split_var"][:, index].mean()),
            "hpo_var": float(decomposition["hpo_var"][:, index].mean()),
        }
    frame.to_parquet(output_dir / "predictions" / "ensemble_test.parquet", index=False)

    split_scores = {
        str(split): score_subsets(test, predictions[index].mean(axis=0), config)
        for index, split in enumerate(splits)
    }
    member_scores = {
        member.name: score_subsets(test, member_predictions[index], config)
        for index, member in enumerate(members)
    }
    write_json(
        output_dir / "ensemble_test_metrics.json",
        {
            "members": [member.name for member in members],
            "seeds": [member.seed for member in members],
            "data_fingerprint": fingerprint,
            "aggregation": "mean",
            "test_observations": len(row_ids),
            "scores": score_subsets(test, prediction, config),
            "variance_means": variance_means,
            "split_mean_scores": split_scores,
            "member_scores": member_scores,
        },
    )
    logger.success("Wrote held-out ensemble test metrics over {} members", len(members))


def assemble_ensemble(
    config: dict[str, Any], members: Sequence[Member], fingerprint: str
) -> Path:
    output_dir = resolve_path(config["output_dir"])
    model_paths = [output_dir / "models" / f"quantile_{member.name}.joblib" for member in members]
    missing = [path for path in model_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot assemble ensemble; missing members: " + ", ".join(str(path) for path in missing)
        )
    estimators = [ta.CatBoostResidualRegressor.load(path) for path in model_paths]
    ensemble = ta.QuantileRegressionEnsemble(
        estimators=estimators,
        seeds=[member.seed for member in members],
        quantiles=quantiles(config),
        aggregation=config["ensemble"]["aggregation"],
        member_splits=[member.split for member in members],
        member_replicates=[member.replicate for member in members],
    )
    ensemble_path = output_dir / "models" / "quantile_ensemble.joblib"
    ensemble.save(ensemble_path)
    write_json(
        output_dir / "models" / "manifest.json",
        {
            "quantiles": quantiles(config),
            "aggregation": config["ensemble"]["aggregation"],
            "data_fingerprint": fingerprint,
            "feature_names": config["data"]["feature_names"],
            "target_name": config["data"]["target_name"],
            "members": [
                {
                    "split": member.split,
                    "replicate": member.replicate,
                    "seed": member.seed,
                    "file": path.name,
                }
                for member, path in zip(members, model_paths)
            ],
            "ensemble": ensemble_path.name,
        },
    )
    logger.success("Saved {}-member ensemble to {}", len(members), ensemble_path)
    return ensemble_path


@app.command(help=__doc__)
def main(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the ensemble configuration file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_CONFIG,
    stage: Annotated[
        Stage,
        typer.Option(help="Pipeline stage to run. Existing Optuna studies are resumed."),
    ] = Stage.ALL,
    split: Annotated[
        int | None,
        typer.Option(help="Run only this split (validation block 1..n_parts-1)."),
    ] = None,
    hpo_replicate: Annotated[
        int | None,
        typer.Option(help="Run only this HPO replicate within each selected split."),
    ] = None,
    n_trials: Annotated[
        int | None,
        typer.Option(help="Override the target number of completed trials per study."),
    ] = None,
    optuna_jobs: Annotated[
        int | None,
        typer.Option(help="Override the number of concurrent Optuna trials."),
    ] = None,
    maximum_iterations: Annotated[
        int | None,
        typer.Option(help="Override CatBoost's maximum iterations (useful for smoke tests)."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Override the artifact directory."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(help="Load data and validate the requested splits without training."),
    ] = False,
) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    config = load_config(config_path)
    if maximum_iterations is not None:
        config["model"]["maximum_iterations"] = maximum_iterations
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    selected = members(config, split, hpo_replicate)
    all_members = members(config)
    target_trials = n_trials or int(config["optimization"]["n_trials"])
    resolved_optuna_jobs = optuna_jobs or int(config["optimization"]["n_jobs"])
    if target_trials < 1:
        raise ValueError("The target number of completed trials must be positive.")
    if resolved_optuna_jobs == 0:
        raise ValueError("Optuna jobs must be nonzero.")
    if config["model"]["maximum_iterations"] < 1:
        raise ValueError("CatBoost maximum iterations must be positive.")
    data = prepare_data(config)
    fingerprint = data_fingerprint(data)
    logger.info("Data fingerprint: {}", fingerprint)
    blocks = outer_blocks(data, config)
    selected_splits = sorted({member.split for member in selected})
    if dry_run:
        for index in selected_splits:
            validate_split(data, split_indices(blocks, index), index)
        logger.success("Dry run completed; no studies or models were written")
        return

    if stage in (Stage.TRAIN, Stage.ALL):
        output_dir = resolve_path(config["output_dir"])
        for index in selected_splits:
            split_dict = split_indices(blocks, index)
            validate_split(data, split_dict, index)
            write_split(output_dir, data, split_dict, index)
        for member in selected:
            train_member(
                member,
                data,
                config,
                blocks,
                target_trials,
                resolved_optuna_jobs,
                fingerprint,
            )

    filtered = split is not None or hpo_replicate is not None
    if stage is Stage.ASSEMBLE or (stage is Stage.ALL and not filtered):
        summarize_ensemble_test(data, config, all_members, fingerprint)
        assemble_ensemble(config, all_members, fingerprint)


if __name__ == "__main__":
    app()

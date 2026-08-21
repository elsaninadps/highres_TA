"""Tune, evaluate, and fit a seeded CatBoost quantile ensemble.

The evaluation stage uses nested, expedition-grouped cross-validation. Every
observation receives one genuinely outer-fold prediction per seed, allowing an
out-of-fold estimate for both individual members and the aggregated ensemble.
The training stage independently tunes and refits one full-data model per seed.

Examples
--------
Smoke-test one outer fold::

    uv run python scripts/train_quantile_ensemble.py --stage evaluate --seed 43 \
        --outer-fold 0 --n-trials 2 --optuna-jobs 1

Run or resume one complete member::

    uv run python scripts/train_quantile_ensemble.py --stage all --seed 43

Run or resume every member and assemble the ensemble::

    uv run python scripts/train_quantile_ensemble.py --stage all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import dotenv
import numpy as np
import optuna
import pandas as pd
import yaml
from loguru import logger

import highres_ta as ta


ROOT = Path(dotenv.find_dotenv("pyproject.toml")).parent
DEFAULT_CONFIG = ROOT / "scripts" / "quantile_ensemble_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        choices=("evaluate", "train", "assemble", "all"),
        default="all",
        help="Pipeline stage to run. Existing Optuna studies are resumed.",
    )
    parser.add_argument("--seed", type=int, help="Run only this configured member seed.")
    parser.add_argument(
        "--outer-fold",
        type=int,
        help="Evaluate only this zero-based outer fold (requires --seed).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        help="Override the target number of completed trials per study.",
    )
    parser.add_argument(
        "--optuna-jobs",
        type=int,
        help="Override the number of concurrent Optuna trials.",
    )
    parser.add_argument(
        "--maximum-iterations",
        type=int,
        help="Override CatBoost's maximum iterations (useful for smoke tests).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the artifact directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and validate all requested grouped folds without training.",
    )
    return parser.parse_args()


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
    raw_feature_names = [
        name for name in feature_names if name not in engineered_feature_names
    ]
    target_name = data_config["target_name"]
    quality_columns = data_config["quality_columns"]
    required_columns = list(
        dict.fromkeys(
            coordinate_columns + raw_feature_names + [target_name] + quality_columns
        )
    )

    data = (
        ta.load_data(str(resolve_path(data_config["parquet_glob"])))[required_columns]
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
        "depth": trial.suggest_int("depth", 4, 12),
        "rsm": trial.suggest_float("rsm", 0.1, 1.0),
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
    seed: int,
    storage_path: Path,
    study_name: str,
    target_completed_trials: int,
    optuna_jobs: int,
) -> optuna.Study:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
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
        "seed": seed,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode()
    ).hexdigest()
    existing_signature = study.user_attrs.get("study_signature")
    if existing_signature is not None and existing_signature != signature:
        raise ValueError(
            f"Study {study_name!r} was created for different data, folds, or model "
            "settings. Use a new output directory or remove the mismatched study."
        )
    study.set_user_attr("study_signature", signature)
    study.set_metric_names(["multi_quantile_loss"])
    completed = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
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
            objective_factory(data, folds, config, seed),
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
    scores = {
        "all": ta.score_quantile_predictions(
            data[target_name], prediction, alpha, intervals
        )
    }
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
    frame = pd.DataFrame(
        {
            "row_id": row_ids,
            "observed": data[target_name].to_numpy(),
            "bottomdepth": data["bottomdepth"].to_numpy(),
        }
    )
    for index, alpha in enumerate(quantiles(config)):
        frame[f"q_{alpha:g}"] = prediction[:, index]
    return frame


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


def run_outer_fold(
    data: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    outer_fold_index: int,
    target_trials: int,
    optuna_jobs: int,
    fingerprint: str,
) -> None:
    output_dir = resolve_path(config["output_dir"])
    cv_config = config["cross_validation"]
    outer_folds = ta.make_train_test_folds(
        data,
        n_splits=cv_config["outer_splits"],
        shuffle=True,
        random_state=seed,
    )
    outer_train_index, outer_test_index = outer_folds[outer_fold_index]
    outer_train = data.iloc[outer_train_index]
    outer_test = data.iloc[outer_test_index]
    inner_folds = ta.make_train_test_folds(
        outer_train,
        n_splits=cv_config["inner_splits"],
        shuffle=True,
        random_state=seed,
    )

    stem = f"seed_{seed}_outer_{outer_fold_index:02d}"
    study = optimize(
        outer_train,
        inner_folds,
        config,
        seed,
        output_dir / "studies" / f"{stem}.sqlite3",
        f"catboost_quantile_{stem}",
        target_trials,
        optuna_jobs,
    )
    model = make_model(
        config,
        seed,
        selected_params(study, config, seed, final_fit=True),
    )
    feature_names = config["data"]["feature_names"]
    target_name = config["data"]["target_name"]
    model.fit(outer_train[feature_names], outer_train[target_name])
    prediction = model.predict_quantiles(outer_test[feature_names])
    scores = score_subsets(outer_test, prediction, config)

    prediction_path = output_dir / "outer_predictions" / f"{stem}.parquet"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame(
        outer_test, outer_test_index, prediction, config
    ).to_parquet(prediction_path, index=False)
    write_json(
        output_dir / "outer_metrics" / f"{stem}.json",
        {
            "seed": seed,
            "outer_fold": outer_fold_index,
            "data_fingerprint": fingerprint,
            "train_observations": len(outer_train),
            "test_observations": len(outer_test),
            "train_cruises": outer_train.index.get_level_values("expocode").nunique(),
            "test_cruises": outer_test.index.get_level_values("expocode").nunique(),
            "best_trial": study.best_trial.number,
            "best_inner_cv_loss": study.best_value,
            "best_params": study.best_trial.params,
            "refit_iterations": study.best_trial.user_attrs["refit_iterations"],
            "scores": scores,
        },
    )
    logger.success("Completed seed {} outer fold {}", seed, outer_fold_index)


def prediction_columns(config: dict[str, Any]) -> list[str]:
    return [f"q_{alpha:g}" for alpha in quantiles(config)]


def summarize_seed(
    data: pd.DataFrame, config: dict[str, Any], seed: int, fingerprint: str
) -> bool:
    output_dir = resolve_path(config["output_dir"])
    outer_splits = config["cross_validation"]["outer_splits"]
    prediction_paths = [
        output_dir
        / "outer_predictions"
        / f"seed_{seed}_outer_{fold_index:02d}.parquet"
        for fold_index in range(outer_splits)
    ]
    metric_paths = [
        output_dir / "outer_metrics" / f"seed_{seed}_outer_{fold_index:02d}.json"
        for fold_index in range(outer_splits)
    ]
    if not all(path.exists() for path in prediction_paths + metric_paths):
        logger.info("Seed {} is not yet complete; skipping its summary", seed)
        return False

    predictions = pd.concat(
        [pd.read_parquet(path) for path in prediction_paths], ignore_index=True
    ).sort_values("row_id")
    expected_rows = np.arange(len(data))
    if not np.array_equal(predictions["row_id"].to_numpy(), expected_rows):
        raise ValueError(f"Seed {seed} outer predictions do not cover every row exactly once.")
    if not np.allclose(predictions["observed"], data[config["data"]["target_name"]]):
        raise ValueError(f"Seed {seed} predictions do not align with the prepared data.")

    pooled_prediction = predictions[prediction_columns(config)].to_numpy()
    pooled_scores = score_subsets(data, pooled_prediction, config)
    fold_metrics = []
    for path in metric_paths:
        with path.open() as handle:
            fold_metrics.append(json.load(handle))
    write_json(
        output_dir / "summaries" / f"nested_cv_seed_{seed}.json",
        {
            "seed": seed,
            "data_fingerprint": fingerprint,
            "pooled_scores": pooled_scores,
            "fold_scores": [result["scores"] for result in fold_metrics],
        },
    )
    logger.success("Wrote pooled nested-CV summary for seed {}", seed)
    return True


def summarize_ensemble_oof(
    data: pd.DataFrame,
    config: dict[str, Any],
    seeds: Sequence[int],
    fingerprint: str,
) -> bool:
    output_dir = resolve_path(config["output_dir"])
    outer_splits = config["cross_validation"]["outer_splits"]
    seed_predictions = []
    fold_rows = []

    for seed in seeds:
        paths = [
            output_dir
            / "outer_predictions"
            / f"seed_{seed}_outer_{fold_index:02d}.parquet"
            for fold_index in range(outer_splits)
        ]
        metric_paths = [
            output_dir / "outer_metrics" / f"seed_{seed}_outer_{fold_index:02d}.json"
            for fold_index in range(outer_splits)
        ]
        if not all(path.exists() for path in paths + metric_paths):
            logger.info("Nested CV is incomplete; skipping the ensemble OOF summary")
            return False
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        frame = frame.sort_values("row_id")
        if not np.array_equal(frame["row_id"].to_numpy(), np.arange(len(data))):
            raise ValueError(
                f"Seed {seed} outer predictions do not cover every row exactly once."
            )
        if not np.allclose(
            frame["observed"], data[config["data"]["target_name"]]
        ):
            raise ValueError(f"Seed {seed} predictions do not align with prepared data.")
        seed_predictions.append(frame[prediction_columns(config)].to_numpy())
        for metric_path in metric_paths:
            with metric_path.open() as handle:
                result = json.load(handle)
            fold_rows.append(
                {
                    "seed": result["seed"],
                    "outer_fold": result["outer_fold"],
                    **{f"all_{key}": value for key, value in result["scores"]["all"].items()},
                    **{
                        f"deep_{key}": value
                        for key, value in result["scores"].get("deep", {}).items()
                    },
                }
            )

    ensemble_prediction = np.mean(np.asarray(seed_predictions), axis=0)
    ensemble_scores = score_subsets(data, ensemble_prediction, config)
    frame = prediction_frame(data, np.arange(len(data)), ensemble_prediction, config)
    oof_path = output_dir / "summaries" / "ensemble_oof_predictions.parquet"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(oof_path, index=False)
    pd.DataFrame(fold_rows).to_csv(
        output_dir / "summaries" / "nested_cv_fold_metrics.csv", index=False
    )
    write_json(
        output_dir / "summaries" / "ensemble_oof_metrics.json",
        {
            "seeds": list(seeds),
            "data_fingerprint": fingerprint,
            "aggregation": "mean",
            "scores": ensemble_scores,
        },
    )
    logger.success("Wrote leakage-free ensemble out-of-fold metrics")
    return True


def train_final_member(
    data: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    target_trials: int,
    optuna_jobs: int,
    fingerprint: str,
) -> Path:
    output_dir = resolve_path(config["output_dir"])
    cv_config = config["cross_validation"]
    folds = ta.make_train_test_folds(
        data,
        n_splits=cv_config["final_tuning_splits"],
        shuffle=True,
        random_state=seed,
    )
    study = optimize(
        data,
        folds,
        config,
        seed,
        output_dir / "studies" / f"seed_{seed}_final.sqlite3",
        f"catboost_quantile_seed_{seed}_final",
        target_trials,
        optuna_jobs,
    )
    params = selected_params(study, config, seed, final_fit=True)
    model = make_model(config, seed, params)
    feature_names = config["data"]["feature_names"]
    target_name = config["data"]["target_name"]
    model.fit(data[feature_names], data[target_name])
    model.training_metadata_ = {
        "seed": seed,
        "data_fingerprint": fingerprint,
        "observation_count": len(data),
        "cruise_count": data.index.get_level_values("expocode").nunique(),
        "best_trial": study.best_trial.number,
        "best_cv_loss": study.best_value,
        "best_params": study.best_trial.params,
        "refit_iterations": study.best_trial.user_attrs["refit_iterations"],
    }

    model_path = output_dir / "models" / f"quantile_seed_{seed}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    write_json(
        output_dir / "models" / f"quantile_seed_{seed}.json",
        model.training_metadata_,
    )
    logger.success("Saved final member for seed {} to {}", seed, model_path)
    return model_path


def assemble_ensemble(
    config: dict[str, Any], seeds: Sequence[int], fingerprint: str
) -> Path:
    output_dir = resolve_path(config["output_dir"])
    model_paths = [
        output_dir / "models" / f"quantile_seed_{seed}.joblib" for seed in seeds
    ]
    missing = [path for path in model_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot assemble ensemble; missing members: "
            + ", ".join(str(path) for path in missing)
        )
    members = [ta.CatBoostResidualRegressor.load(path) for path in model_paths]
    ensemble = ta.QuantileRegressionEnsemble(
        estimators=members,
        seeds=seeds,
        quantiles=quantiles(config),
        aggregation=config["ensemble"]["aggregation"],
    )
    ensemble_path = output_dir / "models" / "quantile_ensemble.joblib"
    ensemble.save(ensemble_path)
    write_json(
        output_dir / "models" / "manifest.json",
        {
            "seeds": list(seeds),
            "quantiles": quantiles(config),
            "aggregation": config["ensemble"]["aggregation"],
            "data_fingerprint": fingerprint,
            "feature_names": config["data"]["feature_names"],
            "target_name": config["data"]["target_name"],
            "members": [path.name for path in model_paths],
            "ensemble": ensemble_path.name,
        },
    )
    logger.success("Saved seven-member ensemble to {}", ensemble_path)
    return ensemble_path


def selected_seeds(config: dict[str, Any], requested_seed: int | None) -> list[int]:
    configured = [int(seed) for seed in config["seeds"]]
    if len(configured) != 7 or len(set(configured)) != 7:
        raise ValueError("Configuration must contain exactly seven unique seeds.")
    if requested_seed is None:
        return configured
    if requested_seed not in configured:
        raise ValueError(f"Seed {requested_seed} is not in configured seeds {configured}.")
    return [requested_seed]


def validate_splits(
    data: pd.DataFrame, config: dict[str, Any], seeds: Sequence[int]
) -> None:
    cv_config = config["cross_validation"]
    groups = data.index.get_level_values("expocode").to_numpy()
    for seed in seeds:
        outer_folds = ta.make_train_test_folds(
            data,
            n_splits=cv_config["outer_splits"],
            shuffle=True,
            random_state=seed,
        )
        covered = np.zeros(len(data), dtype=int)
        for outer_fold_index, (train_index, test_index) in enumerate(outer_folds):
            covered[test_index] += 1
            overlap = set(groups[train_index]).intersection(groups[test_index])
            if overlap:
                raise ValueError(
                    f"Seed {seed}, outer fold {outer_fold_index} leaks cruises: {overlap}"
                )
            inner_folds = ta.make_train_test_folds(
                data.iloc[train_index],
                n_splits=cv_config["inner_splits"],
                shuffle=True,
                random_state=seed,
            )
            inner_groups = groups[train_index]
            for inner_train, inner_validation in inner_folds:
                inner_overlap = set(inner_groups[inner_train]).intersection(
                    inner_groups[inner_validation]
                )
                if inner_overlap:
                    raise ValueError(
                        f"Seed {seed}, outer fold {outer_fold_index} leaks inner cruises."
                    )
        if not np.all(covered == 1):
            raise ValueError(f"Seed {seed} outer folds do not test each row exactly once.")
        logger.info(
            "Validated seed {}: {} outer folds, each with {} inner folds",
            seed,
            len(outer_folds),
            cv_config["inner_splits"],
        )


def main() -> None:
    args = parse_args()
    if args.outer_fold is not None and args.seed is None:
        raise ValueError("--outer-fold requires --seed.")
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    config = load_config(args.config)
    if args.maximum_iterations is not None:
        config["model"]["maximum_iterations"] = args.maximum_iterations
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    seeds = selected_seeds(config, args.seed)
    all_seeds = selected_seeds(config, None)
    target_trials = args.n_trials or int(config["optimization"]["n_trials"])
    optuna_jobs = args.optuna_jobs or int(config["optimization"]["n_jobs"])
    if target_trials < 1:
        raise ValueError("The target number of completed trials must be positive.")
    if optuna_jobs == 0:
        raise ValueError("Optuna jobs must be nonzero.")
    if config["model"]["maximum_iterations"] < 1:
        raise ValueError("CatBoost maximum iterations must be positive.")
    data = prepare_data(config)
    fingerprint = data_fingerprint(data)
    logger.info("Data fingerprint: {}", fingerprint)
    if args.dry_run:
        validate_splits(data, config, seeds)
        logger.success("Dry run completed; no studies or models were written")
        return

    if args.stage in ("evaluate", "all"):
        outer_splits = config["cross_validation"]["outer_splits"]
        fold_indices = (
            [args.outer_fold] if args.outer_fold is not None else range(outer_splits)
        )
        for seed in seeds:
            for fold_index in fold_indices:
                if not 0 <= fold_index < outer_splits:
                    raise IndexError(
                        f"Outer fold {fold_index} is outside [0, {outer_splits})."
                    )
                run_outer_fold(
                    data,
                    config,
                    seed,
                    fold_index,
                    target_trials,
                    optuna_jobs,
                    fingerprint,
                )
            summarize_seed(data, config, seed, fingerprint)
        summarize_ensemble_oof(data, config, all_seeds, fingerprint)

    if args.stage in ("train", "all"):
        for seed in seeds:
            train_final_member(
                data,
                config,
                seed,
                target_trials,
                optuna_jobs,
                fingerprint,
            )

    should_assemble = args.stage == "assemble" or (
        args.stage == "all" and args.seed is None
    )
    if should_assemble:
        assemble_ensemble(config, all_seeds, fingerprint)


if __name__ == "__main__":
    main()

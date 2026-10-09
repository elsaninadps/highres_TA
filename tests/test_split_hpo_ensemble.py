import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from highres_ta import QuantileRegressionEnsemble, decompose_variance

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_quantile_ensemble.py"
QUANTILES = [0.1, 0.5, 0.9]


def load_script():
    spec = importlib.util.spec_from_file_location("train_quantile_ensemble", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script()


def make_config(n_parts=7, n_hpo_replicates=7):
    return {
        "cross_validation": {"n_parts": n_parts, "split_random_state": 42},
        "ensemble": {"aggregation": "mean", "n_hpo_replicates": n_hpo_replicates},
    }


def make_data(n_cruises=70, rows_per_cruise=8, seed=0):
    rng = np.random.default_rng(seed)
    expocodes = np.repeat([f"CRUISE{i:03d}" for i in range(n_cruises)], rows_per_cruise)
    cruise_salinity = rng.uniform(30.0, 37.0, n_cruises)
    salinity = np.repeat(cruise_salinity, rows_per_cruise) + rng.normal(
        0, 0.1, n_cruises * rows_per_cruise
    )
    index = pd.MultiIndex.from_arrays(
        [expocodes, np.arange(len(expocodes))], names=["expocode", "row"]
    )
    return pd.DataFrame({"salinity": salinity, "talk": 2300 + 50 * salinity}, index=index)


class FixedQuantileEstimator:
    def __init__(self, prediction, quantiles):
        self.prediction = np.asarray(prediction, dtype=float)
        self.quantiles_ = np.asarray(quantiles, dtype=float)

    def predict_quantiles(self, X):
        return self.prediction[: len(X)]


class MemberTests(unittest.TestCase):
    def test_full_design_is_six_splits_by_seven_replicates(self):
        selected = script.members(make_config())

        self.assertEqual(len(selected), 42)
        self.assertEqual(
            [(m.split, m.replicate) for m in selected],
            [(k, r) for k in range(1, 7) for r in range(7)],
        )
        self.assertEqual(len({m.seed for m in selected}), 42)
        self.assertEqual(len({m.name for m in selected}), 42)

    def test_member_seed_and_name(self):
        member = script.Member(split=3, replicate=5)

        self.assertEqual(member.seed, 305)
        self.assertEqual(member.name, "split_3_hpo_5")

    def test_filters(self):
        config = make_config()

        by_split = script.members(config, split=2)
        by_replicate = script.members(config, replicate=4)
        single = script.members(config, split=6, replicate=0)

        self.assertEqual([m.replicate for m in by_split], list(range(7)))
        self.assertTrue(all(m.split == 2 for m in by_split))
        self.assertEqual([m.split for m in by_replicate], list(range(1, 7)))
        self.assertEqual(single, [script.Member(split=6, replicate=0)])

    def test_out_of_range_filters_raise(self):
        config = make_config()
        for kwargs in ({"split": 0}, {"split": 7}, {"replicate": -1}, {"replicate": 7}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                script.members(config, **kwargs)


class PartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = make_config()
        cls.data = make_data()
        cls.blocks = script.outer_blocks(cls.data, cls.config)
        cls.groups = cls.data.index.get_level_values("expocode").to_numpy()

    def test_outer_blocks_partition_rows_by_cruise(self):
        self.assertEqual(len(self.blocks), 7)
        covered = np.concatenate(self.blocks)
        self.assertEqual(len(covered), len(self.data))
        self.assertEqual(len(np.unique(covered)), len(self.data))
        cruise_sets = [set(self.groups[block]) for block in self.blocks]
        for i in range(len(cruise_sets)):
            for j in range(i + 1, len(cruise_sets)):
                self.assertFalse(cruise_sets[i] & cruise_sets[j])

    def test_outer_blocks_are_deterministic(self):
        again = script.outer_blocks(self.data, self.config)
        for left, right in zip(self.blocks, again):
            np.testing.assert_array_equal(np.sort(left), np.sort(right))

    def test_outer_blocks_require_three_parts(self):
        with self.assertRaises(ValueError):
            script.outer_blocks(self.data, make_config(n_parts=2))

    def test_test_block_is_shared_and_validation_rotates(self):
        for k in range(1, 7):
            with self.subTest(split=k):
                split = script.split_indices(self.blocks, k)
                np.testing.assert_array_equal(split["test"], np.sort(self.blocks[0]))
                np.testing.assert_array_equal(split["validation"], np.sort(self.blocks[k]))
                self.assertEqual(len(split["train_blocks"]), 5)
                np.testing.assert_array_equal(
                    split["train"], np.sort(np.concatenate(split["train_blocks"]))
                )

    def test_split_is_disjoint_complete_and_leak_free(self):
        for k in range(1, 7):
            with self.subTest(split=k):
                split = script.split_indices(self.blocks, k)
                covered = np.concatenate([split[name] for name in script.SPLIT_NAMES])
                self.assertEqual(len(covered), len(self.data))
                self.assertEqual(len(np.unique(covered)), len(self.data))
                for name in script.SPLIT_NAMES:
                    self.assertTrue(np.all(np.diff(split[name]) > 0))
                script.validate_split(self.data, split, k)

    def test_invalid_split_raises(self):
        for k in (0, 7):
            with self.subTest(split=k), self.assertRaises(ValueError):
                script.split_indices(self.blocks, k)

    def test_tuning_folds_are_leave_one_block_out_within_train(self):
        for k in range(1, 7):
            with self.subTest(split=k):
                split = script.split_indices(self.blocks, k)
                train_rows = split["train"]
                folds = script.tuning_folds(split)

                self.assertEqual(len(folds), 5)
                for (fit, holdout), block in zip(folds, split["train_blocks"]):
                    np.testing.assert_array_equal(train_rows[holdout], np.sort(block))
                    self.assertEqual(len(fit) + len(holdout), len(train_rows))
                    self.assertFalse(set(fit) & set(holdout))
                    self.assertFalse(
                        set(self.groups[train_rows[fit]]) & set(self.groups[train_rows[holdout]])
                    )
                holdouts = np.concatenate([holdout for _, holdout in folds])
                np.testing.assert_array_equal(np.sort(holdouts), np.arange(len(train_rows)))

    def test_tuning_folds_do_not_depend_on_replicate(self):
        split = script.split_indices(self.blocks, 3)
        first = script.tuning_folds(split)
        second = script.tuning_folds(script.split_indices(self.blocks, 3))
        for (fit_a, holdout_a), (fit_b, holdout_b) in zip(first, second):
            np.testing.assert_array_equal(fit_a, fit_b)
            np.testing.assert_array_equal(holdout_a, holdout_b)


class DecompositionTests(unittest.TestCase):
    def test_terms_sum_to_total_member_variance(self):
        rng = np.random.default_rng(1)
        predictions = rng.normal(size=(6, 7, 11, 3))

        result = decompose_variance(predictions)
        flat = predictions.reshape(42, 11, 3)

        np.testing.assert_allclose(result["mean"], flat.mean(axis=0))
        np.testing.assert_allclose(result["split_var"] + result["hpo_var"], flat.var(axis=0))
        self.assertTrue(np.all(result["split_var"] >= 0))
        self.assertTrue(np.all(result["hpo_var"] >= 0))

    def test_constant_within_split_has_no_hpo_variance(self):
        split_offsets = np.arange(6, dtype=float)[:, None, None, None]
        predictions = np.broadcast_to(split_offsets, (6, 7, 4, 3)).copy()

        result = decompose_variance(predictions)

        np.testing.assert_allclose(result["hpo_var"], 0.0)
        np.testing.assert_allclose(result["split_var"], np.var(np.arange(6.0)))

    def test_identical_splits_have_no_split_variance(self):
        rng = np.random.default_rng(2)
        one_split = rng.normal(size=(1, 7, 4, 3))
        predictions = np.repeat(one_split, 6, axis=0)

        result = decompose_variance(predictions)

        np.testing.assert_allclose(result["split_var"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["hpo_var"], one_split[0].var(axis=0))

    def test_rejects_wrong_rank(self):
        with self.assertRaises(ValueError):
            decompose_variance(np.zeros((42, 4, 3)))


class EnsembleDecompositionTests(unittest.TestCase):
    n_splits = 3
    n_replicates = 4
    n_rows = 5

    def make_ensemble(self, predictions, order=None):
        design = [(k, r) for k in range(1, self.n_splits + 1) for r in range(self.n_replicates)]
        if order is not None:
            design = [design[i] for i in order]
        estimators = [
            FixedQuantileEstimator(predictions[k - 1, r], QUANTILES) for k, r in design
        ]
        return QuantileRegressionEnsemble(
            estimators=estimators,
            seeds=[100 * k + r for k, r in design],
            quantiles=QUANTILES,
            member_splits=[k for k, _ in design],
            member_replicates=[r for _, r in design],
        )

    def make_predictions(self, seed=3):
        rng = np.random.default_rng(seed)
        base = np.sort(rng.normal(size=(self.n_splits, self.n_replicates, self.n_rows, 3)), axis=-1)
        return base

    def test_predict_decomposition_matches_function(self):
        predictions = self.make_predictions()
        ensemble = self.make_ensemble(predictions)
        X = np.zeros((self.n_rows, 1))

        result = ensemble.predict_decomposition(X)
        expected = decompose_variance(predictions)

        for key in ("mean", "split_var", "hpo_var"):
            np.testing.assert_allclose(result[key], expected[key], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            result["mean"], ensemble.predict_quantiles(X), rtol=1e-5, atol=1e-6
        )

    def test_predict_decomposition_is_independent_of_member_order(self):
        predictions = self.make_predictions()
        order = np.random.default_rng(4).permutation(self.n_splits * self.n_replicates)
        ensemble = self.make_ensemble(predictions, order=list(order))

        result = ensemble.predict_decomposition(np.zeros((self.n_rows, 1)))
        expected = decompose_variance(predictions)

        for key in ("mean", "split_var", "hpo_var"):
            np.testing.assert_allclose(result[key], expected[key], rtol=1e-5, atol=1e-6)

    def test_predict_decomposition_requires_member_design(self):
        prediction = np.zeros((self.n_rows, 3))
        ensemble = QuantileRegressionEnsemble(
            estimators=[FixedQuantileEstimator(prediction, QUANTILES) for _ in range(2)],
            seeds=[0, 1],
            quantiles=QUANTILES,
        )
        with self.assertRaises(ValueError):
            ensemble.predict_decomposition(np.zeros((self.n_rows, 1)))

    def test_predict_decomposition_rejects_unbalanced_design(self):
        prediction = np.zeros((self.n_rows, 3))
        ensemble = QuantileRegressionEnsemble(
            estimators=[FixedQuantileEstimator(prediction, QUANTILES) for _ in range(3)],
            seeds=[100, 101, 200],
            quantiles=QUANTILES,
            member_splits=[1, 1, 2],
            member_replicates=[0, 1, 0],
        )
        with self.assertRaises(ValueError):
            ensemble.predict_decomposition(np.zeros((self.n_rows, 1)))

    def test_rejects_mismatched_design_lengths(self):
        prediction = np.zeros((self.n_rows, 3))
        estimators = [FixedQuantileEstimator(prediction, QUANTILES) for _ in range(2)]
        bad_designs = (
            {"member_splits": [1], "member_replicates": [0, 1]},
            {"member_splits": [1, 1], "member_replicates": [0]},
            {"member_splits": [1, 1]},
            {"member_replicates": [0, 1]},
        )
        for kwargs in bad_designs:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                QuantileRegressionEnsemble(
                    estimators=estimators, seeds=[100, 101], quantiles=QUANTILES, **kwargs
                )


if __name__ == "__main__":
    unittest.main()

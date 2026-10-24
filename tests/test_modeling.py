"""Tests for modeling.py (imputation, scaling, K-Means, evaluation, PCA, interpretation).

Run from the project folder:
    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from data_loader import DATA_DIR, load_data  # noqa: E402
from modeling import (  # noqa: E402
    DEFAULT_K, INDICATOR_FEATURES, MAX_K, MIN_DEFINED_FOR_MEDIAN, MODEL_FEATURES, ModelingError,
    best_silhouette_k, build_model_matrix, centroid_distances, cluster_profiles, describe_silhouette,
    elbow_heuristic, evaluate_k_range, fit_kmeans, interpret_clusters, most_similar_players,
    run_pca, run_pipeline, silhouette, validate_k,
)
from preprocessing import activity_indicators, prepare_features  # noqa: E402

NAN = np.nan


def player(name, mat, bat=None, bowl=None, field=None):
    """One row in the merged-table format (same layout as test_preprocessing)."""
    bat = bat or (NAN,) * 6
    bowl = bowl or (NAN, "-", NAN, NAN)
    field = field or (NAN,) * 5
    return {
        "Player": name, "Mat_Bat": mat, "Mat_Bowl": mat, "Mat_Field": mat,
        "Inns_Bat": bat[0], "NO_Bat": bat[1], "Runs_Bat": bat[2], "BF_Bat": bat[3],
        "4s_Bat": bat[4], "6s_Bat": bat[5],
        "Inns_Bowl": bowl[0], "Overs_Bowl": bowl[1], "Runs_Bowl": bowl[2], "Wkts_Bowl": bowl[3],
        "Inns_Field": field[0], "Dis_Field": field[1], "Ct_Field": field[2],
        "St_Field": field[3], "Ct Wk_Field": field[4],
    }


def synthetic_league(per_group=15, seed=0):
    """Four clearly different player types with small random variation."""
    rng = np.random.default_rng(seed)
    rows, groups = [], {}

    def jitter(value, spread=0.1):
        return max(0, int(round(value * (1 + rng.uniform(-spread, spread)))))

    for i in range(per_group):
        rows.append(player(f"Batter {i} (AAA)", 20,
                           bat=(20, 2, jitter(520), jitter(380), jitter(50), jitter(20)),
                           field=(20, jitter(8), jitter(8), 0, 0)))
        groups[f"Batter {i} (AAA)"] = "batter"
        rows.append(player(f"Bowler {i} (BBB)", 20,
                           bat=(6, 3, jitter(30), jitter(35), 2, 1),
                           bowl=(20, f"{jitter(70)}.{rng.integers(0, 6)}", jitter(520), jitter(28)),
                           field=(20, jitter(5), jitter(5), 0, 0)))
        groups[f"Bowler {i} (BBB)"] = "bowler"
        rows.append(player(f"Keeper {i} (CCC)", 20,
                           bat=(18, 3, jitter(330), jitter(300), jitter(30), jitter(8)),
                           field=(20, 24, 16, 8, 16)))
        groups[f"Keeper {i} (CCC)"] = "keeper"
        rows.append(player(f"Allrounder {i} (DDD)", 20,
                           bat=(18, 3, jitter(380), jitter(280), jitter(35), jitter(16)),
                           bowl=(18, f"{jitter(50)}.{rng.integers(0, 6)}", jitter(400), jitter(18)),
                           field=(20, jitter(6), jitter(6), 0, 0)))
        groups[f"Allrounder {i} (DDD)"] = "allrounder"
    return pd.DataFrame(rows), pd.Series(groups)


def model_input(data, min_matches=5):
    features = prepare_features(data, min_matches=min_matches)
    activity = activity_indicators(data, features.index)
    return features, activity, build_model_matrix(features, activity)


class TestModelMatrix(unittest.TestCase):
    def setUp(self):
        self.data, self.groups = synthetic_league()
        self.features, self.activity, self.matrix = model_input(self.data)

    def test_finite_numeric_and_standardized(self):
        X = self.matrix["X"]
        self.assertTrue(np.isfinite(X.to_numpy()).all())
        self.assertTrue(all(pd.api.types.is_float_dtype(t) for t in X.dtypes))
        self.assertTrue(set(X.columns) <= set(MODEL_FEATURES + INDICATOR_FEATURES))
        np.testing.assert_allclose(X.mean().to_numpy(), 0, atol=1e-9)
        np.testing.assert_allclose(X.std(ddof=0).to_numpy(), 1, atol=1e-9)

    def test_players_are_index_not_features(self):
        X = self.matrix["X"]
        self.assertEqual(list(X.index), list(self.features.index))
        self.assertNotIn("Player", X.columns)
        self.assertIn("Batter 0 (AAA)", X.index)

    def test_inputs_not_mutated(self):
        features, activity = self.features.copy(), self.activity.copy()
        build_model_matrix(self.features, self.activity)
        pd.testing.assert_frame_equal(self.features, features)
        pd.testing.assert_frame_equal(self.activity, activity)

    def test_non_bowler_gets_median_not_elite_values(self):
        processed, medians = self.matrix["X_processed"], self.matrix["medians"]
        batter = "Batter 0 (AAA)"
        self.assertTrue(np.isnan(self.features.loc[batter, "bowl_economy"]))  # undefined before
        self.assertEqual(processed.loc[batter, "bowl_economy"], medians["bowl_economy"])
        bowled = self.features["bowl_economy"].dropna()
        self.assertGreater(processed.loc[batter, "bowl_economy"], bowled.min())  # not the best economy
        self.assertEqual(processed.loc[batter, "has_bowled"], 0)
        self.assertEqual(processed.loc[batter, "bowl_wkts_per_match"], 0)

    def test_indicator_comes_from_raw_data(self):
        # Every bowler and all-rounder bowled in the raw data, nobody else did.
        expected = self.groups.isin(["bowler", "allrounder"]).astype(int)
        pd.testing.assert_series_equal(
            self.activity["has_bowled"].sort_index(), expected.sort_index().rename("has_bowled"),
            check_names=False, check_index_type=False)
        self.assertEqual(int(self.matrix["imputed"]["bowl_economy"]), int((expected == 0).sum()))

    def test_constant_and_all_missing_features_removed(self):
        no_bowlers = self.data[~self.data["Player"].str.startswith(("Bowler", "Allrounder"))]
        features, activity, matrix = model_input(no_bowlers)
        X = matrix["X"]
        self.assertFalse(any(c.startswith("bowl_") for c in X.columns))
        self.assertNotIn("has_bowled", X.columns)  # constant 0
        self.assertTrue(np.isfinite(X.to_numpy()).all())

    def test_too_few_players(self):
        tiny = self.data.head(2)
        features = prepare_features(tiny)
        with self.assertRaises(ModelingError):
            build_model_matrix(features, activity_indicators(tiny, features.index))
        with self.assertRaises(ModelingError):
            build_model_matrix(features.iloc[:0], activity_indicators(tiny, features.index[:0]))


class TestKMeans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.groups = synthetic_league()
        cls.features, cls.activity, cls.matrix = model_input(cls.data)
        cls.X = cls.matrix["X"]

    def test_valid_labels_mapped_to_players(self):
        fit = fit_kmeans(self.X, 4)
        labels = fit["labels"]
        self.assertEqual(list(labels.index), list(self.X.index))
        self.assertEqual(sorted(labels.unique()), [1, 2, 3, 4])
        sizes = labels.value_counts().sort_index()
        self.assertTrue((sizes.diff().dropna() <= 0).all())  # renumbered largest first
        self.assertEqual(list(fit["centers"].index), [1, 2, 3, 4])

    def test_recovers_synthetic_groups(self):
        labels = fit_kmeans(self.X, 4)["labels"]
        table = pd.crosstab(self.groups.reindex(labels.index), labels)
        purity = table.max(axis=1).sum() / len(labels)
        self.assertGreaterEqual(purity, 0.95)

    def test_assignment_is_nearest_centre(self):
        fit = fit_kmeans(self.X, 4)
        for p in self.X.index[::7]:
            self.assertEqual(centroid_distances(self.X, fit["centers"], p).index[0], fit["labels"][p])

    def test_other_k_values(self):
        for k in range(2, 7):
            fit = fit_kmeans(self.X, k)
            self.assertEqual(fit["labels"].nunique(), k)

    def test_invalid_k(self):
        for bad in (1, MAX_K + 1, 2.5, True, "4"):
            with self.assertRaises(ModelingError):
                validate_k(bad, self.X)
        small = self.X.iloc[:4]
        with self.assertRaisesRegex(ModelingError, "too large"):
            validate_k(4, small)

    def test_reproducible(self):
        a, b = fit_kmeans(self.X, 4), fit_kmeans(self.X, 4)
        pd.testing.assert_series_equal(a["labels"], b["labels"])
        self.assertEqual(a["inertia"], b["inertia"])


class TestEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data, _ = synthetic_league()
        cls.X = model_input(data)[2]["X"]
        cls.evaluation = evaluate_k_range(cls.X)

    def test_inertia_for_each_k(self):
        inertia = self.evaluation["inertia"]
        self.assertEqual(list(inertia.index), list(range(2, MAX_K + 1)))
        self.assertTrue(inertia.notna().all())
        self.assertTrue((inertia.diff().dropna() <= 1e-6).all())  # never increases with K

    def test_silhouette_range(self):
        scores = self.evaluation["silhouette"]
        self.assertTrue(((scores >= -1) & (scores <= 1)).all())
        self.assertEqual(best_silhouette_k(self.evaluation), int(scores.idxmax()))

    def test_silhouette_only_for_valid_partitions(self):
        self.assertIsNone(silhouette(self.X, np.ones(len(self.X))))  # one cluster
        self.assertIsNone(silhouette(self.X.iloc[:3], [1, 2, 3]))  # as many clusters as players
        self.assertIsNotNone(silhouette(self.X, fit_kmeans(self.X, 3)["labels"]))

    def test_invalid_k_reported_not_fitted(self):
        evaluation = evaluate_k_range(self.X.iloc[:5])
        self.assertTrue(evaluation.loc[MAX_K, "note"])
        self.assertTrue(np.isnan(evaluation.loc[MAX_K, "inertia"]))

    def test_elbow_heuristic(self):
        manual = pd.DataFrame({"inertia": [100, 40, 30, 25, 22], "silhouette": [0.3, 0.5, 0.4, 0.3, 0.2]},
                              index=pd.Index([2, 3, 4, 5, 6], name="k"))
        self.assertEqual(elbow_heuristic(manual), 3)
        self.assertEqual(best_silhouette_k(manual), 3)
        self.assertIsNone(elbow_heuristic(manual.iloc[:2]))

    def test_describe_silhouette(self):
        self.assertEqual(describe_silhouette(0.8), "strong structure")
        self.assertTrue(describe_silhouette(0.3).startswith("weak"))
        self.assertEqual(describe_silhouette(None), "not defined")


class TestPCA(unittest.TestCase):
    def test_dimensions_and_variance(self):
        data, _ = synthetic_league()
        X = model_input(data)[2]["X"]
        pca = run_pca(X)
        self.assertEqual(pca["coords"].shape, (len(X), 2))
        self.assertEqual(list(pca["coords"].index), list(X.index))
        self.assertEqual(pca["loadings"].shape, (X.shape[1], 2))
        explained = pca["explained"]
        self.assertTrue(0 <= explained["PC2"] <= explained["PC1"] <= 1)
        self.assertAlmostEqual(explained["total"], explained["PC1"] + explained["PC2"])
        self.assertLessEqual(explained["total"], 1 + 1e-9)

    def test_needs_two_features(self):
        with self.assertRaises(ModelingError):
            run_pca(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))


class TestInterpretation(unittest.TestCase):
    def test_profiles_use_real_values(self):
        data, _ = synthetic_league()
        features, activity, matrix = model_input(data)
        labels = fit_kmeans(matrix["X"], 4)["labels"]
        profile, overall, defined = cluster_profiles(features, activity, labels)
        for c in labels.unique():
            members = features[labels == c]
            self.assertEqual(profile.loc[c, "players"], len(members))
            runs = members["bat_runs_per_inns"].dropna()
            if len(runs) >= MIN_DEFINED_FOR_MEDIAN:
                self.assertAlmostEqual(profile.loc[c, "bat_runs_per_inns"], runs.median())
            else:
                self.assertTrue(np.isnan(profile.loc[c, "bat_runs_per_inns"]))
            self.assertAlmostEqual(profile.loc[c, "has_bowled"], activity.loc[members.index, "has_bowled"].mean())
        self.assertAlmostEqual(overall["bat_runs_per_inns"], features["bat_runs_per_inns"].median())

    def test_medians_from_too_few_players_are_hidden(self):
        features = pd.DataFrame({"bowl_economy": [7.0] + [NAN] * 11}, index=[f"P{i}" for i in range(12)])
        activity = pd.DataFrame({"has_bowled": [1] + [0] * 11}, index=features.index)
        profile, _, defined = cluster_profiles(features, activity, pd.Series(1, index=features.index))
        self.assertEqual(defined.loc[1, "bowl_economy"], 1)
        self.assertTrue(np.isnan(profile.loc[1, "bowl_economy"]))

    def test_label_rules(self):
        cols = ["bat_runs_per_inns", "bat_strike_rate", "bat_sixes_per_inns", "bowl_wkts_per_match",
                "has_bowled", "field_catches_per_match", "field_keeper_share"]
        relative = pd.DataFrame([
            [0.2, 0.2, 0.2, -1.0, -1.5, 0.5, 3.0],    # keeper
            [0.8, 0.8, 0.8, 0.7, 0.7, 0.0, -0.3],     # all-round
            [0.9, 0.9, 0.9, -0.2, 0.6, 0.0, -0.3],    # batting, bowls
            [0.4, 0.4, 0.4, -1.0, -1.5, 0.0, -0.3],   # batting by gap, rarely bowls
            [-0.8, -0.8, -0.8, 0.8, 0.6, 0.0, -0.3],  # bowling
            [-0.2, -0.2, -0.2, -0.2, 0.2, 0.9, -0.3], # fielding
            [-0.1, -0.1, -0.1, -0.3, -1.0, 0.0, -0.3],  # limited bowling data
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -0.3],     # neutral
        ], columns=cols, index=pd.Index(range(1, 9), name="cluster"))
        profile = pd.DataFrame({"has_bowled": [0.05, 1.0, 0.9, 0.1, 1.0, 0.6, 0.2, 0.7]}, index=relative.index)
        result = interpret_clusters(relative, profile)["label"].tolist()
        self.assertEqual(result, [
            "Wicket-Keeping Profile", "All-Round Statistical Profile", "Batting-Oriented, Also Bowls",
            "Batting-Oriented, Rarely Bowls", "Bowling-Oriented", "Fielding-Oriented",
            "Limited Bowling Data", "Cluster 8",
        ])

    def test_repeated_labels_made_unique(self):
        relative = pd.DataFrame({"bat_runs_per_inns": [1.0, 1.0], "bowl_wkts_per_match": [-1.0, -1.0],
                                 "has_bowled": [0.5, 0.5]}, index=pd.Index([1, 2], name="cluster"))
        profile = pd.DataFrame({"has_bowled": [0.9, 0.9]}, index=relative.index)
        labels = interpret_clusters(relative, profile)["label"]
        self.assertTrue(labels.is_unique)
        self.assertTrue(all(name.startswith("Batting-Oriented") for name in labels))


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, _ = synthetic_league()
        cls.result = run_pipeline(cls.data, min_matches=5, k=DEFAULT_K)

    def test_results_preserve_alignment(self):
        r = self.result
        results = r["results"].set_index("Player")
        self.assertTrue(results.index.is_unique)
        self.assertEqual(set(results.index), set(r["labels"].index))
        for p in results.index[::5]:
            self.assertEqual(results.loc[p, "cluster_id"], r["labels"][p])
            self.assertEqual(results.loc[p, "cluster_label"], r["interpretation"].loc[r["labels"][p], "label"])
            self.assertAlmostEqual(results.loc[p, "PC1"], r["pca"]["coords"].loc[p, "PC1"])
            self.assertAlmostEqual(results.loc[p, "PC2"], r["pca"]["coords"].loc[p, "PC2"])
            for f in r["features"].columns:
                a, b = results.loc[p, f], r["features"].loc[p, f]
                self.assertTrue((np.isnan(a) and np.isnan(b)) or a == b, f)

    def test_similar_players(self):
        X = self.result["matrix"]["X"]
        similar = most_similar_players(X, "Batter 0 (AAA)", n=5)
        self.assertNotIn("Batter 0 (AAA)", similar.index)
        self.assertTrue(similar.is_monotonic_increasing)
        self.assertTrue(all(p.startswith("Batter") for p in similar.index))

    def test_empty_filter_is_an_error(self):
        with self.assertRaisesRegex(ModelingError, "No players"):
            run_pipeline(self.data, min_matches=1000, k=4)


@unittest.skipUnless((DATA_DIR / "Batting" / "t20.csv").is_file(), "real T20 CSVs not present")
class TestRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.merged = load_data()["merged"]
        cls.result = run_pipeline(cls.merged, min_matches=5, k=DEFAULT_K)

    def test_model_input_is_clean(self):
        X = self.result["matrix"]["X"]
        self.assertTrue(np.isfinite(X.to_numpy()).all())
        self.assertEqual(len(X), len(prepare_features(self.merged, min_matches=5)))
        self.assertTrue(set(X.index) <= set(self.merged["Player"]))
        self.assertGreaterEqual(X.shape[1], 2)

    def test_clusters_and_sizes(self):
        labels = self.result["labels"]
        self.assertEqual(labels.nunique(), DEFAULT_K)
        self.assertEqual(int(self.result["profile"]["players"].sum()), len(labels))
        self.assertTrue(self.result["interpretation"]["label"].is_unique)
        # Check K = 6 too: sklearn's raw label order is not size-ordered there.
        for check in (labels, fit_kmeans(self.result["matrix"]["X"], 6)["labels"]):
            sizes = check.value_counts().sort_index()
            self.assertTrue((sizes.diff().dropna() <= 0).all(), "clusters must be numbered largest first")

    def test_every_player_nearest_its_centre(self):
        X, centers, labels = self.result["matrix"]["X"], self.result["centers"], self.result["labels"]
        dist = np.linalg.norm(X.to_numpy()[:, None, :] - centers.to_numpy()[None, :, :], axis=2)
        nearest = centers.index.to_numpy()[dist.argmin(axis=1)]
        self.assertTrue((nearest == labels.to_numpy()).all())

    def test_reproducible_on_real_data(self):
        again = run_pipeline(self.merged, min_matches=5, k=DEFAULT_K)
        pd.testing.assert_series_equal(again["labels"], self.result["labels"])
        self.assertEqual(again["inertia"], self.result["inertia"])

    def test_evaluation_on_real_data(self):
        evaluation = evaluate_k_range(self.result["matrix"]["X"])
        self.assertTrue(evaluation["inertia"].notna().all())
        self.assertTrue(evaluation["silhouette"].between(-1, 1).all())
        self.assertAlmostEqual(evaluation.loc[DEFAULT_K, "inertia"], self.result["inertia"])


if __name__ == "__main__":
    unittest.main()

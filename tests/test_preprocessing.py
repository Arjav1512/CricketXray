"""Tests for preprocessing.py (and the overs conversion in data_loader.py).

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

from data_loader import DATA_DIR, DataValidationError, load_data, overs_to_balls  # noqa: E402
from preprocessing import (  # noqa: E402
    FEATURE_DEFINITIONS, PreprocessingError, activity_indicators, get_preprocessing_report,
    prepare_features, safe_divide,
)

NAN = np.nan


def player(name, mat, bat=None, bowl=None, field=None):
    """One row in the merged-table format.

    bat   = (inns, not_outs, runs, balls_faced, fours, sixes), None = never batted
    bowl  = (inns, overs_text, runs, wickets),                  None = never bowled
    field = (inns, dismissals, catches, stumpings, keeper_catches), None = never fielded
    """
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


def sample_data():
    return pd.DataFrame([
        # Batter who never bowled.
        player("A Batter (IND)", 10, bat=(10, 2, 400, 300, 40, 10), field=(10, 5, 5, 0, 0)),
        # Same name, different country: must stay a separate player.
        player("A Batter (SA)", 6, bat=(6, 0, 120, 100, 12, 3), bowl=(2, "4.0", 30, 1), field=(6, 1, 1, 0, 0)),
        # Never dismissed -> batting average undefined.
        player("B Bowler (AUS)", 8, bat=(2, 2, 5, 6, 0, 0), bowl=(8, "30.0", 240, 12), field=(8, 2, 2, 0, 0)),
        # Keeper; bowled 10.1 overs (61 balls) with no wickets.
        player("C Keeper (ENG)", 6, bat=(5, 1, 100, 90, 8, 2), bowl=(1, "10.1", 70, 0), field=(6, 9, 6, 3, 6)),
        # Only 2 matches -> removed by the default filter.
        player("D Rookie (NZ)", 2, bat=(1, 0, 0, 0, 0, 0), field=(2, 0, 0, 0, 0)),
        # Faced 0 balls; 10.5 overs; no dismissals as a fielder.
        player("E Tailender (WI)", 5, bat=(1, 0, 0, 0, 0, 0), bowl=(5, "10.5", 80, 4), field=(5, 0, 0, 0, 0)),
        # Never batted.
        player("F NoBat (PAK)", 7, bowl=(7, "20.3", 150, 9), field=(7, 1, 1, 0, 0)),
    ])


class TestPrepareFeatures(unittest.TestCase):
    def setUp(self):
        self.data = sample_data()
        self.features = prepare_features(self.data)

    # 1. Valid data produces numeric features with the expected values.
    def test_valid_data_produces_numeric_features(self):
        self.assertEqual(list(self.features.columns), list(FEATURE_DEFINITIONS))
        self.assertTrue(all(pd.api.types.is_float_dtype(t) for t in self.features.dtypes))
        row = self.features.loc["A Batter (IND)"]
        self.assertAlmostEqual(row["bat_runs_per_inns"], 40.0)
        self.assertAlmostEqual(row["bat_strike_rate"], 100 * 400 / 300)
        self.assertAlmostEqual(row["bat_fours_per_inns"], 4.0)
        self.assertAlmostEqual(row["bat_sixes_per_inns"], 1.0)
        self.assertAlmostEqual(row["bat_average"], 400 / 8)
        self.assertAlmostEqual(row["field_catches_per_match"], 0.5)
        bowler = self.features.loc["B Bowler (AUS)"]
        self.assertAlmostEqual(bowler["bowl_wkts_per_match"], 12 / 8)
        self.assertAlmostEqual(bowler["bowl_economy"], 240 / 30)
        self.assertAlmostEqual(bowler["bowl_average"], 20.0)
        self.assertAlmostEqual(bowler["bowl_strike_rate"], 180 / 12)
        keeper = self.features.loc["C Keeper (ENG)"]
        self.assertAlmostEqual(keeper["field_stumpings_per_match"], 0.5)
        self.assertAlmostEqual(keeper["field_keeper_share"], 1.0)

    # 2. The input DataFrame is not mutated.
    def test_input_not_mutated(self):
        data = sample_data()
        before = data.copy()
        prepare_features(data)
        pd.testing.assert_frame_equal(data, before)

    # 3. Missing required columns give a clear error.
    def test_missing_required_column(self):
        with self.assertRaisesRegex(PreprocessingError, "Wkts_Bowl"):
            prepare_features(self.data.drop(columns=["Wkts_Bowl"]))
        with self.assertRaises(PreprocessingError):
            prepare_features({"not": "a DataFrame"})

    def test_non_numeric_value_is_an_error(self):
        data = self.data.copy()
        data["Runs_Bat"] = data["Runs_Bat"].astype(object)
        data.loc[0, "Runs_Bat"] = "abc"
        with self.assertRaisesRegex(PreprocessingError, "Runs_Bat"):
            prepare_features(data)

    # 4. The minimum-match filter works and is configurable.
    def test_min_matches_filter(self):
        self.assertNotIn("D Rookie (NZ)", self.features.index)
        self.assertEqual(len(self.features), 6)
        self.assertEqual(len(prepare_features(self.data, min_matches=0)), 7)
        self.assertEqual(set(prepare_features(self.data, min_matches=7).index),
                         {"A Batter (IND)", "B Bowler (AUS)", "F NoBat (PAK)"})
        for bad in (-1, 2.5, "5", True):
            with self.assertRaises(PreprocessingError):
                prepare_features(self.data, min_matches=bad)

    def test_match_counts_stored_as_text_are_not_dropped(self):
        data = self.data.copy()
        for col in ("Mat_Bat", "Mat_Bowl", "Mat_Field"):
            data[col] = data[col].astype(str)
        pd.testing.assert_frame_equal(prepare_features(data), self.features)

    def test_inconsistent_match_counts_are_an_error(self):
        data = self.data.copy()
        data.loc[0, "Mat_Bowl"] = 11
        with self.assertRaisesRegex(PreprocessingError, "Mat differs"):
            prepare_features(data)

    # 5. Players with no bowling activity.
    def test_player_who_never_bowled(self):
        row = self.features.loc["A Batter (IND)"]
        self.assertEqual(row["bowl_wkts_per_match"], 0.0)  # count: did not bowl -> 0 wickets
        for rate in ("bowl_economy", "bowl_average", "bowl_strike_rate"):
            self.assertTrue(np.isnan(row[rate]), rate)  # rates: undefined, not 0

    def test_player_who_never_batted(self):
        row = self.features.loc["F NoBat (PAK)"]
        for name in FEATURE_DEFINITIONS:
            if name.startswith("bat_"):
                self.assertTrue(np.isnan(row[name]), name)

    # 6. Undefined averages are not turned into zero.
    def test_undefined_averages_stay_nan(self):
        self.assertTrue(np.isnan(self.features.loc["B Bowler (AUS)", "bat_average"]))  # never out
        self.assertTrue(np.isnan(self.features.loc["C Keeper (ENG)", "bowl_average"]))  # 0 wickets
        self.assertTrue(np.isnan(self.features.loc["C Keeper (ENG)", "bowl_strike_rate"]))
        # A genuine average of 0 (0 runs, 1 dismissal) stays 0.
        self.assertEqual(self.features.loc["E Tailender (WI)", "bat_average"], 0.0)

    # 7. Zero denominators never give infinity.
    def test_zero_denominators_give_nan_not_inf(self):
        self.assertFalse(np.isinf(self.features.to_numpy()).any())
        self.assertTrue(np.isnan(self.features.loc["E Tailender (WI)", "bat_strike_rate"]))  # 0 balls
        self.assertTrue(np.isnan(self.features.loc["E Tailender (WI)", "field_keeper_share"]))  # 0 dismissals
        result = safe_divide(pd.Series([1.0, 0.0, 5.0]), pd.Series([0.0, 0.0, 2.0]))
        self.assertTrue(result.iloc[:2].isna().all())
        self.assertEqual(result.iloc[2], 2.5)

    def test_infinite_input_becomes_nan(self):
        data = self.data.copy()
        data["BF_Bat"] = data["BF_Bat"].astype(float)
        data.loc[0, "BF_Bat"] = np.inf
        features = prepare_features(data)
        self.assertFalse(np.isinf(features.to_numpy()).any())
        self.assertTrue(np.isnan(features.loc["A Batter (IND)", "bat_strike_rate"]))

    # 8. Full player identifiers stay intact and are not features.
    def test_player_identifiers_intact(self):
        self.assertEqual(self.features.index.name, "Player")
        self.assertIn("A Batter (IND)", self.features.index)
        self.assertIn("A Batter (SA)", self.features.index)
        self.assertTrue(self.features.index.is_unique)
        self.assertNotIn("Player", self.features.columns)
        self.assertNotAlmostEqual(self.features.loc["A Batter (IND)", "bat_runs_per_inns"],
                                  self.features.loc["A Batter (SA)", "bat_runs_per_inns"])

    def test_duplicate_player_is_an_error(self):
        data = pd.concat([self.data, self.data.head(1)], ignore_index=True)
        with self.assertRaisesRegex(PreprocessingError, "Duplicate"):
            prepare_features(data)

    # 9. Overs conversion uses cricket notation.
    def test_overs_to_balls(self):
        result = overs_to_balls(["10.0", "10.1", "10.5", "10", "0.3", 10.1, 10.5, "-", None])
        self.assertEqual(result.iloc[:7].tolist(), [60, 61, 65, 60, 3, 61, 65])
        self.assertTrue(result.iloc[7:].isna().all())
        for bad in ("10.6", "10.25", "abc"):
            with self.assertRaises(DataValidationError):
                overs_to_balls([bad])
        # Economy uses the converted balls: 70 runs off 10.1 overs (61 balls).
        self.assertAlmostEqual(self.features.loc["C Keeper (ENG)", "bowl_economy"], 70 / (61 / 6))

    # 10. Entirely missing or constant features are removed.
    def test_unusable_features_removed(self):
        no_bowlers = pd.DataFrame([
            player("P One (IND)", 6, bat=(6, 1, 100, 80, 10, 2), field=(6, 2, 2, 0, 0)),
            player("P Two (AUS)", 6, bat=(5, 0, 60, 70, 5, 1), field=(6, 1, 1, 0, 0)),
        ])
        features = prepare_features(no_bowlers)
        self.assertFalse(any(c.startswith("bowl_") for c in features.columns))
        report = get_preprocessing_report(no_bowlers, features)
        self.assertIn("bowl_economy", report["dropped_features"])
        self.assertNotIn("bowling", report["categories_represented"])
        self.assertEqual(report["rows_no_bowling"], 2)

    # 11. The report counts players accurately.
    def test_report_counts(self):
        report = get_preprocessing_report(self.data, self.features)
        self.assertEqual(report["initial_players"], 7)
        self.assertEqual(report["excluded_players"], 1)
        self.assertEqual(report["remaining_players"], 6)
        self.assertEqual(report["feature_count"], 12)
        self.assertEqual(report["missing_per_feature"], self.features.isna().sum().to_dict())
        self.assertEqual(report["never_batted"], 1)
        self.assertEqual(report["never_bowled"], 1)
        self.assertEqual(report["rows_no_batting"], 1)
        report10 = get_preprocessing_report(self.data, prepare_features(self.data, 10), 10)
        self.assertEqual((report10["excluded_players"], report10["remaining_players"]), (6, 1))

    # 12. All three categories are represented.
    def test_all_categories_represented(self):
        report = get_preprocessing_report(self.data, self.features)
        self.assertEqual(report["categories_represented"], ["batting", "bowling", "fielding"])
        for category in ("batting", "bowling", "fielding"):
            self.assertTrue(report["features_by_category"][category])


class TestActivityIndicators(unittest.TestCase):
    def setUp(self):
        self.data = sample_data()
        self.features = prepare_features(self.data, min_matches=0)
        self.activity = activity_indicators(self.data, self.features.index)

    def test_flags_come_from_raw_innings(self):
        a = self.activity
        self.assertEqual(a.loc["A Batter (IND)", "has_bowled"], 0)   # never bowled
        self.assertEqual(a.loc["C Keeper (ENG)", "has_bowled"], 1)   # bowled 10.1 overs, 0 wickets
        self.assertEqual(a.loc["F NoBat (PAK)", "has_batted"], 0)
        self.assertEqual(a.loc["C Keeper (ENG)", "has_kept_wicket"], 1)
        self.assertEqual(a.loc["B Bowler (AUS)", "has_kept_wicket"], 0)
        self.assertEqual(list(a.index), list(self.features.index))

    def test_unknown_player_is_an_error(self):
        with self.assertRaises(PreprocessingError):
            activity_indicators(self.data, ["Nobody (XYZ)"])

    def test_input_not_mutated(self):
        before = self.data.copy()
        activity_indicators(self.data, self.features.index)
        pd.testing.assert_frame_equal(self.data, before)


@unittest.skipUnless((DATA_DIR / "Batting" / "t20.csv").is_file(), "real T20 CSVs not present")
class TestRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.merged = load_data()["merged"]
        cls.features = prepare_features(cls.merged)
        cls.report = get_preprocessing_report(cls.merged, cls.features)

    def test_real_data_passes_through(self):
        self.assertEqual(self.report["initial_players"], len(self.merged))
        self.assertEqual(self.report["excluded_players"] + self.report["remaining_players"],
                         len(self.merged))
        self.assertEqual(self.report["remaining_players"], int((self.merged["Mat_Bat"] >= 5).sum()))
        self.assertEqual(self.report["categories_represented"], ["batting", "bowling", "fielding"])
        self.assertFalse(np.isinf(self.features.to_numpy()).any())
        self.assertTrue(self.features.index.is_unique)
        self.assertTrue(set(self.features.index) <= set(self.merged["Player"]))

    def test_rates_match_source_columns(self):
        """Computed rates agree with the file's rounded columns, and are NaN exactly where the file has '-'."""
        source = self.merged.set_index("Player").loc[self.features.index]
        checks = [("bat_average", "Ave_Bat", 0.01), ("bat_strike_rate", "SR_Bat", 0.01),
                  ("bowl_economy", "Econ_Bowl", 0.01), ("bowl_average", "Ave_Bowl", 0.01),
                  ("bowl_strike_rate", "SR_Bowl", 0.1)]  # SR_Bowl has 1 decimal place
        for feature, column, tolerance in checks:
            with self.subTest(feature=feature):
                ours, theirs = self.features[feature], source[column]
                self.assertTrue((ours.isna() == theirs.isna()).all())
                self.assertLessEqual((ours - theirs).abs().max(), tolerance + 1e-9)


if __name__ == "__main__":
    unittest.main()

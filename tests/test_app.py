"""Streamlit tests for app.py using the built-in headless AppTest runner.

Run from the project folder:
    python -m unittest discover -s tests -v
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from data_loader import DATA_DIR, DATASETS, load_data  # noqa: E402
from modeling import build_model_matrix, evaluate_k_range  # noqa: E402
from preprocessing import activity_indicators, prepare_features  # noqa: E402

APP_FILES = ["app.py", "data_loader.py", "preprocessing.py", "modeling.py", "charts.py"]
HAS_DATA = all((DATA_DIR / spec["file"]).is_file() for spec in DATASETS.values())
TIMEOUT = 120


def run_app(app_path=PROJECT_DIR / "app.py"):
    return AppTest.from_file(str(app_path), default_timeout=TIMEOUT).run()


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def download_ids(at):
    return [d.proto.url for d in at.get("download_button")]


def copy_app(target, players=None):
    """Copy the app into `target`; with `players`, also copy the CSVs filtered to those players."""
    for name in APP_FILES:
        shutil.copy(PROJECT_DIR / name, target / name)
    (target / ".streamlit").mkdir()
    shutil.copy(PROJECT_DIR / ".streamlit" / "config.toml", target / ".streamlit" / "config.toml")
    if players is not None:
        for spec in DATASETS.values():
            df = pd.read_csv(DATA_DIR / spec["file"], dtype=str, keep_default_na=False)
            (target / "data" / spec["file"]).parent.mkdir(parents=True, exist_ok=True)
            df[df["Player"].isin(players)].to_csv(target / "data" / spec["file"], index=False)
    return target / "app.py"


@unittest.skipUnless(HAS_DATA, "real T20 CSVs not present")
class TestAppWithData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.merged = load_data()["merged"]
        cls.at = run_app()

    def test_renders_without_exceptions(self):
        self.assertFalse(self.at.exception)
        self.assertFalse(self.at.error)
        self.assertEqual(len(self.at.tabs), 7)
        headers = [h.value for h in self.at.header]
        for expected in ("Data overview (T20)", "Feature engineering", "Choosing the number of clusters",
                         "Cluster analysis (K = 4)", "Player explorer", "Methodology"):
            self.assertIn(expected, headers)

    def test_metrics_match_live_computation(self):
        remaining = len(prepare_features(self.merged, min_matches=5))
        self.assertEqual(metric(self.at, "Source players"), f"{len(self.merged):,}")
        self.assertEqual(metric(self.at, "Players with ≥ 5 matches"), f"{remaining:,}")
        self.assertEqual(metric(self.at, "Remaining players"), str(remaining))  # Features tab agrees
        self.assertEqual(metric(self.at, "Selected K"), "4")

    def test_data_overview_still_available(self):
        self.assertTrue(any("All three datasets loaded and merged" in s.value for s in self.at.success))
        self.assertTrue(any(t.value.shape[0] == len(DATASETS) for t in self.at.table))

    def test_changing_k_updates_everything(self):
        at = run_app()
        before = download_ids(at)
        at.slider[1].set_value(3).run()
        self.assertFalse(at.exception)
        self.assertEqual(metric(at, "Selected K"), "3")
        self.assertIn("Cluster analysis (K = 3)", [h.value for h in at.header])
        overview_profiles = at.dataframe[0].value
        self.assertEqual(len(overview_profiles), 3)
        # Silhouette shown for K = 3 equals an independent computation.
        features = prepare_features(self.merged, min_matches=5)
        X = build_model_matrix(features, activity_indicators(self.merged, features.index))["X"]
        expected = evaluate_k_range(X, [3]).loc[3, "silhouette"]
        self.assertEqual(metric(at, "Silhouette score"), f"{expected:.3f}")
        # Downloads are regenerated from the new model (file ids are content hashes).
        after = download_ids(at)
        self.assertEqual(len(after), 2)
        self.assertNotEqual(before, after)
        at.slider[1].set_value(4).run()
        self.assertEqual(download_ids(at), before)  # same settings -> identical files

    def test_changing_min_matches(self):
        at = run_app()
        at.slider[0].set_value(10).run()
        self.assertFalse(at.exception)
        remaining = len(prepare_features(self.merged, min_matches=10))
        self.assertEqual(metric(at, "Players with ≥ 10 matches"), f"{remaining:,}")
        self.assertEqual(metric(at, "Remaining players"), str(remaining))

    def test_player_explorer_selection(self):
        at = run_app()
        player_box = next(s for s in at.selectbox if s.label.startswith("Player"))
        choice = player_box.options[1]
        player_box.select(choice).run()
        self.assertFalse(at.exception)
        self.assertIn(choice, [s.value for s in at.subheader])


class TestAppFailureStates(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_missing_data_files(self):
        at = run_app(copy_app(self.tmp))
        self.assertFalse(at.exception)
        self.assertTrue(any("file not found" in e.value for e in at.error))
        self.assertIn("Methodology", [h.value for h in at.header])  # page still usable

    @unittest.skipUnless(HAS_DATA, "real T20 CSVs not present")
    def test_k_too_large_for_data(self):
        players = load_data()["merged"].nlargest(7, "Mat_Bat")["Player"].tolist()
        at = run_app(copy_app(self.tmp, players))
        self.assertFalse(at.exception)
        at.slider[1].set_value(8).run()
        self.assertFalse(at.exception)
        self.assertTrue(any("too large" in e.value for e in at.error))
        # Data inspection keeps working when the model cannot be trained.
        self.assertTrue(any("All three datasets loaded and merged" in s.value for s in at.success))


if __name__ == "__main__":
    unittest.main()

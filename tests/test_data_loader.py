"""Tests for data_loader.py: schema validation, value parsing, merging, file safety.

Run from the project folder:
    python -m unittest discover -s tests -v
"""

import hashlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from data_loader import DATA_DIR, DATASETS, DataValidationError, load_data, overs_to_balls  # noqa: E402

HAS_DATA = all((DATA_DIR / spec["file"]).is_file() for spec in DATASETS.values())
NOTEBOOK = PROJECT_DIR / "notebooks" / "Cricket_Xray.ipynb"


def file_hashes():
    paths = [DATA_DIR / spec["file"] for spec in DATASETS.values()] + [NOTEBOOK]
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


class TestOversConversion(unittest.TestCase):
    def test_cricket_notation(self):
        result = overs_to_balls(["10.0", "10.1", "10.5", "0.3", "284.5", 10.1, 10, "-", None])
        self.assertEqual(result.iloc[:7].tolist(), [60, 61, 65, 3, 1709, 61, 60])
        self.assertTrue(result.iloc[7:].isna().all())

    def test_invalid_overs(self):
        for bad in ("10.6", "10.25", "ten", "1/2"):
            with self.assertRaises(DataValidationError):
                overs_to_balls([bad])


@unittest.skipUnless(HAS_DATA, "real T20 CSVs not present")
class TestRealFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = file_hashes()
        cls.data = load_data()

    def test_source_files_not_modified(self):
        self.assertEqual(file_hashes(), self.before)

    def test_raw_tables_equal_files(self):
        for name, spec in DATASETS.items():
            on_disk = pd.read_csv(DATA_DIR / spec["file"], dtype=str, keep_default_na=False)
            self.assertTrue(self.data["raw"][name].equals(on_disk), name)

    def test_schema_and_uniqueness(self):
        for name, spec in DATASETS.items():
            raw = self.data["raw"][name]
            self.assertTrue(set(spec["required"]) <= set(raw.columns), name)
            self.assertEqual(self.data["clean"][name]["Player"].duplicated().sum(), 0, name)
        merged = self.data["merged"]
        self.assertTrue(merged["Player"].is_unique)
        self.assertFalse(any(c.startswith("Unnamed") for c in merged.columns))

    def test_merge_matches_overlap(self):
        merged, overlap = self.data["merged"], self.data["overlap"]
        self.assertEqual(len(merged), overlap["In all three"])
        players = [set(self.data["clean"][n]["Player"]) for n in DATASETS]
        self.assertEqual(len(merged), len(set.intersection(*players)))

    def test_numeric_conversion_is_exact(self):
        """Every value except '-' parses to exactly the same number; '-' becomes NaN."""
        for name, spec in DATASETS.items():
            raw, clean = self.data["raw"][name], self.data["clean"][name]
            for col in spec["numeric"]:
                dash = raw[col].str.strip() == "-"
                self.assertTrue(clean.loc[dash, col].isna().all(), f"{name}.{col}")
                self.assertTrue(np.allclose(clean.loc[~dash, col], raw.loc[~dash, col].astype(float)),
                                f"{name}.{col}")

    def test_missing_summary_counts_dashes(self):
        missing = self.data["missing"].set_index(["Dataset", "Column"])["Missing"]
        for name, spec in DATASETS.items():
            for col in spec["numeric"]:
                expected = int((self.data["raw"][name][col].str.strip() == "-").sum())
                self.assertEqual(int(missing.get((name, col), 0)), expected, f"{name}.{col}")

    def test_special_formats(self):
        bat = self.data["clean"]["batting"]
        not_out = bat["HS"].str.endswith("*")
        self.assertTrue((bat.loc[not_out, "HS_NotOut"] == True).all())  # noqa: E712
        rebuilt = bat["HS_Runs"].dropna().astype(int).astype(str) + np.where(
            bat["HS_NotOut"].dropna() == True, "*", "")  # noqa: E712
        self.assertEqual(rebuilt.tolist(), bat.loc[bat["HS"] != "-", "HS"].tolist())

        bowl = self.data["clean"]["bowling"].dropna(subset=["Balls"])
        econ = bowl["Runs"] / (bowl["Balls"] / 6)
        self.assertLessEqual((econ - bowl["Econ"]).abs().max(), 0.01 + 1e-9)  # overs parsed as balls

    def test_fielding_identities(self):
        f = self.data["clean"]["fielding"].dropna(subset=["Dis"])
        self.assertTrue((f["Dis"] == f["Ct"] + f["St"]).all())
        self.assertTrue((f["Ct"] == f["Ct Wk"] + f["Ct Fi"]).all())


@unittest.skipUnless(HAS_DATA, "real T20 CSVs not present")
class TestValidationErrors(unittest.TestCase):
    """Each case runs on modified copies in a temporary folder, never on data/."""

    def run_with(self, mutate):
        tmp = Path(tempfile.mkdtemp())
        try:
            for spec in DATASETS.values():
                (tmp / spec["file"]).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(DATA_DIR / spec["file"], tmp / spec["file"])
            mutate(tmp)
            return load_data(tmp)
        finally:
            shutil.rmtree(tmp)

    @staticmethod
    def edit(dataset, change):
        def mutate(tmp):
            path = tmp / DATASETS[dataset]["file"]
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
            change(df).to_csv(path, index=False)
        return mutate

    def test_missing_file(self):
        with self.assertRaisesRegex(DataValidationError, "file not found"):
            self.run_with(lambda tmp: (tmp / DATASETS["bowling"]["file"]).unlink())

    def test_missing_column(self):
        with self.assertRaisesRegex(DataValidationError, "missing required column"):
            self.run_with(self.edit("batting", lambda df: df.drop(columns=["Runs"])))

    def test_duplicate_player(self):
        with self.assertRaisesRegex(DataValidationError, "duplicate player"):
            self.run_with(self.edit("fielding", lambda df: pd.concat([df, df.head(1)])))

    def test_unparseable_value(self):
        with self.assertRaisesRegex(DataValidationError, "unparseable"):
            self.run_with(self.edit("batting", lambda df: df.assign(Runs=df["Runs"].where(df.index != 3, "abc"))))

    def test_bad_overs(self):
        with self.assertRaisesRegex(DataValidationError, "Overs"):
            self.run_with(self.edit("bowling", lambda df: df.assign(Overs=df["Overs"].where(df.index != 0, "12.7"))))

    def test_no_common_players(self):
        with self.assertRaisesRegex(DataValidationError, "no player appears in all three"):
            self.run_with(self.edit("fielding", lambda df: df.assign(Player=df["Player"] + " X")))


if __name__ == "__main__":
    unittest.main()

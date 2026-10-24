"""Load, validate, and merge the T20 batting, bowling, and fielding datasets.

Usage:
    from data_loader import load_data
    data = load_data()
    merged = data["merged"]

Design notes:
- Raw CSV text is kept unchanged in data["raw"] so no information is lost.
- "-" in the source means "did not bat/bowl" or "not defined" (for example an
  average with zero dismissals). It becomes NaN here and is NOT replaced by 0;
  how to fill it is a preprocessing decision for a later phase.
- Any other value that cannot be parsed raises an error instead of being
  silently dropped.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

# Values that the source uses to mean "no value".
MISSING_TOKENS = {"-", ""}

# File location, required columns, and plain numeric columns for each dataset.
DATASETS = {
    "batting": {
        "file": "Batting/t20.csv",
        "suffix": "_Bat",
        "required": ["Player", "Span", "Mat", "Inns", "NO", "Runs", "HS", "Ave",
                     "BF", "SR", "100", "50", "0", "4s", "6s"],
        "numeric": ["Mat", "Inns", "NO", "Runs", "Ave", "BF", "SR",
                    "100", "50", "0", "4s", "6s"],
    },
    "bowling": {
        "file": "Bowling/Bowling_t20.csv",
        "suffix": "_Bowl",
        "required": ["Player", "Span", "Mat", "Inns", "Overs", "Mdns", "Runs",
                     "Wkts", "BBI", "Ave", "Econ", "SR", "4", "5"],
        "numeric": ["Mat", "Inns", "Mdns", "Runs", "Wkts", "Ave", "Econ",
                    "SR", "4", "5"],
    },
    "fielding": {
        "file": "Fielding/Fielding_t20.csv",
        "suffix": "_Field",
        "required": ["Player", "Span", "Mat", "Inns", "Dis", "Ct", "St",
                     "Ct Wk", "Ct Fi", "MD", "D/I"],
        "numeric": ["Mat", "Inns", "Dis", "Ct", "St", "Ct Wk", "Ct Fi", "D/I"],
    },
}


class DataValidationError(Exception):
    """Raised when a dataset is missing, malformed, or cannot be merged."""


def _is_missing(series):
    return series.str.strip().isin(MISSING_TOKENS)


def _to_number(series, table, column):
    """Convert text to numbers. "-" becomes NaN; anything else unparseable is an error."""
    values = pd.to_numeric(series.where(~_is_missing(series)), errors="coerce")
    bad = series[values.isna() & ~_is_missing(series)]
    if not bad.empty:
        raise DataValidationError(
            f"{table}: column '{column}' has {len(bad)} unparseable value(s), "
            f"e.g. {bad.unique()[:5].tolist()}"
        )
    return values


def _parse_pattern(series, pattern, table, column):
    """Extract regex groups as numbers. Missing tokens give NaN; other mismatches are errors."""
    parts = series.str.strip().str.extract(pattern)
    unmatched = series[parts.isna().all(axis=1) & ~_is_missing(series)]
    if not unmatched.empty:
        raise DataValidationError(
            f"{table}: column '{column}' has {len(unmatched)} value(s) in an "
            f"unexpected format, e.g. {unmatched.unique()[:5].tolist()}"
        )
    return parts.apply(pd.to_numeric)


def overs_to_balls(overs):
    """Convert cricket overs notation to the number of balls bowled.

    Overs are NOT decimals: "10.1" means 10 overs and 1 ball, so
    balls = whole_overs * 6 + balls_component. Examples: 10.0 -> 60,
    10.1 -> 61, 10.5 -> 65. The balls component must be 0 to 5.

    Accepts text ("284.5") or numbers (284.5). "-", "" and NaN give NaN.
    Any other value raises DataValidationError.
    """
    overs = pd.Series(overs)
    # Work on text so 10.1 is read as "10" and "1", never as the fraction 0.1.
    text = pd.Series([None if pd.isna(v) else str(v).strip() for v in overs],
                     index=overs.index, dtype=object)
    missing = text.isna() | text.isin(MISSING_TOKENS)
    parts = text.where(~missing).str.extract(r"^(\d+)(?:\.([0-5]))?$")
    bad = text[parts[0].isna() & ~missing]
    if not bad.empty:
        raise DataValidationError(
            f"Overs: {len(bad)} value(s) are not valid cricket overs "
            f"(whole overs + 0 to 5 balls), e.g. {bad.unique()[:5].tolist()}"
        )
    whole = pd.to_numeric(parts[0])
    balls = pd.to_numeric(parts[1]).fillna(0)
    return (whole * 6 + balls).rename("Balls")


def read_raw(name, data_dir=DATA_DIR):
    """Read one CSV exactly as text and check that its required columns exist."""
    spec = DATASETS[name]
    path = Path(data_dir) / spec["file"]
    if not path.is_file():
        raise DataValidationError(f"{name}: file not found at {path}")

    raw = pd.read_csv(path, dtype=str, keep_default_na=False)

    missing_cols = [c for c in spec["required"] if c not in raw.columns]
    if missing_cols:
        raise DataValidationError(f"{name}: missing required column(s) {missing_cols} in {path.name}")
    if (raw["Player"].str.strip() == "").any():
        raise DataValidationError(f"{name}: some rows have an empty Player name")
    duplicates = raw.loc[raw["Player"].duplicated(), "Player"]
    if not duplicates.empty:
        raise DataValidationError(
            f"{name}: {len(duplicates)} duplicate player row(s), e.g. {duplicates.head(5).tolist()}"
        )
    return raw


def clean_table(name, raw):
    """Drop exported index columns and convert values to numbers.

    Columns with a special format keep their original text, and the parsed
    numbers are added as new columns next to them.
    """
    spec = DATASETS[name]
    # "Unnamed: ..." columns are the scraper's page index and an empty trailing column.
    df = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")]).copy()
    df["Player"] = df["Player"].str.strip()

    for col in spec["numeric"]:
        df[col] = _to_number(df[col], name, col)

    if name == "batting":
        # "94*" means 94 not out.
        hs = _parse_pattern(df["HS"], r"^(\d+)\*?$", name, "HS")
        df["HS_Runs"] = hs[0]
        df["HS_NotOut"] = df["HS"].str.endswith("*").where(hs[0].notna())
    elif name == "bowling":
        # "5/6" = 5 wickets for 6 runs.
        bbi = _parse_pattern(df["BBI"], r"^(\d+)/(\d+)$", name, "BBI")
        df["BBI_Wkts"], df["BBI_Runs"] = bbi[0], bbi[1]
        # Overs use cricket notation: "284.5" = 284 overs and 5 balls, not 284.5.
        df["Balls"] = overs_to_balls(df["Overs"])
    elif name == "fielding":
        # "5 (5ct 0st)" = 5 dismissals in one innings; a plain "0" also appears.
        md = _parse_pattern(df["MD"], r"^(\d+)(?:\s*\((\d+)ct (\d+)st\))?$", name, "MD")
        df["MD_Dis"] = md[0]

    return df


def merge_tables(clean):
    """Merge the three cleaned tables on Player (inner join, one row per player).

    Every column except Player gets a suffix (_Bat, _Bowl, _Field) so that
    columns with the same name, such as Runs or Ave, stay distinguishable.
    """
    renamed = {
        name: df.rename(columns={c: c + DATASETS[name]["suffix"] for c in df.columns if c != "Player"})
        for name, df in clean.items()
    }
    try:
        merged = (
            renamed["batting"]
            .merge(renamed["bowling"], on="Player", how="inner", validate="one_to_one")
            .merge(renamed["fielding"], on="Player", how="inner", validate="one_to_one")
        )
    except pd.errors.MergeError as error:
        raise DataValidationError(f"Merge failed: {error}") from error

    if merged.empty:
        raise DataValidationError("Merge produced no rows: no player appears in all three datasets.")
    if merged["Player"].duplicated().any():
        raise DataValidationError("Merge produced duplicate player rows.")
    return merged


def player_overlap(clean):
    """Count how many players appear in each combination of datasets."""
    bat, bowl, field = (set(clean[n]["Player"]) for n in ("batting", "bowling", "fielding"))
    return {
        "In all three": len(bat & bowl & field),
        "Batting only (not in bowling or fielding)": len(bat - bowl - field),
        "Bowling only (not in batting or fielding)": len(bowl - bat - field),
        "Fielding only (not in batting or bowling)": len(field - bat - bowl),
        "In some but not all three": len((bat | bowl | field) - (bat & bowl & field)),
    }


def missing_summary(clean):
    """Count missing (NaN) values per column, for columns that have any."""
    rows = []
    for name, df in clean.items():
        counts = df.isna().sum()
        for col, count in counts[counts > 0].items():
            rows.append({"Dataset": name, "Column": col, "Missing": int(count),
                         "Missing %": round(100 * count / len(df), 1)})
    return pd.DataFrame(rows, columns=["Dataset", "Column", "Missing", "Missing %"])


def load_data(data_dir=DATA_DIR):
    """Load, validate, clean, and merge all three datasets.

    Returns a dict with:
        raw      - original CSV text for each dataset (unchanged)
        clean    - each dataset with numeric columns converted
        merged   - one row per player with all three datasets joined
        overlap  - player overlap counts
        missing  - missing-value summary per dataset and column
    Raises DataValidationError with a readable message if anything is wrong.
    """
    raw = {name: read_raw(name, data_dir) for name in DATASETS}
    clean = {name: clean_table(name, df) for name, df in raw.items()}
    return {
        "raw": raw,
        "clean": clean,
        "merged": merge_tables(clean),
        "overlap": player_overlap(clean),
        "missing": missing_summary(clean),
    }

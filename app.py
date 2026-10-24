"""Cricket X-Ray — setup check, data overview, and feature preview.

Phase 2: loads and validates the T20 data and previews the engineered features.
No machine learning is run yet.
"""

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import streamlit as st

from data_loader import DATASETS, DataValidationError, load_data
from preprocessing import (
    DEFAULT_MIN_MATCHES, EXCLUDED_FEATURES, FEATURE_DEFINITIONS,
    get_preprocessing_report, prepare_features,
)

# Paths are relative to this file, so the app works from any launch directory.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

CORE_PACKAGES = ["pandas", "numpy", "scikit-learn", "matplotlib", "streamlit"]

# (stage, done?)
ROADMAP = [
    ("Data loading and validation", True),
    ("Data preprocessing", True),
    ("Feature engineering", True),
    ("Feature scaling", False),
    ("K-Means clustering", False),
    ("PCA visualization", False),
    ("Cluster evaluation", False),
    ("Player exploration", False),
]


def package_version(name):
    """Return the installed version of a package, or None if missing."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@st.cache_data
def cached_load_data():
    # Cached so the CSVs are read once, not on every page interaction.
    return load_data(DATA_DIR)


st.set_page_config(page_title="Cricket X-Ray", layout="wide")

st.title("Cricket X-Ray")
st.write(
    "Discovering cricket player performance profiles from batting, bowling, "
    "and fielding statistics using unsupervised machine learning "
    "(K-Means clustering and PCA)."
)

st.info(
    "Phase 2: data loading, preprocessing, and feature engineering. No model has been trained yet."
)

st.header("Setup status")

# Check the data folder and list any CSV files (including subfolders).
if DATA_DIR.is_dir():
    st.success(f"Data folder found: `{DATA_DIR}`")
    csv_files = sorted(DATA_DIR.rglob("*.csv"))
    if csv_files:
        st.write(f"Found {len(csv_files)} CSV file(s):")
        for csv_file in csv_files:
            st.write(f"- `{csv_file.relative_to(DATA_DIR)}`")
    else:
        st.warning(
            "No CSV datasets found yet. Place the batting, bowling, and fielding "
            "CSV files in the `data/` folder."
        )
else:
    st.error(f"Data folder is missing: `{DATA_DIR}`. Create it to continue.")

st.subheader("Environment")
st.write(f"Python version: `{sys.version.split()[0]}`")

versions = {name: package_version(name) or "NOT INSTALLED" for name in CORE_PACKAGES}
st.table({"Package": list(versions.keys()), "Version": list(versions.values())})

if "NOT INSTALLED" in versions.values():
    st.error("Some core packages are missing. Run `pip install -r requirements.txt`.")
else:
    st.success("All core packages are installed.")

st.header("Data overview (T20)")

try:
    data = cached_load_data()
except DataValidationError as error:
    # Show the problem but keep the rest of the page working.
    st.error(f"Could not load the datasets: {error}")
    data = None

if data is not None:
    merged = data["merged"]
    st.success(
        f"All three datasets loaded and merged: {merged.shape[0]} players, "
        f"{merged.shape[1]} columns."
    )

    st.subheader("Source files")
    st.table({
        "Dataset": list(DATASETS),
        "File": [f"data/{spec['file']}" for spec in DATASETS.values()],
        "Rows": [len(data["raw"][name]) for name in DATASETS],
        "Columns (raw)": [data["raw"][name].shape[1] for name in DATASETS],
        "Columns (cleaned)": [data["clean"][name].shape[1] for name in DATASETS],
        "Duplicate players": [int(data["clean"][name]["Player"].duplicated().sum()) for name in DATASETS],
    })
    st.caption(
        "Cleaning drops the exported `Unnamed` index columns and adds parsed number "
        "columns for HS, BBI, Overs, and MD. The original text columns are kept."
    )

    st.subheader("Player overlap")
    st.table({"Group": list(data["overlap"]), "Players": list(data["overlap"].values())})

    st.subheader("Missing values")
    st.write(
        "In the source files, `-` means the player did not bat/bowl, or the value is "
        "undefined (for example a batting average with no dismissals). These are kept "
        "as missing values here; they are not filled in yet."
    )
    st.dataframe(data["missing"], hide_index=True)

    st.subheader("Merged dataset")
    st.write(f"Shape: **{merged.shape[0]} rows x {merged.shape[1]} columns**")
    st.dataframe(merged.head(10), hide_index=True)

    st.header("Feature engineering preview")
    st.info(
        "Preprocessing preview only. No scaling or clustering has been done and no "
        "model has been trained."
    )

    min_matches = st.slider(
        "Minimum T20 matches per player", min_value=1, max_value=30, value=DEFAULT_MIN_MATCHES,
        help="A practical threshold to avoid profiles based on one or two games. "
             "5 is a default, not a statistically optimal value.",
    )

    try:
        features = prepare_features(merged, min_matches=min_matches)
        report = get_preprocessing_report(merged, features, min_matches=min_matches)
    except Exception as error:
        # Show the problem but keep the rest of the page working.
        st.error(f"Preprocessing failed: {type(error).__name__}: {error}")
        features = None

    if features is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Initial players", report["initial_players"])
        col2.metric(f"Excluded (< {min_matches} matches)", report["excluded_players"])
        col3.metric("Remaining players", report["remaining_players"])
        col4.metric("Features", report["feature_count"])

        st.subheader("Features")
        st.write(
            "Categories represented: "
            + ", ".join(f"**{cat}** ({len(report['features_by_category'][cat])})"
                        for cat in report["categories_represented"])
        )
        st.dataframe(
            {
                "Feature": list(features.columns),
                "Category": [FEATURE_DEFINITIONS[c][0] for c in features.columns],
                "Definition": [FEATURE_DEFINITIONS[c][1] for c in features.columns],
                "Missing (NaN)": [report["missing_per_feature"][c] for c in features.columns],
            },
            hide_index=True,
        )
        if report["dropped_features"]:
            st.warning(f"Dropped (all missing or constant): {report['dropped_features']}")
        for name, reason in EXCLUDED_FEATURES.items():
            st.caption(f"`{name}` is not included: {reason}")

        st.subheader("Missing values")
        st.write(
            f"Of the remaining players, **{report['never_batted']}** never batted, "
            f"**{report['never_bowled']}** never bowled, and **{report['never_fielded']}** "
            "never fielded. For these players, count-based features (for example wickets per "
            "match) are 0, while rates (averages, strike rates, economy) are left missing, "
            "because an undefined average is not a poor average. Rates are also missing when "
            "the denominator is 0 (for example a batting average for a player who was never "
            "dismissed). They will be handled in the modelling phase."
        )

        st.subheader("Feature table (first 20 players)")
        st.dataframe(features.head(20).round(3))

        st.subheader("Summary statistics")
        st.dataframe(features.describe().T.round(3))

st.header("Roadmap")
for step_number, (step, done) in enumerate(ROADMAP, start=1):
    st.write(f"{step_number}. {step} — {'done' if done else 'planned'}")

st.warning("Not implemented yet: feature scaling, K-Means, PCA, and cluster evaluation.")

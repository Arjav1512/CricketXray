"""Cricket X-Ray: discovering cricket player performance profiles with K-Means and PCA.

Run with:  streamlit run app.py

The sidebar settings (minimum matches, K) drive every tab. All numbers on the
page are computed live from the CSV files in data/; nothing is hardcoded.
Visual design: theme tokens in .streamlit/config.toml, chart styling in charts.py,
and a small CSS block below for the few details the theme cannot express.
"""

import html
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import charts
from data_loader import DATASETS, DataValidationError, load_data
from modeling import (
    DEFAULT_K, EXCLUDED_FROM_MODEL, FEATURE_UNITS, INDICATOR_FEATURES, LABEL_THRESHOLD,
    MAX_K, MIN_DEFINED_FOR_MEDIAN, MIN_K, MODEL_FEATURES, SILHOUETTE_SAMPLE_SIZE, ModelingError,
    best_silhouette_k, build_model_matrix, centroid_distances, describe_silhouette,
    elbow_heuristic, evaluate_k_range, label, model_settings, most_similar_players, run_pipeline,
)
from preprocessing import (
    CATEGORIES, DEFAULT_MIN_MATCHES, EXCLUDED_FEATURES, FEATURE_DEFINITIONS, PreprocessingError,
    activity_indicators, get_preprocessing_report, prepare_features,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CORE_PACKAGES = ["pandas", "numpy", "scikit-learn", "matplotlib", "streamlit"]
KNOWN_ERRORS = (DataValidationError, PreprocessingError, ModelingError)

# Raw source columns shown in the Player Explorer (text kept exactly as in the CSV).
SOURCE_COLUMNS = {
    "batting": ["Span", "Mat", "Inns", "NO", "Runs", "HS", "Ave", "BF", "SR", "100", "50", "4s", "6s"],
    "bowling": ["Inns", "Overs", "Mdns", "Runs", "Wkts", "BBI", "Ave", "Econ", "SR", "4", "5"],
    "fielding": ["Inns", "Dis", "Ct", "St", "Ct Wk", "Ct Fi", "MD"],
}
# Merged columns shown in the compact data preview (all columns are one toggle away).
PREVIEW_COLUMNS = ["Player", "Span_Bat", "Mat_Bat", "Runs_Bat", "HS_Bat", "Ave_Bat", "SR_Bat",
                   "Wkts_Bowl", "Econ_Bowl", "BBI_Bowl", "Ct_Field", "St_Field"]

# Design-system details that .streamlit/config.toml cannot express. Only our own
# cx-* classes and plain heading tags are styled; no Streamlit internals are targeted.
CSS = """
<style>
h1, h2, h3 { letter-spacing: -0.02em; }
.cx-eyebrow {
  font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 0.68rem;
  letter-spacing: 0.1em; text-transform: uppercase; color: #707070; margin: 0 0 0.35rem 0;
}
.cx-lead { font-size: 1.05rem; line-height: 1.6; color: #A3A3A3; max-width: 46rem; margin: 0; }
.cx-meta {
  font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 0.72rem; letter-spacing: 0.06em;
  text-transform: uppercase; color: #707070; text-align: right; line-height: 1.9; margin: 0;
}
.cx-meta b { color: #A3A3A3; font-weight: 500; }
@media (max-width: 640px) { .cx-meta { text-align: left; } }  /* header columns stack on phones */
.cx-step-title { font-weight: 500; color: #F5F5F3; margin: 0 0 0.25rem 0; }
.cx-muted { color: #A3A3A3; font-size: 0.9rem; line-height: 1.55; margin: 0; }
</style>
"""


# ---------------------------------------------------------------------------
# Cached computation (keyed by the data files and the sidebar settings)
# ---------------------------------------------------------------------------

def data_signature():
    """Data folder plus size and modification time of each CSV. Passed to every cached
    function so that replacing the files (or using another folder) never shows stale results."""
    files = [DATA_DIR / spec["file"] for spec in DATASETS.values()]
    return (str(DATA_DIR),) + tuple(
        (f.name, f.stat().st_size, f.stat().st_mtime_ns) if f.is_file() else (f.name, None, None)
        for f in files)


@st.cache_data(show_spinner="Loading and validating the CSV files...", max_entries=4)
def cached_load_data(signature):
    return load_data(DATA_DIR)


@st.cache_data(show_spinner="Engineering features...", max_entries=32)
def cached_features(signature, min_matches):
    merged = cached_load_data(signature)["merged"]
    features = prepare_features(merged, min_matches=min_matches)
    return features, get_preprocessing_report(merged, features, min_matches=min_matches)


@st.cache_data(show_spinner="Training K-Means...", max_entries=32)
def cached_pipeline(signature, min_matches, k):
    return run_pipeline(cached_load_data(signature)["merged"], min_matches=min_matches, k=k)


@st.cache_data(show_spinner="Evaluating K = 2 to 8...", max_entries=32)
def cached_evaluation(signature, min_matches):
    merged = cached_load_data(signature)["merged"]
    features = prepare_features(merged, min_matches=min_matches)
    matrix = build_model_matrix(features, activity_indicators(merged, features.index))
    return evaluate_k_range(matrix["X"])


# ---------------------------------------------------------------------------
# Small display helpers
# ---------------------------------------------------------------------------

def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def eyebrow(text):
    """Small monospaced uppercase label, as used for section labels in the design system."""
    st.markdown(f'<p class="cx-eyebrow">{html.escape(text)}</p>', unsafe_allow_html=True)


def show_chart(fig, alt):
    st.pyplot(fig, alt=alt)
    plt.close(fig)


def chart_card(title, fig, alt, caption=None):
    with st.container(border=True):
        eyebrow(title)
        show_chart(fig, alt)
        if caption:
            st.caption(caption)


def show_error(where, error):
    """Known problems get a readable message; anything unexpected also shows the traceback."""
    if isinstance(error, KNOWN_ERRORS):
        st.error(str(error), title=f"{where} is unavailable", icon=":material/error:")
    else:
        st.error(f"{type(error).__name__}: {error}", title=f"{where} failed unexpectedly",
                 icon=":material/error:")
        st.exception(error)


def fmt(value, feature=None):
    """Format a real statistic for tables; NaN means undefined / too few players."""
    if pd.isna(value):
        return "n/a"
    if feature in ("has_batted", "has_bowled", "has_kept_wicket"):
        return f"{value:.0%}"
    return f"{value:,.2f}"


def cluster_title(cluster_id, interpretation):
    return f"Cluster {cluster_id} · {interpretation.loc[cluster_id, 'label']}"


def cluster_badge(cluster_id, interpretation=None):
    """Badge in the cluster's chart colour; the number is always shown so colour is not the only cue."""
    text = f"Cluster {cluster_id}" if interpretation is None else cluster_title(cluster_id, interpretation)
    st.badge(text, color=charts.cluster_badge_color(cluster_id))


def pc_description(loadings, component, top=3):
    weights = loadings[component]
    strongest = weights.abs().sort_values(ascending=False).index[:top]
    parts = [f"{'+' if weights[f] > 0 else '−'} {label(f).lower()} ({weights[f]:+.2f})" for f in strongest]
    return ", ".join(parts)


def section_intro(title_eyebrow, text):
    eyebrow(title_eyebrow)
    st.markdown(f'<p class="cx-muted">{html.escape(text)}</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page shell: settings, header, data and model
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Cricket X-Ray", page_icon=":material/sports_cricket:", layout="wide")
st.html(CSS)

with st.sidebar:
    eyebrow("Cricket X-Ray")
    st.markdown("**Analysis settings**")
    st.caption("Every tab, chart and download follows these settings.")
    st.space("small")
    eyebrow("01 · Player filter")
    min_matches = st.slider(
        "Minimum T20 matches", min_value=1, max_value=30, value=DEFAULT_MIN_MATCHES,
        help="Players with fewer matches are left out. 5 is a practical default to avoid "
             "profiles built on one or two games, not a statistically optimal value.",
    )
    eyebrow("02 · Clustering")
    k = st.slider(
        "Number of clusters (K)", min_value=MIN_K, max_value=MAX_K, value=DEFAULT_K,
        help="K-Means needs K in advance. Use the Model evaluation tab to compare values.",
    )
    st.caption("Changing a setting retrains the model.")
    st.divider()
    sidebar_status = st.container()  # filled in once the model has run
    with st.expander("Reproducibility settings", icon=":material/tune:"):
        st.table(pd.DataFrame(model_settings(k).items(), columns=["Setting", "Value"]).astype(str),
                 hide_index=True)

# Load data, features, model and evaluation. Each step can fail independently so
# the data tabs stay usable even when modeling is not possible.
data = features = report = result = evaluation = None
load_error = feature_error = model_error = evaluation_error = None
signature = data_signature()
try:
    data = cached_load_data(signature)
except Exception as error:  # shown to the user below, never swallowed
    load_error = error
if data is not None:
    try:
        features, report = cached_features(signature, min_matches)
    except Exception as error:
        feature_error = error
    try:
        result = cached_pipeline(signature, min_matches, k)
    except Exception as error:
        model_error = error
    try:
        evaluation = cached_evaluation(signature, min_matches)
    except Exception as error:
        evaluation_error = error

with sidebar_status:
    eyebrow("Current run")
    if result is not None:
        st.markdown(
            f":green[●] Model trained  \n"
            f"{len(result['labels']):,} players · {result['matrix']['X'].shape[1]} features · K = {k}"
        )
    elif data is not None:
        st.markdown(":red[●] Model not trained (see the Overview tab)")
    else:
        st.markdown(":red[●] Data not loaded")

header_left, header_right = st.columns([3, 2], vertical_alignment="bottom")
with header_left:
    eyebrow("T20 internationals · Unsupervised learning")
    st.title("Cricket X-Ray")
    st.markdown('<p class="cx-lead">Cricket performance intelligence</p>', unsafe_allow_html=True)
with header_right:
    players_text = "—" if data is None else f"{len(data['merged']):,} players"
    st.markdown(
        f'<p class="cx-meta">Dataset · <b>{players_text}</b><br>'
        f'Filter · <b>≥ {min_matches} matches</b><br>'
        f'Model · <b>K-Means · K = {k}</b></p>',
        unsafe_allow_html=True,
    )
st.space("small")

if load_error is not None:
    show_error("Data", load_error)
    st.info("Place the three T20 CSV files in `data/` (see the README) and refresh the page. "
            "The Methodology tab is still available.", icon=":material/info:")

tabs = st.tabs(["Overview", "Data", "Features", "Model evaluation", "Clusters", "Player explorer",
                "Methodology"])
overview_tab, data_tab, features_tab, evaluation_tab, clusters_tab, player_tab, method_tab = tabs

# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------
with overview_tab:
    st.space("small")
    st.markdown(
        '<p class="cx-lead">Cricket X-Ray groups international T20 players by <i>how</i> they '
        "contribute (batting, bowling, fielding, wicket-keeping) using only their career "
        "statistics. No player is labelled in advance: K-Means discovers the groups, PCA draws "
        "them in 2D, and explicit rules turn each group's statistics into a readable profile.</p>",
        unsafe_allow_html=True,
    )
    st.space("small")
    with st.container(horizontal=True):
        st.metric("Source players", "—" if data is None else f"{len(data['merged']):,}", border=True)
        st.metric(f"Players with ≥ {min_matches} matches",
                  "—" if report is None else f"{report['remaining_players']:,}", border=True)
        st.metric("Model features", "—" if result is None else result["matrix"]["X"].shape[1], border=True)
        st.metric("Selected K", k, border=True)
        st.metric("Silhouette score",
                  "—" if result is None or result["silhouette"] is None else f"{result['silhouette']:.3f}",
                  border=True)

    if result is None:
        if data is not None:
            show_error("The model", model_error)
    else:
        interpretation, labels = result["interpretation"], result["labels"]
        names = interpretation["label"].to_dict()
        sizes = labels.value_counts().sort_index()

        left, right = st.columns(2)
        with left:
            chart_card("PCA preview", charts.pca_scatter(result["pca"]["coords"], labels, names,
                                                         result["pca"]["explained"], width=4.4, height=3.5,
                                                         compact_legend=True),
                       alt="PCA scatter plot of players coloured by cluster",
                       caption="Full-size version, loadings and profiles in the Clusters tab.")
        with right:
            chart_card("Cluster distribution", charts.cluster_sizes_chart(sizes, names),
                       alt="Bar chart of players per cluster with counts and shares",
                       caption="Profile names are analytical interpretations of the statistics, not "
                               "official cricket roles, and they do not rank players by quality.")

        with st.container(border=True):
            eyebrow("Analysis summary")
            largest = sizes.idxmax()
            lines = [
                f"With **K = {k}** on **{len(labels):,} players** (at least {min_matches} matches), "
                f"the largest group is **{cluster_title(largest, interpretation)}** "
                f"({sizes[largest] / sizes.sum():.0%} of players).",
                f"The silhouette score is **{result['silhouette']:.3f}**, which is "
                f"{describe_silhouette(result['silhouette'])}." if result["silhouette"] is not None else
                "The silhouette score is not defined for this run.",
                f"The PCA map keeps **{result['pca']['explained']['total']:.0%}** of the variance of "
                f"the {result['matrix']['X'].shape[1]} standardized features.",
            ]
            if evaluation is not None and best_silhouette_k(evaluation) is not None:
                best_k, elbow_k = best_silhouette_k(evaluation), elbow_heuristic(evaluation)
                lines.append(
                    f"Across K = {MIN_K}–{MAX_K}, the highest silhouette is at K = {best_k} and the "
                    f"inertia curve bends most at K = {elbow_k}; see Model evaluation.")
            st.markdown("\n".join(f"- {line}" for line in lines))

    st.space("small")
    eyebrow("Workflow")
    steps = [
        ("Load and validate", "Check and merge the batting, bowling and fielding files on the full player name."),
        ("Engineer features", "Per-innings and per-match rates; undefined rates stay missing."),
        ("Prepare the model input", "Clip extremes, fill undefined rates with medians, add a has-bowled flag, standardize."),
        ("Cluster", "K-Means groups players; the elbow method and silhouette scores guide K."),
        ("Visualize and interpret", "PCA maps the clusters; rules turn real statistics into profiles."),
        ("Explore and export", "Inspect any player and download the results."),
    ]
    for row_start in range(0, len(steps), 3):
        for i, column in enumerate(st.columns(3), start=row_start):
            title, text = steps[i]
            with column:
                with st.container(border=True, height="stretch"):
                    eyebrow(f"{i + 1:02d}")
                    st.markdown(f'<p class="cx-step-title">{title}</p><p class="cx-muted">{text}</p>',
                                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------
with data_tab:
    st.header("Data overview (T20)")
    if data is None:
        show_error("Data", load_error)
    else:
        merged = data["merged"]
        rows_per_file = sorted({len(data["raw"][name]) for name in DATASETS})
        numeric_cells = sum(len(data["clean"][n]) * len(spec["numeric"]) for n, spec in DATASETS.items())
        missing_cells = int(data["missing"]["Missing"].sum())
        with st.container(horizontal=True):
            st.metric("Source files", len(DATASETS), border=True)
            st.metric("Players per file", " / ".join(f"{n:,}" for n in rows_per_file), border=True)
            st.metric("Merged players", f"{merged.shape[0]:,}", border=True)
            st.metric("Merged columns", merged.shape[1], border=True)
            st.metric("Missing numeric values", f"{missing_cells / numeric_cells:.1%}", border=True,
                      help=f"{missing_cells:,} of {numeric_cells:,} numeric source values are '-' "
                           "(did not bat/bowl, or not defined).")
        st.success(f"All three datasets loaded and merged: {merged.shape[0]} players, "
                   f"{merged.shape[1]} columns.", icon=":material/check_circle:")

        left, right = st.columns([3, 2])
        with left:
            with st.container(border=True, height="stretch"):
                eyebrow("Source files")
                st.table({
                    "Dataset": list(DATASETS),
                    "File": [f"data/{spec['file']}" for spec in DATASETS.values()],
                    "Rows": [len(data["raw"][name]) for name in DATASETS],
                    "Columns (raw)": [data["raw"][name].shape[1] for name in DATASETS],
                    "Columns (cleaned)": [data["clean"][name].shape[1] for name in DATASETS],
                    "Duplicate players": [int(data["clean"][name]["Player"].duplicated().sum()) for name in DATASETS],
                })
                st.caption("Cleaning drops the exported `Unnamed` index columns and adds parsed number "
                           "columns for HS, BBI, Overs, and MD. The original text columns are kept.")
        with right:
            with st.container(border=True, height="stretch"):
                eyebrow("Player overlap")
                st.table({"Group": list(data["overlap"]), "Players": list(data["overlap"].values())})

        with st.container(border=True):
            eyebrow("Missing values by source column")
            st.markdown('<p class="cx-muted">In the source files, <code>-</code> means the player did '
                        "not bat/bowl, or the value is undefined (for example a batting average with no "
                        "dismissals). The loader keeps these as missing values; it never turns them into 0.</p>",
                        unsafe_allow_html=True)
            st.space("small")
            st.dataframe(data["missing"], hide_index=True, alt="Missing values per source column",
                         column_config={"Missing %": st.column_config.NumberColumn(format="%.1f%%")})

        with st.container(border=True):
            eyebrow(f"Merged dataset · {merged.shape[0]} rows × {merged.shape[1]} columns")
            show_all = st.toggle("Show all columns", value=False)
            shown = merged if show_all else merged[[c for c in PREVIEW_COLUMNS if c in merged.columns]]
            st.dataframe(shown.head(10), hide_index=True, alt="First 10 rows of the merged dataset")
            st.caption(f"First 10 rows. {'All' if show_all else len(shown.columns)} of "
                       f"{merged.shape[1]} columns shown.")

    with st.expander("Setup status", icon=":material/settings:"):
        if DATA_DIR.is_dir():
            csv_files = sorted(DATA_DIR.rglob("*.csv"))
            st.write(f"Data folder: `{DATA_DIR}` ({len(csv_files)} CSV file(s) found)")
            for csv_file in csv_files:
                st.write(f"- `{csv_file.relative_to(DATA_DIR)}`")
        else:
            st.error(f"Data folder is missing: `{DATA_DIR}`.")
        st.write(f"Python version: `{sys.version.split()[0]}`")
        versions = {name: package_version(name) or "NOT INSTALLED" for name in CORE_PACKAGES}
        st.table({"Package": list(versions), "Version": list(versions.values())})

# ---------------------------------------------------------------------------
# 3. Features
# ---------------------------------------------------------------------------
with features_tab:
    st.header("Feature engineering")
    if features is None:
        show_error("Feature engineering", feature_error or load_error)
    else:
        st.markdown(f'<p class="cx-muted">Minimum-matches threshold: <b>{min_matches}</b>. '
                    "Adjust it under <i>Player filter</i> in the sidebar; every tab updates.</p>",
                    unsafe_allow_html=True)
        st.space("small")
        with st.container(horizontal=True):
            st.metric("Initial players", report["initial_players"], border=True)
            st.metric(f"Excluded (< {min_matches} matches)", report["excluded_players"], border=True)
            st.metric("Remaining players", report["remaining_players"], border=True)
            st.metric("Engineered features", report["feature_count"], border=True)

        eyebrow("Feature categories")
        category_columns = st.columns(len(CATEGORIES))
        for column, category in zip(category_columns, CATEGORIES):
            with column:
                with st.container(border=True, height="stretch"):
                    names_in_category = report["features_by_category"][category]
                    st.markdown(f"**{category.capitalize()}** · {len(names_in_category)}")
                    st.markdown("\n".join(f"- {label(f)}" for f in names_in_category) or "—")

        with st.container(border=True):
            eyebrow("Feature definitions")
            st.dataframe(
                {
                    "Feature": list(features.columns),
                    "Name": [label(c) for c in features.columns],
                    "Category": [FEATURE_DEFINITIONS[c][0] for c in features.columns],
                    "Definition": [FEATURE_DEFINITIONS[c][1] for c in features.columns],
                    "Unit": [FEATURE_UNITS.get(c, "") for c in features.columns],
                    "Missing (undefined)": [report["missing_per_feature"][c] for c in features.columns],
                },
                hide_index=True, alt="Engineered feature definitions",
            )
            if report["dropped_features"]:
                st.warning(f"Dropped (all missing or constant): {report['dropped_features']}")
            for name, reason in EXCLUDED_FEATURES.items():
                st.caption(f"`{name}` is not included: {reason}")

        with st.container(border=True):
            eyebrow("Missing values")
            st.markdown(
                f"Of the remaining players, **{report['never_batted']}** never batted, "
                f"**{report['never_bowled']}** never bowled, and **{report['never_fielded']}** never "
                "fielded. For these players, count-based features (for example wickets per match) are 0, "
                "while rates (averages, strike rates, economy) are left missing, because an undefined "
                "average is not a poor average. Rates are also missing when the denominator is 0 (for "
                "example a batting average for a player who was never dismissed). The Methodology tab "
                "explains how the model handles them."
            )

        with st.container(border=True):
            eyebrow("Engineered features · first 20 players")
            st.dataframe(features.head(20).round(3), alt="Engineered feature values for the first 20 players")
        with st.container(border=True):
            eyebrow("Summary statistics")
            st.dataframe(features.describe().T.round(3), alt="Summary statistics of the engineered features")

# ---------------------------------------------------------------------------
# 4. Model evaluation
# ---------------------------------------------------------------------------
with evaluation_tab:
    st.header("Choosing the number of clusters")
    if evaluation is None:
        show_error("Model evaluation", evaluation_error or load_error)
    else:
        elbow_k, best_k = elbow_heuristic(evaluation), best_silhouette_k(evaluation)
        selected_score = evaluation["silhouette"].get(k)
        st.markdown(
            f'<p class="cx-muted">K-Means was fitted for every K from {MIN_K} to {MAX_K} on the same '
            f"standardized input (players with at least {min_matches} matches). Orange marks the K "
            f"selected in the sidebar (K = {k}).</p>", unsafe_allow_html=True)
        st.space("small")
        with st.container(horizontal=True):
            st.metric("Selected K", k, border=True)
            st.metric("Silhouette at selected K",
                      "—" if pd.isna(selected_score) else f"{selected_score:.3f}", border=True)
            st.metric("Highest silhouette", "—" if best_k is None else f"K = {best_k}", border=True)
            st.metric("Sharpest elbow", "—" if elbow_k is None else f"K = {elbow_k}", border=True)
            st.metric("PCA variance (PC1 + PC2)",
                      "—" if result is None else f"{result['pca']['explained']['total']:.1%}", border=True)

        left, right = st.columns(2)
        with left:
            chart_card("Elbow method · inertia", charts.elbow_chart(evaluation, k, elbow_k),
                       alt="Elbow chart of inertia against K")
        with right:
            chart_card("Silhouette score", charts.silhouette_chart(evaluation, k),
                       alt="Silhouette score for each K")

        with st.container(border=True):
            eyebrow("What the metrics say")
            if elbow_k is None or best_k is None:
                # Do not st.stop() here: that would also stop the remaining tabs.
                st.warning("Too few valid K values to compare. Lower the minimum-matches threshold.")
            else:
                st.markdown(
                    f"- **Elbow:** the inertia curve bends most sharply at **K = {elbow_k}** "
                    "(largest second difference). Inertia always falls as K grows, so the elbow is a "
                    "visual heuristic, not proof of a correct K.\n"
                    f"- **Silhouette:** the highest score is at **K = {best_k}** "
                    f"({evaluation['silhouette'].max():.3f}, {describe_silhouette(evaluation['silhouette'].max())}).\n"
                    f"- **Selected K = {k}:** silhouette "
                    + ("not available" if pd.isna(selected_score) else
                       f"{selected_score:.3f} ({describe_silhouette(selected_score)})") + "."
                )
                if elbow_k == best_k:
                    st.info(f"Both heuristics point to K = {best_k}.", icon=":material/info:")
                else:
                    st.info(
                        f"The heuristics disagree (elbow K = {elbow_k}, best silhouette K = {best_k}). "
                        "Neither is decisive: the choice should also consider whether the clusters are "
                        "interpretable in cricket terms, which is why K stays adjustable.",
                        icon=":material/info:")
            st.caption(
                f"Silhouette scores use every player up to {SILHOUETTE_SAMPLE_SIZE:,}, and a seeded "
                "random sample of that size above it. Scores between 0.25 and 0.5 are conventionally read "
                "as weak structure: cricket roles overlap (many players both bat and bowl), so clear, "
                "well-separated clusters are not expected."
            )

        with st.container(border=True):
            eyebrow("All values")
            table = evaluation.reset_index().rename(columns={"k": "K", "inertia": "Inertia",
                                                             "silhouette": "Silhouette", "note": "Note"})
            if not table["Note"].astype(bool).any():
                table = table.drop(columns="Note")
            st.dataframe(table.round(3), hide_index=True, alt="Inertia and silhouette score for each K")

# ---------------------------------------------------------------------------
# 5. Clusters
# ---------------------------------------------------------------------------
with clusters_tab:
    st.header(f"Cluster analysis (K = {k})")
    if result is None:
        show_error("Cluster analysis", model_error or load_error)
    else:
        interpretation, labels, pca = result["interpretation"], result["labels"], result["pca"]
        names = interpretation["label"].to_dict()
        sizes = labels.value_counts().sort_index()

        with st.container(horizontal=True):
            st.metric("PC1 variance explained", f"{pca['explained']['PC1']:.1%}", border=True)
            st.metric("PC2 variance explained", f"{pca['explained']['PC2']:.1%}", border=True)
            st.metric("PC1 + PC2 combined", f"{pca['explained']['total']:.1%}", border=True)
            st.metric("Inertia", f"{result['inertia']:,.1f}", border=True)

        chart_card(
            "PCA projection · players on the first two principal components",
            charts.pca_scatter(pca["coords"], labels, names, pca["explained"]),
            alt="PCA scatter plot of players coloured by cluster",
            caption=(
                f"A 2D projection of a {result['matrix']['X'].shape[1]}-dimensional space, keeping "
                f"{pca['explained']['total']:.0%} of the variance. Clusters were found in the full "
                "space, so some overlap in 2D is expected. Each cluster has its own colour and marker shape. "
                f"PC1 is driven mostly by {pc_description(pca['loadings'], 'PC1')}; "
                f"PC2 by {pc_description(pca['loadings'], 'PC2')}. The sign of a component is arbitrary."
            ),
        )

        left, right = st.columns(2)
        with left:
            chart_card("Players per cluster", charts.cluster_sizes_chart(sizes, names),
                       alt="Bar chart of players per cluster")
        with right:
            chart_card("PCA loadings · feature weights", charts.loadings_chart(pca["loadings"], label),
                       alt="Bar chart of PCA loadings for each feature")

        st.subheader("Cluster profiles")
        st.caption(
            f"Labels come from explicit rules on each cluster's relative profile (threshold "
            f"{LABEL_THRESHOLD} SD; see Methodology). They are analytical interpretations, not "
            "official classifications, and they do not identify the best players or predict "
            "future performance."
        )
        cluster_ids = list(interpretation.index)
        per_row = min(k, 4)
        for row_start in range(0, len(cluster_ids), per_row):
            row = cluster_ids[row_start:row_start + per_row]
            for column, cluster_id in zip(st.columns(per_row), row):
                with column, st.container(border=True, height="stretch"):
                    cluster_badge(cluster_id)
                    st.markdown(f"**{interpretation.loc[cluster_id, 'label']}**")
                    st.caption(f"{int(result['profile'].loc[cluster_id, 'players'])} players · "
                               f"{sizes[cluster_id] / sizes.sum():.0%}")
                    st.markdown(f'<p class="cx-muted">{html.escape(interpretation.loc[cluster_id, "explanation"])}</p>',
                                unsafe_allow_html=True)

        profile, overall = result["profile"], result["overall"]
        rows = [c for c in profile.columns if c != "players"]
        table = pd.DataFrame(
            {f"{c} · {names[c]}": [fmt(profile.loc[c, f], f) for f in rows] for c in profile.index}
            | {"All modeled players": [fmt(overall[f], f) for f in rows]},
        )
        table.insert(0, "Unit", [FEATURE_UNITS.get(f, "") for f in rows])
        table.insert(0, "Statistic (median)", [label(f) for f in rows])
        with st.container(border=True):
            eyebrow("Profile table · real statistics")
            st.dataframe(table, hide_index=True, alt="Median real statistics per cluster")
            st.caption(
                "Values are real cricket statistics: medians over the cluster's players, using only "
                "players for whom the statistic is defined. Activity rows are the share of players. "
                f"'n/a' means fewer than {MIN_DEFINED_FOR_MEDIAN} players in the cluster have that "
                "statistic (for example bowling rates in a cluster of non-bowlers)."
            )

        chart_card(
            "Relative strengths · standard deviations from the average player",
            charts.relative_heatmap(result["relative"], label, names),
            alt="Heatmap of cluster centres in standard deviations",
            caption=(
                "Each cell is the cluster's average standardized value: +1 (orange) means one standard "
                "deviation above the average modeled player, −1 (violet) one below. These are relative "
                "scores, not real statistics: −1 economy means a lower (better) economy than average, not "
                "a negative economy. Undefined rates were filled with medians before standardizing."
            ),
        )

        with st.container(border=True):
            eyebrow("Download results")
            summary = profile.copy()
            summary.insert(0, "label", interpretation["label"])
            summary["explanation"] = interpretation["explanation"]
            with st.container(horizontal=True):
                st.download_button(
                    "Player results (CSV)", result["results"].to_csv(index=False).encode("utf-8"),
                    file_name=f"cricket_xray_players_k{k}_min{min_matches}.csv", mime="text/csv",
                    icon=":material/download:", on_click="ignore", type="primary",
                )
                st.download_button(
                    "Cluster summary (CSV)", summary.to_csv().encode("utf-8"),
                    file_name=f"cricket_xray_clusters_k{k}_min{min_matches}.csv", mime="text/csv",
                    icon=":material/download:", on_click="ignore",
                )
            st.caption(f"Both files match the current settings (K = {k}, minimum {min_matches} "
                       "matches). Player results contain the full player name, cluster, label, PCA "
                       "coordinates and engineered features; undefined rates are left blank.")

# ---------------------------------------------------------------------------
# 6. Player explorer
# ---------------------------------------------------------------------------
with player_tab:
    st.header("Player explorer")
    if result is None:
        show_error("Player explorer", model_error or load_error)
    else:
        interpretation, labels = result["interpretation"], result["labels"]
        X, features_used = result["matrix"]["X"], result["features"]
        matches = data["merged"].set_index("Player")["Mat_Bat"]

        filter_options = ["All clusters"] + [cluster_title(c, interpretation) for c in interpretation.index]
        left, right = st.columns([1, 2], vertical_alignment="bottom")
        with left:
            cluster_filter = st.selectbox("Show players from", filter_options)
        if cluster_filter == "All clusters":
            players = sorted(labels.index)
        else:
            chosen = interpretation.index[filter_options.index(cluster_filter) - 1]
            players = sorted(labels[labels == chosen].index)
        default_player = matches.loc[players].idxmax()  # most-capped player as a starting point
        with right:
            player = st.selectbox("Player (type to search)", players, index=players.index(default_player),
                                  help="Full player name including the country tag, as in the source data.")

        cluster_id = labels[player]
        activity = result["activity"].loc[player]
        with st.container(border=True):
            eyebrow("Player")
            st.subheader(player)
            with st.container(horizontal=True):
                cluster_badge(cluster_id, interpretation)
                st.badge(f"{int(matches[player])} T20 matches", color="gray")
                st.badge("Has bowled" if activity["has_bowled"] else "Has not bowled", color="gray")
                if activity["has_kept_wicket"]:
                    st.badge("Has kept wicket", color="gray")
            st.markdown(f'<p class="cx-muted">{html.escape(interpretation.loc[cluster_id, "explanation"])}</p>',
                        unsafe_allow_html=True)

        with st.container(horizontal=True):
            for feature in ["bat_runs_per_inns", "bat_strike_rate", "bowl_wkts_per_match",
                            "bowl_economy", "field_catches_per_match"]:
                if feature in features_used.columns:
                    st.metric(label(feature), fmt(features_used.loc[player, feature]), border=True,
                              help=FEATURE_UNITS.get(feature))

        left, right = st.columns(2)
        with left:
            with st.container(border=True, height="stretch"):
                eyebrow("Engineered features")
                same_cluster = labels[labels == cluster_id].index
                st.dataframe(
                    pd.DataFrame({
                        "Feature": [label(f) for f in features_used.columns],
                        "Player": [fmt(features_used.loc[player, f]) for f in features_used.columns],
                        "Cluster median": [fmt(features_used.loc[same_cluster, f].median()) for f in features_used.columns],
                        "All players median": [fmt(features_used[f].median()) for f in features_used.columns],
                        "Unit": [FEATURE_UNITS.get(f, "") for f in features_used.columns],
                    }),
                    hide_index=True, alt="Player's engineered features compared with medians",
                )
                st.caption("'n/a' = undefined for this player (for example no wickets, so no bowling average).")
        with right:
            with st.container(border=True, height="stretch"):
                eyebrow("Why this cluster")
                distances = centroid_distances(X, result["centers"], player)
                nearest, second = distances.index[0], distances.index[1]
                st.markdown(
                    f"K-Means assigns each player to the nearest cluster centre in the standardized "
                    f"feature space. {player} is closest to **Cluster {nearest}** (distance "
                    f"{distances.iloc[0]:.2f}); the next closest is Cluster {second} "
                    f"({distances.iloc[1]:.2f})."
                )
                centre = result["centers"].loc[cluster_id]
                st.dataframe(
                    pd.DataFrame({
                        "Model feature": [label(f) for f in X.columns],
                        "Player (SD)": X.loc[player].round(2).to_numpy(),
                        "Cluster centre (SD)": centre.round(2).to_numpy(),
                    }),
                    hide_index=True, alt="Player's standardized values compared with the cluster centre",
                )
                st.caption("Standardized values (SD from the average modeled player), after median "
                           "filling of undefined rates. They are relative scores, not statistics.")

        with st.container(border=True):
            eyebrow("Source statistics")
            st.caption("Exactly as in the CSV files ('-' = did not bat/bowl or not defined; '*' = not out).")
            for dataset, cols in SOURCE_COLUMNS.items():
                raw = data["raw"][dataset].assign(Player=lambda d: d["Player"].str.strip()).set_index("Player")
                st.markdown(f"**{dataset.capitalize()}**")
                st.dataframe(raw.loc[[player], cols], hide_index=True, alt=f"{dataset} source statistics")

        with st.container(border=True):
            eyebrow("Most similar players")
            similar = most_similar_players(X, player, n=5)
            st.dataframe(
                pd.DataFrame({
                    "Player": similar.index,
                    "Distance": similar.round(3).to_numpy(),
                    "Cluster": [cluster_title(labels[p], interpretation) for p in similar.index],
                }),
                hide_index=True, alt="Five most similar players by Euclidean distance",
            )
            st.caption("Nearest neighbours by Euclidean distance in the same standardized feature space "
                       "used by K-Means (smaller = more similar statistical profile).")

        with st.container(border=True):
            eyebrow(f"Other players in {cluster_title(cluster_id, interpretation)}")
            members = result["results"][result["results"]["cluster_id"] == cluster_id]
            members = members.assign(Matches=members["Player"].map(matches).astype(int))
            member_cols = ["Player", "Matches", "bat_runs_per_inns", "bat_strike_rate", "bowl_wkts_per_match",
                           "bowl_economy", "field_catches_per_match", "field_keeper_share"]
            st.dataframe(
                members[member_cols].rename(columns={c: label(c) for c in member_cols})
                .sort_values("Matches", ascending=False).round(2),
                hide_index=True, alt="Players in the same cluster",
            )

# ---------------------------------------------------------------------------
# 7. Methodology
# ---------------------------------------------------------------------------
METHOD_SECTIONS = [
    ("Problem", "Cricket players are usually described by fixed roles (batter, bowler, all-rounder, "
     "wicket-keeper). This project asks whether similar profiles can be *discovered* from career "
     "statistics alone, without giving the algorithm any labels."),
    ("Why unsupervised learning", "There is no trustworthy label column to learn from: roles are "
     "informal and many players do several jobs. Clustering groups players by the similarity of "
     "their numbers, and the groups are interpreted afterwards."),
    ("Preprocessing and features", "Players below the minimum-matches threshold are removed. Career "
     "totals become rates so that long and short careers are comparable, e.g. runs per innings, "
     "strike rate, wickets per match, economy, catches per match. Cricket overs (10.5 = 10 overs and "
     "5 balls) are converted to balls before computing economy and strike rate."),
    ("Missing values", "A player who never bowled has 0 wickets per match, but no economy or bowling "
     "average: those are *undefined*, not 0, and stay missing during feature engineering. For the "
     "model, which cannot accept missing values:\n"
     "- undefined rates are filled with the feature's **median** over players where it is defined "
     "(fitted on the modeled players only), so a non-bowler gets a typical economy, never an elite one;\n"
     "- a **has bowled** flag, read from the raw innings column (never from filled values), plus 0 "
     "wickets per match keeps non-bowlers separate from real bowlers;\n"
     "- each feature is **clipped to its 1st–99th percentile** so a few tiny-sample extremes (e.g. an "
     "economy of 24 from two overs) cannot dominate distances."),
    ("StandardScaler", "Strike rate is in the hundreds while wickets per match is around 1. K-Means "
     "uses distances, so without scaling the largest-unit features would dominate. Each feature is "
     "rescaled to mean 0 and standard deviation 1."),
    ("K-Means", "Chooses K centres and assigns every player to the nearest centre, moving the centres "
     "until the within-cluster sum of squared distances (**inertia**) stops falling. `k-means++` "
     "initialization, 10 restarts and a fixed random seed make results reproducible."),
    ("Elbow method", "Plots inertia against K. Inertia always decreases with more clusters; the "
     "\"elbow\" where improvement slows is a heuristic for a sensible K."),
    ("Silhouette score", "For each player, compares the average distance to its own cluster (a) with "
     "the nearest other cluster (b): s = (b − a) / max(a, b), between −1 and 1. Higher means better "
     "separated clusters."),
    ("PCA", "Finds the two directions of greatest variance in the standardized features and projects "
     "players onto them to draw a 2D map. It is used for visualization only; the loadings show which "
     "features drive each axis."),
    ("Interpreting clusters", "Each cluster gets a label from explicit rules on its relative profile "
     "(e.g. batting score at least 0.5 SD above average → Batting-Oriented). Clusters that match no "
     "rule keep a neutral name."),
]

with method_tab:
    st.header("Methodology")
    if data is not None:
        span_end = data["merged"]["Span_Bat"].str.split("-").str[-1].astype(int).max()
        data_scope = (f"{len(data['merged']):,} players, careers up to {span_end}, from the files in "
                      f"`data/` (" + ", ".join(f"`{spec['file']}`" for spec in DATASETS.values()) + ")")
    else:
        data_scope = "the three T20 CSV files in `data/` (not loaded)"
    sections = METHOD_SECTIONS[:2] + [(
        "Data", "Career tables for T20 internationals, one CSV each for batting, bowling and "
        f"fielding: {data_scope}. The three files are validated and merged on the full player name "
        "*including the country tag* (some names repeat across countries).")] + METHOD_SECTIONS[2:]

    for number, (title, body) in enumerate(sections, start=1):
        st.divider()
        left, right = st.columns([1, 3])
        with left:
            eyebrow(f"{number:02d}")
            st.markdown(f"**{title}**")
        with right:
            st.markdown(body)

    if result is not None:
        st.divider()
        st.subheader("Model input used in this run")
        used = list(result["matrix"]["X"].columns)
        input_table = pd.DataFrame({
            "Feature": [label(f) for f in used],
            "Type": ["activity flag (0/1)" if f in INDICATOR_FEATURES else "rate" for f in used],
            "Values filled with median": [int(result["matrix"]["imputed"].get(f, 0)) for f in used],
            "Median used": [fmt(result["matrix"]["medians"].get(f)) for f in used],
            "Clipped to": [
                "—" if f not in result["matrix"]["clip_bounds"].index else
                f"{result['matrix']['clip_bounds'].loc[f, 'lower']:.2f} – {result['matrix']['clip_bounds'].loc[f, 'upper']:.2f}"
                for f in used],
        })
        st.dataframe(input_table, hide_index=True, alt="Model input features and imputation details")

        st.subheader("Engineered features left out of the model")
        feats = result["features"]
        rows = []
        for feature, (kept, reason) in EXCLUDED_FROM_MODEL.items():
            if feature in feats.columns and kept in feats.columns:
                r = feats[[feature, kept]].dropna().corr().iloc[0, 1]
                rows.append({"Left out": label(feature), "Overlaps with": label(kept),
                             "Correlation (r)": round(r, 2), "Reason": reason})
        st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Features excluded from the model")
        st.caption("Correlations are computed live on the current players. Left-out features "
                   "still appear in cluster profiles, the player explorer and the downloads.")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Limitations")
        st.markdown("""
- Career totals cannot separate batting position, era, opposition strength or format changes.
- Median filling makes non-bowlers look like typical bowlers on economy and strike rate; the
  *has bowled* flag and wickets per match offset this, but not perfectly.
- *Has bowled* counts any bowling at all, so part-time bowlers count as bowlers.
- K-Means assumes roughly round, similar-sized clusters; real roles overlap, so silhouette
  scores are modest.
- Labels come from simple rules and describe statistics, not official roles or player quality.
- The data is a historical snapshot and is not committed to the repository.
""")
    with right:
        st.subheader("Future improvements")
        st.markdown("""
- Add ODI and Test data (already present in `data/`) for multi-format profiles.
- Try Gaussian mixture models or hierarchical clustering, which allow softer or nested groups.
- Weight recent seasons more heavily, or use ball-by-ball data for batting position and phase of play.
""")

"""Imputation, scaling, K-Means clustering, evaluation, PCA and cluster interpretation.

Usage:
    from data_loader import load_data
    from modeling import run_pipeline, evaluate_k_range

    merged = load_data()["merged"]
    result = run_pipeline(merged, min_matches=5, k=4)
    evaluation = evaluate_k_range(result["matrix"]["X"])

Pipeline (every step is fitted on the clustering input only, i.e. the players
that passed the minimum-matches filter):

1. Select model features from the Phase 2 feature table (see MODEL_FEATURES;
   redundant features are left out, see EXCLUDED_FROM_MODEL).
2. Add the activity indicator has_bowled, read from the raw innings columns.
3. Clip each continuous feature to its 1st-99th percentile, so a handful of
   tiny-sample extremes (e.g. an economy of 24) cannot dominate distances.
4. Fill undefined rates with the feature's median among players where it is
   defined. Non-bowlers therefore get a typical (median) economy and strike
   rate, never an elite one, and has_bowled = 0 plus 0 wickets per match keep
   them apart from real bowlers.
5. Standardize with StandardScaler (mean 0, standard deviation 1).
6. K-Means with a fixed random_state; clusters are renumbered 1..K by size.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from preprocessing import FEATURE_DEFINITIONS, activity_indicators, prepare_features

# Reproducibility and model settings.
RANDOM_STATE = 42
N_INIT = 10            # K-Means restarts; the best (lowest inertia) run is kept
MAX_ITER = 300
DEFAULT_K = 4
MIN_K, MAX_K = 2, 8
SILHOUETTE_SAMPLE_SIZE = 2000   # exact silhouette up to this many players, sampled above
CLIP_QUANTILES = (0.01, 0.99)
LABEL_THRESHOLD = 0.5  # "clearly above average" = at least half a standard deviation
DOMINANCE_GAP = 1.0    # one discipline at least 1 SD stronger than the other
MIN_DEFINED_FOR_MEDIAN = 10

# Continuous features used by the model (all defined in preprocessing.py).
MODEL_FEATURES = [
    "bat_runs_per_inns", "bat_strike_rate", "bat_sixes_per_inns",
    "bowl_wkts_per_match", "bowl_economy", "bowl_strike_rate",
    "field_catches_per_match", "field_keeper_share",
]
# 0/1 activity indicators added from the raw data (never imputed).
INDICATOR_FEATURES = ["has_bowled"]

# Engineered features kept for profiles and exports but left out of the model,
# with the feature each one duplicates (the app shows the live correlation).
EXCLUDED_FROM_MODEL = {
    "bat_fours_per_inns": ("bat_runs_per_inns", "moves almost in step with runs per innings"),
    "bat_average": ("bat_runs_per_inns", "strongly tied to runs per innings and undefined for players never dismissed"),
    "bowl_average": ("bowl_strike_rate", "exactly equals economy x strike rate / 6, so it adds no new information"),
    "field_stumpings_per_match": ("field_keeper_share", "strongly tied to keeper share; keeping both would double-weight wicket-keepers"),
}

FEATURE_LABELS = {
    "bat_runs_per_inns": "Runs per innings",
    "bat_strike_rate": "Batting strike rate",
    "bat_fours_per_inns": "Fours per innings",
    "bat_sixes_per_inns": "Sixes per innings",
    "bat_average": "Batting average",
    "bowl_wkts_per_match": "Wickets per match",
    "bowl_economy": "Economy rate",
    "bowl_average": "Bowling average",
    "bowl_strike_rate": "Bowling strike rate",
    "field_catches_per_match": "Catches per match",
    "field_stumpings_per_match": "Stumpings per match",
    "field_keeper_share": "Keeper share",
    "has_batted": "Has batted",
    "has_bowled": "Has bowled",
    "has_kept_wicket": "Has kept wicket",
}

FEATURE_UNITS = {
    "bat_runs_per_inns": "runs / innings",
    "bat_strike_rate": "runs / 100 balls",
    "bat_fours_per_inns": "fours / innings",
    "bat_sixes_per_inns": "sixes / innings",
    "bat_average": "runs / dismissal",
    "bowl_wkts_per_match": "wickets / match",
    "bowl_economy": "runs / over (lower is better)",
    "bowl_average": "runs / wicket (lower is better)",
    "bowl_strike_rate": "balls / wicket (lower is better)",
    "field_catches_per_match": "catches / match",
    "field_stumpings_per_match": "stumpings / match",
    "field_keeper_share": "fraction of dismissals as keeper (0-1)",
    "has_batted": "share of players (0-1)",
    "has_bowled": "share of players (0-1)",
    "has_kept_wicket": "share of players (0-1)",
}


class ModelingError(Exception):
    """Raised when the model cannot be built with the current data or settings."""


def label(name):
    return FEATURE_LABELS.get(name, name)


# ---------------------------------------------------------------------------
# Model input
# ---------------------------------------------------------------------------

def build_model_matrix(features, activity):
    """Clip, impute and standardize the model features.

    features: output of prepare_features() (undefined rates are NaN).
    activity: output of activity_indicators() for the same players.

    Returns a dict with:
        X           standardized matrix (DataFrame indexed by Player)
        X_processed clipped + imputed values before scaling (real units)
        scaler      the fitted StandardScaler
        medians     imputation value per feature
        clip_bounds lower / upper clip value per feature
        imputed     number of values filled per feature
        dropped     features removed because they were unusable
    """
    if features is None or len(features) == 0:
        raise ModelingError("No players are left to model. Lower the minimum-matches threshold.")
    if len(features) < MIN_K + 1:
        raise ModelingError(
            f"Only {len(features)} player(s) are left after filtering; at least {MIN_K + 1} "
            "are needed to form clusters. Lower the minimum-matches threshold."
        )

    continuous = [c for c in MODEL_FEATURES if c in features.columns]
    dropped = [c for c in MODEL_FEATURES if c not in features.columns]
    X = features[continuous].astype(float).replace([np.inf, -np.inf], np.nan)

    for col in INDICATOR_FEATURES:
        values = activity.reindex(X.index)[col] if col in activity.columns else None
        if values is None or values.isna().any():
            raise ModelingError(f"Activity indicator '{col}' is missing for some players.")
        X[col] = values.astype(float)

    # Features with no observed values at all cannot be imputed.
    all_missing = [c for c in continuous if X[c].isna().all()]
    X = X.drop(columns=all_missing)
    dropped += all_missing
    continuous = [c for c in continuous if c not in all_missing]

    # 1. Clip to percentiles computed from observed (non-missing) values.
    lower = X[continuous].quantile(CLIP_QUANTILES[0])
    upper = X[continuous].quantile(CLIP_QUANTILES[1])
    X[continuous] = X[continuous].clip(lower=lower, upper=upper, axis=1)

    # 2. Median imputation for undefined rates.
    medians = X[continuous].median()
    imputed = X[continuous].isna().sum().astype(int)
    X[continuous] = X[continuous].fillna(medians)

    # 3. Remove constant columns (no variation cannot separate players).
    constant = [c for c in X.columns if X[c].nunique() <= 1]
    X = X.drop(columns=constant)
    dropped += constant
    if X.shape[1] < 2:
        raise ModelingError(
            f"Only {X.shape[1]} usable feature(s) remain; at least 2 are needed for "
            "clustering and PCA. Try a lower minimum-matches threshold."
        )

    # 4. Standardize.
    scaler = StandardScaler().fit(X.to_numpy())
    X_scaled = pd.DataFrame(scaler.transform(X.to_numpy()), index=X.index, columns=X.columns)
    if not np.isfinite(X_scaled.to_numpy()).all():
        raise ModelingError("Scaling produced non-finite values.")

    return {
        "X": X_scaled,
        "X_processed": X,
        "scaler": scaler,
        "medians": medians.drop(labels=constant, errors="ignore"),
        "clip_bounds": pd.DataFrame({"lower": lower, "upper": upper}).drop(index=constant, errors="ignore"),
        "imputed": imputed.drop(labels=constant, errors="ignore"),
        "dropped": dropped,
    }


# ---------------------------------------------------------------------------
# K-Means and evaluation
# ---------------------------------------------------------------------------

def max_valid_k(X):
    """Largest K that the data supports (at most MAX_K)."""
    distinct = len(np.unique(np.asarray(X), axis=0))
    return min(MAX_K, distinct - 1, len(X) - 1)


def validate_k(k, X):
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ModelingError(f"K must be a whole number, got {k!r}.")
    if not MIN_K <= k <= MAX_K:
        raise ModelingError(f"K must be between {MIN_K} and {MAX_K}, got {k}.")
    limit = max_valid_k(X)
    if k > limit:
        raise ModelingError(
            f"K = {k} is too large for {len(X)} players with these settings; "
            f"the maximum supported K is {limit}."
        )


def make_kmeans(k):
    return KMeans(n_clusters=k, init="k-means++", n_init=N_INIT, max_iter=MAX_ITER,
                  random_state=RANDOM_STATE)


def fit_kmeans(X, k):
    """Fit K-Means and renumber clusters 1..K from largest to smallest.

    Returns a dict with labels (Series indexed by Player), centers (scaled,
    indexed by cluster id), inertia, and the fitted model.
    """
    validate_k(k, X)
    model = make_kmeans(k).fit(X.to_numpy())
    raw_labels = pd.Series(model.labels_, index=X.index)
    sizes = raw_labels.value_counts()
    # Largest cluster first; ties broken by the original label for determinism.
    order = sorted(sizes.index, key=lambda c: (-sizes[c], c))
    mapping = {old: new for new, old in enumerate(order, start=1)}
    labels = raw_labels.map(mapping).rename("cluster")
    if labels.nunique() != k:
        raise ModelingError(f"K-Means produced {labels.nunique()} non-empty clusters instead of {k}.")
    centers = pd.DataFrame(model.cluster_centers_[order], columns=X.columns,
                           index=pd.Index(range(1, k + 1), name="cluster"))
    return {"labels": labels, "centers": centers, "inertia": float(model.inertia_), "model": model}


def silhouette(X, labels):
    """Silhouette score, or None when it is undefined (fewer than 2 clusters, or
    as many clusters as players). Sampled with a fixed seed above SILHOUETTE_SAMPLE_SIZE."""
    labels = np.asarray(labels)
    n_clusters = len(np.unique(labels))
    if n_clusters < 2 or n_clusters > len(labels) - 1:
        return None
    sample = SILHOUETTE_SAMPLE_SIZE if len(labels) > SILHOUETTE_SAMPLE_SIZE else None
    return float(silhouette_score(np.asarray(X), labels, sample_size=sample,
                                  random_state=RANDOM_STATE))


def evaluate_k_range(X, k_values=range(MIN_K, MAX_K + 1)):
    """Inertia and silhouette score for each K. Invalid K values are reported, not fitted."""
    rows = []
    for k in k_values:
        try:
            validate_k(k, X)
        except ModelingError as error:
            rows.append({"k": k, "inertia": np.nan, "silhouette": np.nan, "note": str(error)})
            continue
        fit = fit_kmeans(X, k)
        score = silhouette(X, fit["labels"])
        rows.append({"k": k, "inertia": fit["inertia"],
                     "silhouette": np.nan if score is None else score, "note": ""})
    return pd.DataFrame(rows).set_index("k")


def elbow_heuristic(evaluation):
    """K at the sharpest bend of the inertia curve (largest second difference
    I[k-1] - 2*I[k] + I[k+1]). A heuristic only; None if fewer than 3 valid K."""
    inertia = evaluation["inertia"].dropna()
    ks = list(inertia.index)
    bends = {ks[i]: inertia.iloc[i - 1] - 2 * inertia.iloc[i] + inertia.iloc[i + 1]
             for i in range(1, len(ks) - 1) if ks[i + 1] - ks[i - 1] == 2}
    return max(bends, key=bends.get) if bends else None


def best_silhouette_k(evaluation):
    scores = evaluation["silhouette"].dropna()
    return int(scores.idxmax()) if len(scores) else None


def describe_silhouette(score):
    """Common rule of thumb (Kaufman & Rousseeuw, 1990) for the mean silhouette."""
    if score is None or pd.isna(score):
        return "not defined"
    if score > 0.70:
        return "strong structure"
    if score > 0.50:
        return "reasonable structure"
    if score > 0.25:
        return "weak structure; clusters overlap noticeably"
    return "no substantial structure; clusters overlap heavily"


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def run_pca(X):
    """Two-component PCA of the standardized matrix (for visualization only)."""
    if X.shape[1] < 2 or len(X) < 3:
        raise ModelingError("PCA needs at least 2 features and 3 players.")
    pca = PCA(n_components=2, svd_solver="full").fit(X.to_numpy())
    coords = pd.DataFrame(pca.transform(X.to_numpy()), index=X.index, columns=["PC1", "PC2"])
    loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=["PC1", "PC2"])
    ratio = pca.explained_variance_ratio_
    return {"coords": coords, "loadings": loadings,
            "explained": {"PC1": float(ratio[0]), "PC2": float(ratio[1]), "total": float(ratio.sum())},
            "model": pca}


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def cluster_profiles(features, activity, labels):
    """Typical real values per cluster: player count, median of every engineered
    feature (undefined values skipped) and the share of players active in each
    discipline. These are actual cricket statistics, not scaled values.

    A median based on fewer than MIN_DEFINED_FOR_MEDIAN players is replaced by
    NaN (shown as "too few players") because it would describe one or two people,
    not the cluster. `defined` gives the number of players behind each median.
    """
    df = features.join(activity).assign(cluster=labels)
    groups = df.groupby("cluster")
    feature_cols = list(features.columns)
    defined = groups[feature_cols].count()
    medians = groups[feature_cols].median().where(defined >= MIN_DEFINED_FOR_MEDIAN)
    shares = groups[list(activity.columns)].mean()
    profile = pd.concat([groups.size().rename("players"), medians, shares], axis=1)
    overall = pd.concat([pd.Series({"players": len(df)}),
                         df[feature_cols].median(),
                         df[list(activity.columns)].mean()])
    return profile, overall, defined


def relative_profiles(X, labels):
    """Mean standardized value per cluster: + means above the modeled-player
    average, - below, in standard deviations. Not a real cricket statistic."""
    return X.groupby(labels).mean()


def _score(row, names, signs=None):
    values = [row[n] * (signs or {}).get(n, 1) for n in names if n in row.index]
    return float(np.mean(values)) if values else 0.0


def interpret_clusters(relative, profile):
    """Give each cluster a cautious label using explicit rules on its relative profile.

    batting score  = mean relative value of runs per innings, strike rate, sixes per innings
    bowling score  = mean relative value of wickets per match and has_bowled
    Rules, first match wins (T = LABEL_THRESHOLD, G = DOMINANCE_GAP, both in SD):
      keeper share >= T                                  -> Wicket-Keeping Profile
      batting >= T and bowling >= T                      -> All-Round Statistical Profile
      batting >= T, or batting > 0 and batting - bowling >= G -> Batting-Oriented
      bowling >= T, or bowling > 0 and bowling - batting >= G -> Bowling-Oriented
      catches per match >= T                             -> Fielding-Oriented
      fewer than half of the players bowled              -> Limited Bowling Data
      otherwise                                          -> neutral "Cluster N"
    Batting-Oriented clusters get ", Rarely Bowls" (under half bowled) or
    ", Also Bowls" (at least half bowled), from the actual share of bowlers.
    """
    t, gap = LABEL_THRESHOLD, DOMINANCE_GAP
    rows = []
    for cluster, row in relative.iterrows():
        bat = _score(row, ["bat_runs_per_inns", "bat_strike_rate", "bat_sixes_per_inns"])
        bowl = _score(row, ["bowl_wkts_per_match", "has_bowled"])
        keeper = row.get("field_keeper_share", 0.0)
        catches = row.get("field_catches_per_match", 0.0)
        bowled_share = profile.loc[cluster, "has_bowled"] if "has_bowled" in profile.columns else np.nan

        if keeper >= t:
            name, reason = "Wicket-Keeping Profile", f"keeper share is {keeper:+.1f} SD above average"
        elif bat >= t and bowl >= t:
            name, reason = "All-Round Statistical Profile", f"batting score {bat:+.1f} SD and bowling score {bowl:+.1f} SD are both above average"
        elif bat >= t or (bat > 0 and bat - bowl >= gap):
            name = "Batting-Oriented, " + ("Rarely Bowls" if bowled_share < 0.5 else "Also Bowls")
            reason = (f"batting score {bat:+.1f} SD is at least {t} SD above average" if bat >= t else
                      f"batting score {bat:+.1f} SD is at least {gap:.0f} SD above its bowling score {bowl:+.1f} SD")
        elif bowl >= t or (bowl > 0 and bowl - bat >= gap):
            name = "Bowling-Oriented"
            reason = (f"bowling score {bowl:+.1f} SD is at least {t} SD above average" if bowl >= t else
                      f"bowling score {bowl:+.1f} SD is at least {gap:.0f} SD above its batting score {bat:+.1f} SD")
        elif catches >= t:
            name, reason = "Fielding-Oriented", f"catches per match is {catches:+.1f} SD above average"
        elif bowled_share < 0.5:
            name, reason = "Limited Bowling Data", f"only {bowled_share:.0%} of these players have bowled"
        else:
            name, reason = f"Cluster {cluster}", "no rule matched clearly, so a neutral name is used"

        # Largest positive and negative relative values, with real medians.
        ordered = row.sort_values()
        def describe(feature):
            real = profile.loc[cluster, feature] if feature in profile.columns else np.nan
            real_text = "n/a" if pd.isna(real) else f"{real:.2f}"
            return f"{label(feature)} ({row[feature]:+.1f} SD, median {real_text})"
        explanation = (f"Labelled because {reason}. Highest: {describe(ordered.index[-1])}. "
                       f"Lowest: {describe(ordered.index[0])}. "
                       f"{bowled_share:.0%} of players have bowled.")
        rows.append({"cluster": cluster, "label": name, "batting_score": bat,
                     "bowling_score": bowl, "explanation": explanation})

    result = pd.DataFrame(rows).set_index("cluster")
    # Keep names unique: repeat labels get the cluster number appended.
    repeated = result["label"].duplicated(keep=False)
    result.loc[repeated, "label"] = [f"{name} ({c})" for c, name in result.loc[repeated, "label"].items()]
    return result


def centroid_distances(X, centers, player):
    """Euclidean distance from one player to every cluster centre (standardized space)."""
    point = X.loc[player].to_numpy()
    return pd.Series(np.linalg.norm(centers.to_numpy() - point, axis=1), index=centers.index,
                     name="distance").sort_values()


def most_similar_players(X, player, n=5):
    """Players with the smallest Euclidean distance to `player` in the standardized
    model space (an actual nearest-neighbour calculation)."""
    distances = pd.Series(np.linalg.norm(X.to_numpy() - X.loc[player].to_numpy(), axis=1),
                          index=X.index, name="distance").drop(player)
    return distances.nsmallest(n)


def results_table(features, activity, labels, coords, interpretation):
    """One row per modeled player: identifier, cluster, label, PCA coordinates and
    the engineered features (real values; undefined rates left blank)."""
    table = pd.DataFrame({
        "cluster_id": labels,
        "cluster_label": labels.map(interpretation["label"]),
        "PC1": coords["PC1"],
        "PC2": coords["PC2"],
    }).join(features).join(activity)
    table.index.name = "Player"
    return table.reset_index()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(merged, min_matches, k):
    """Run preprocessing and the full model for one setting of (min_matches, K)."""
    features = prepare_features(merged, min_matches=min_matches)
    if len(features) == 0:
        raise ModelingError(f"No players have at least {min_matches} matches.")
    activity = activity_indicators(merged, features.index)
    matrix = build_model_matrix(features, activity)
    X = matrix["X"]
    fit = fit_kmeans(X, k)
    pca = run_pca(X)
    profile, overall, defined = cluster_profiles(features, activity, fit["labels"])
    relative = relative_profiles(X, fit["labels"])
    interpretation = interpret_clusters(relative, profile)
    return {
        "min_matches": min_matches,
        "k": k,
        "features": features,
        "activity": activity,
        "matrix": matrix,
        "labels": fit["labels"],
        "centers": fit["centers"],
        "inertia": fit["inertia"],
        "silhouette": silhouette(X, fit["labels"]),
        "pca": pca,
        "profile": profile,
        "overall": overall,
        "defined": defined,
        "relative": relative,
        "interpretation": interpretation,
        "results": results_table(features, activity, fit["labels"], pca["coords"], interpretation),
        "settings": model_settings(k),
    }


def model_settings(k):
    return {
        "K (clusters)": k,
        "random_state": RANDOM_STATE,
        "n_init": N_INIT,
        "max_iter": MAX_ITER,
        "init": "k-means++",
        "Imputation": "median (fitted on modeled players)",
        "Clipping": f"{CLIP_QUANTILES[0]:.0%}-{CLIP_QUANTILES[1]:.0%} percentiles",
        "Scaling": "StandardScaler",
        "Silhouette sample": f"all players up to {SILHOUETTE_SAMPLE_SIZE}, then a seeded sample",
        "PCA": "2 components, full SVD",
    }

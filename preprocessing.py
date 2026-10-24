"""Turn the merged T20 table into a numeric feature table for clustering.

Usage:
    from data_loader import load_data
    from preprocessing import prepare_features, get_preprocessing_report

    merged = load_data()["merged"]
    features = prepare_features(merged, min_matches=5)
    report = get_preprocessing_report(merged, features, min_matches=5)

Missing-value policy (Phase 2):
- Count statistics (runs, fours, sixes, wickets, catches, stumpings, ...) are set
  to 0 ONLY when the raw data shows the player did not take part in that
  discipline, i.e. its innings column is "-" (NaN after loading). A missing count
  for a player who did take part is left as NaN.
- Rate statistics (averages, strike rates, economy, keeper share) are computed
  from the counts. When the denominator is 0 or missing (no innings, no balls,
  no dismissals, no wickets) the rate is undefined and stays NaN. It is never set
  to 0, because an undefined average is not a poor average. Phase 3 decides how to
  impute these.
- Divisions never produce infinity: a zero denominator gives NaN.

No scaling, clustering, or PCA happens here.
"""

import numpy as np
import pandas as pd

from data_loader import overs_to_balls

DEFAULT_MIN_MATCHES = 5

# Merged columns (from data_loader.load_data) that the features are built from.
MATCH_COLUMNS = ["Mat_Bat", "Mat_Bowl", "Mat_Field"]
BATTING_COUNTS = ["Inns_Bat", "NO_Bat", "Runs_Bat", "BF_Bat", "4s_Bat", "6s_Bat"]
BOWLING_COUNTS = ["Inns_Bowl", "Runs_Bowl", "Wkts_Bowl"]
FIELDING_COUNTS = ["Inns_Field", "Dis_Field", "Ct_Field", "St_Field", "Ct Wk_Field"]
REQUIRED_COLUMNS = (["Player", "Overs_Bowl"] + MATCH_COLUMNS
                    + BATTING_COUNTS + BOWLING_COUNTS + FIELDING_COUNTS)

# Every engineered feature: (category, definition). The prefix matches the category.
FEATURE_DEFINITIONS = {
    "bat_runs_per_inns": ("batting", "Runs_Bat / Inns_Bat (innings batted)"),
    "bat_strike_rate": ("batting", "100 * Runs_Bat / BF_Bat (balls faced)"),
    "bat_fours_per_inns": ("batting", "4s_Bat / Inns_Bat"),
    "bat_sixes_per_inns": ("batting", "6s_Bat / Inns_Bat"),
    "bat_average": ("batting", "Runs_Bat / (Inns_Bat - NO_Bat), i.e. runs per dismissal"),
    "bowl_wkts_per_match": ("bowling", "Wkts_Bowl / matches"),
    "bowl_economy": ("bowling", "Runs_Bowl / (balls bowled / 6), i.e. runs per over"),
    "bowl_average": ("bowling", "Runs_Bowl / Wkts_Bowl"),
    "bowl_strike_rate": ("bowling", "balls bowled / Wkts_Bowl"),
    "field_catches_per_match": ("fielding", "Ct_Field / matches (includes keeper catches)"),
    "field_stumpings_per_match": ("fielding", "St_Field / matches"),
    "field_keeper_share": ("fielding", "(Ct Wk_Field + St_Field) / Dis_Field, share of dismissals made as keeper"),
}

# Calculated but deliberately left out of the feature table: Dis = Ct + St for
# every player, so it is exactly catches/match + stumpings/match and would only
# double-count wicket-keepers.
EXCLUDED_FEATURES = {
    "field_dismissals_per_match": "Dis_Field / matches. Excluded: identical to "
                                  "field_catches_per_match + field_stumpings_per_match.",
}

CATEGORIES = ["batting", "bowling", "fielding"]


class PreprocessingError(Exception):
    """Raised when the input cannot be turned into features."""


def safe_divide(numerator, denominator):
    """numerator / denominator, but NaN (never infinity) when the denominator is 0 or missing."""
    numerator = pd.to_numeric(numerator)
    denominator = pd.to_numeric(denominator)
    return (numerator / denominator.where(denominator > 0)).astype(float)


def _check_input(data):
    if not isinstance(data, pd.DataFrame):
        raise PreprocessingError("Expected the merged DataFrame returned by load_data()['merged'].")
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise PreprocessingError(f"Missing required column(s): {missing}")
    if data["Player"].isna().any() or (data["Player"].astype(str).str.strip() == "").any():
        raise PreprocessingError("Some rows have an empty Player identifier.")
    duplicates = data.loc[data["Player"].duplicated(), "Player"]
    if not duplicates.empty:
        raise PreprocessingError(f"Duplicate Player identifiers: {duplicates.head(5).tolist()}")


def _to_numeric_columns(df, columns):
    """Convert columns to numbers. Unparseable text raises an error; infinity becomes NaN."""
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce")
        bad = df.loc[values.isna() & df[col].notna(), col]
        if not bad.empty:
            raise PreprocessingError(
                f"Column '{col}' has non-numeric value(s), e.g. {bad.unique()[:5].tolist()}"
            )
        df[col] = values.replace([np.inf, -np.inf], np.nan)
    return df


def matches_played(data):
    """Matches per player, after checking Mat agrees across the three datasets."""
    mat = _to_numeric_columns(data[MATCH_COLUMNS].copy(), MATCH_COLUMNS)
    if mat.isna().any().any():
        raise PreprocessingError("Match count (Mat) is missing for some players.")
    disagree = ~(mat["Mat_Bat"].eq(mat["Mat_Bowl"]) & mat["Mat_Bowl"].eq(mat["Mat_Field"]))
    if disagree.any():
        raise PreprocessingError(
            f"Mat differs between batting, bowling and fielding for {int(disagree.sum())} "
            f"player(s), e.g. {data.loc[disagree, 'Player'].head(5).tolist()}"
        )
    return mat["Mat_Bat"]


def _zero_counts_if_did_not_play(df, innings_col, count_cols):
    """Set missing counts to 0 only for players whose innings column is missing
    (raw "-"), which in the source means they never batted / bowled / fielded."""
    did_not_play = df[innings_col].isna()
    for col in count_cols:
        df.loc[did_not_play & df[col].isna(), col] = 0
    return df


def prepare_features(data, min_matches=DEFAULT_MIN_MATCHES):
    """Build the numeric feature table, one row per player.

    data: the merged DataFrame from load_data()["merged"]. It is not modified.
    min_matches: keep players with at least this many T20 matches. 5 is a
        practical default to avoid profiles built on one or two games; it is not
        a statistically optimal value and can be changed.

    Returns a DataFrame indexed by the full Player string (country tag included),
    containing only numeric feature columns. Undefined rates are NaN.
    """
    _check_input(data)
    if not isinstance(min_matches, (int, np.integer)) or isinstance(min_matches, bool) or min_matches < 0:
        raise PreprocessingError(f"min_matches must be a whole number >= 0, got {min_matches!r}")

    df = data[REQUIRED_COLUMNS].copy()  # work on a copy; the input is never mutated
    count_columns = BATTING_COUNTS + BOWLING_COUNTS + FIELDING_COUNTS
    df = _to_numeric_columns(df, count_columns)
    df["matches"] = matches_played(data)

    # Overs use cricket notation, so convert to real balls (10.5 -> 65 balls).
    df["balls_bowled"] = overs_to_balls(df["Overs_Bowl"]).to_numpy()

    # Minimum-matches filter.
    df = df[df["matches"] >= min_matches].copy()

    df = _zero_counts_if_did_not_play(df, "Inns_Bat", BATTING_COUNTS)
    df = _zero_counts_if_did_not_play(df, "Inns_Bowl", BOWLING_COUNTS + ["balls_bowled"])
    df = _zero_counts_if_did_not_play(df, "Inns_Field", FIELDING_COUNTS)

    dismissed = df["Inns_Bat"] - df["NO_Bat"]
    keeper_dismissals = df["Ct Wk_Field"] + df["St_Field"]

    features = pd.DataFrame(index=pd.Index(df["Player"], name="Player"))
    values = {
        "bat_runs_per_inns": safe_divide(df["Runs_Bat"], df["Inns_Bat"]),
        "bat_strike_rate": 100 * safe_divide(df["Runs_Bat"], df["BF_Bat"]),
        "bat_fours_per_inns": safe_divide(df["4s_Bat"], df["Inns_Bat"]),
        "bat_sixes_per_inns": safe_divide(df["6s_Bat"], df["Inns_Bat"]),
        "bat_average": safe_divide(df["Runs_Bat"], dismissed),
        "bowl_wkts_per_match": safe_divide(df["Wkts_Bowl"], df["matches"]),
        "bowl_economy": safe_divide(df["Runs_Bowl"], df["balls_bowled"] / 6),
        "bowl_average": safe_divide(df["Runs_Bowl"], df["Wkts_Bowl"]),
        "bowl_strike_rate": safe_divide(df["balls_bowled"], df["Wkts_Bowl"]),
        "field_catches_per_match": safe_divide(df["Ct_Field"], df["matches"]),
        "field_stumpings_per_match": safe_divide(df["St_Field"], df["matches"]),
        "field_keeper_share": safe_divide(keeper_dismissals, df["Dis_Field"]),
    }
    for name, series in values.items():
        features[name] = series.to_numpy()

    # Drop unusable features: entirely missing, or a single constant value
    # (no variation, so it cannot separate players).
    unusable = [c for c in features.columns if features[c].nunique(dropna=True) <= 1]
    return features.drop(columns=unusable)


def feature_category(name):
    return FEATURE_DEFINITIONS[name][0]


def get_preprocessing_report(data, features, min_matches=DEFAULT_MIN_MATCHES):
    """Summarise what prepare_features did, using the actual data and output."""
    matches = matches_played(data)
    initial = len(data)
    kept = matches >= min_matches
    passed_filter = int(kept.sum())

    def did_not_play(innings_col):
        # Raw "-" in the innings column: never batted / bowled / fielded.
        innings = pd.to_numeric(data.loc[kept, innings_col], errors="coerce")
        return int(innings.isna().sum())

    by_category = {cat: [c for c in features.columns if feature_category(c) == cat]
                   for cat in CATEGORIES}

    def rows_without(category):
        cols = by_category[category]
        if not cols:
            return len(features)
        return int(features[cols].isna().all(axis=1).sum())

    return {
        "min_matches": min_matches,
        "initial_players": initial,
        "excluded_players": initial - passed_filter,
        "remaining_players": len(features),
        "feature_count": features.shape[1],
        "features_by_category": by_category,
        "categories_represented": [cat for cat in CATEGORIES if by_category[cat]],
        "dropped_features": [f for f in FEATURE_DEFINITIONS if f not in features.columns],
        "missing_per_feature": features.isna().sum().astype(int).to_dict(),
        # Rows where every feature of the category is NaN.
        "rows_no_batting": rows_without("batting"),
        "rows_no_bowling": rows_without("bowling"),
        "rows_no_fielding": rows_without("fielding"),
        # Remaining players who never took part in the discipline. Their per-match
        # counts are 0 (e.g. wickets per match), but all their rates are NaN.
        "never_batted": did_not_play("Inns_Bat"),
        "never_bowled": did_not_play("Inns_Bowl"),
        "never_fielded": did_not_play("Inns_Field"),
    }

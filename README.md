# Cricket X-Ray

**Discovering Cricket Player Performance Profiles Using Unsupervised Machine Learning**

Third-semester Statistical Machine Learning project (B.Tech CSE).

## Objective

Group cricket players into performance profiles (for example, specialist batters,
bowlers, all-rounders, wicket-keepers) using their batting, bowling, and fielding
statistics, without using any predefined labels.

## Current phase

**Phase 2: data preprocessing and feature engineering.** (Phases 0 and 1 are complete.)

| Status | Feature |
|---|---|
| Implemented | Project structure, virtual environment, dependencies |
| Implemented | Streamlit setup-check page (`app.py`) |
| Implemented | Data loading and validation (`data_loader.py`) and a Data Overview section in the app |
| Implemented | Data preprocessing and feature engineering (`preprocessing.py`) and a Feature Engineering Preview in the app |
| Planned | Feature scaling |
| Planned | K-Means clustering |
| Planned | PCA visualization |
| Planned | Cluster evaluation |
| Planned | Player exploration |

No clustering, PCA, or model evaluation has been implemented yet.

## Planned methodology

1. Merge the batting, bowling, and fielding tables on player name.
2. Clean the data and derive per-match / per-innings features.
3. Standardize features (`StandardScaler`).
4. Cluster players with K-Means and choose *k* using the elbow method and silhouette score.
5. Project the data to 2D with PCA to visualize the clusters.
6. Interpret each cluster from its average statistics.

## Project structure

```
CricketXray/
├── app.py               # Streamlit app (setup check, data overview, feature preview)
├── data_loader.py       # load_data(): read, validate, clean, and merge the CSVs
├── preprocessing.py     # prepare_features(): build the numeric feature table
├── tests/
│   └── test_preprocessing.py  # unittest tests (no extra packages needed)
├── requirements.txt     # Direct dependencies
├── README.md
├── data/                # CSV datasets (not committed to Git)
│   ├── Batting/         # t20.csv, ODI data.csv, test.csv
│   ├── Bowling/         # Bowling_t20.csv, Bowling_ODI.csv, Bowling_test.csv
│   └── Fielding/        # Fielding_t20.csv, Fielding_ODI.csv, Fielding_test.csv
├── notebooks/
│   └── Cricket_Xray.ipynb  # Original exploration notebook
└── outputs/             # Generated results (not committed to Git)
```

## Prerequisites

- Python 3.11 or 3.12 (the project was set up with Python 3.11.13)

## Setup

### macOS / Linux

```bash
cd CricketXray
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
cd CricketXray
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use
Command Prompt with `.venv\Scripts\activate.bat`.

## Run the app

With the virtual environment activated:

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser. Stop the server with `Ctrl+C`.

## Run the tests

With the virtual environment activated, from the project folder:

```bash
python -m unittest discover -s tests -v
```

The tests use Python's built-in `unittest`. Most use small hand-made tables; two
run against the real T20 files and are skipped if those files are missing.

## Datasets

The datasets are in subfolders of `data/`. This project uses the **T20** files,
the same ones the original notebook used:

| Dataset | File | Rows | Raw columns |
|---|---|---|---|
| Batting | `data/Batting/t20.csv` | 2006 | 17 |
| Bowling | `data/Bowling/Bowling_t20.csv` | 2006 | 16 |
| Fielding | `data/Fielding/Fielding_t20.csv` | 2006 | 13 |

The ODI and Test files are present but are not used.
CSV files are excluded from Git by `.gitignore`.

## Loading the data

```python
from data_loader import load_data

data = load_data()
merged = data["merged"]   # one row per player, 2006 x 44
```

`load_data()` returns a dict with `raw` (the original CSV text, unchanged),
`clean` (numbers converted), `merged`, `overlap`, and `missing`. If a file is
missing, a required column is missing, a value cannot be parsed, a player appears
twice, or the merge fails, it raises `DataValidationError` with a readable message.

How values are handled:

| Source value | Meaning | Loaded as |
|---|---|---|
| `-` | Did not bat/bowl, or undefined (e.g. average with 0 dismissals) | `NaN`, **not** 0 (filling is a Phase 2 decision) |
| `HS` = `94*` | 94, not out | `HS` text kept; `HS_Runs` = 94, `HS_NotOut` = True |
| `BBI` = `5/6` | 5 wickets for 6 runs | `BBI` text kept; `BBI_Wkts` = 5, `BBI_Runs` = 6 |
| `Overs` = `284.5` | 284 overs and 5 balls (cricket notation, not a decimal) | `Overs` text kept; `Balls` = 1709 |
| `MD` = `5 (5ct 0st)` | Most dismissals in one innings | `MD` text kept; `MD_Dis` = 5 |
| `Unnamed: ...` columns | Scraper page index (repeats 0 to 49) and an empty trailing column | Dropped |

Merged column names get a suffix per dataset (`Runs_Bat`, `Runs_Bowl`, `Mat_Field`, ...).
The merge uses the full `Player` string, including the country tag, because 14
names repeat once the tag is removed.

## Verified dataset findings (T20)

All of these were checked against the real files:

- All three files have 2006 rows and 0 duplicate players. Every player appears in all
  three files, so the merge keeps all 2006 players.
- The notebook's merge logic gives (2006, 38). `load_data()` gives (2006, 44): the
  same 38 columns plus 6 parsed number columns.
- `Span` and `Mat` are identical across the three files for every player.
- Fielding: `Dis = Ct + St` and `Ct = Ct Wk + Ct Fi` hold for every row.
- Converting `Overs` to balls is consistent with the file's economy rate
  (`Runs / (Balls / 6)` is within 0.01 of `Econ` for all 1,355 bowlers).
- Missing values (from `-`):
  - Batting: 152 players never batted, so 152 values are missing in each batting
    stat. `Ave` has 328 (176 more were never dismissed), and `SR` has 175 (23 more
    faced 0 balls).
  - Bowling: 651 players never bowled. `Ave`, `SR`, and `BBI` have 865 (214 more
    bowled but took no wickets).
  - Fielding: 2 players have no fielding innings.
  - 30 players neither batted nor bowled.

## Preprocessing and features

```python
from data_loader import load_data
from preprocessing import prepare_features, get_preprocessing_report

merged = load_data()["merged"]
features = prepare_features(merged, min_matches=5)   # 1105 players x 12 features
report = get_preprocessing_report(merged, features, min_matches=5)
```

`prepare_features()` works on a copy (the input is never changed) and returns only
numeric feature columns. The rows are indexed by the full `Player` string, including
the country tag. The player name is never used as a feature.

### Minimum-matches filter

Players with fewer than `min_matches` T20 matches are removed (default **5**). This is
a practical threshold to avoid profiles built on one or two games, not a
statistically optimal value. It can be changed in code or with the slider in the app.
`Mat` is checked to be identical in the batting, bowling and fielding files first.

| `min_matches` | Players kept (of 2006) |
|---|---|
| 1 | 2006 |
| 5 (default) | 1105 (901 excluded) |
| 10 | 662 |

### Features

Rates are calculated from the count columns rather than taken from the file's
rounded rate columns. The tests confirm they match the file's `Ave`, `SR` and `Econ`
columns within rounding. They are also missing for exactly the same players as the
file's `-` values.

| Feature | Category | Definition |
|---|---|---|
| `bat_runs_per_inns` | batting | `Runs_Bat / Inns_Bat` (innings batted, not matches) |
| `bat_strike_rate` | batting | `100 * Runs_Bat / BF_Bat` (balls faced) |
| `bat_fours_per_inns` | batting | `4s_Bat / Inns_Bat` |
| `bat_sixes_per_inns` | batting | `6s_Bat / Inns_Bat` |
| `bat_average` | batting | `Runs_Bat / (Inns_Bat - NO_Bat)`, runs per dismissal |
| `bowl_wkts_per_match` | bowling | `Wkts_Bowl / matches` |
| `bowl_economy` | bowling | `Runs_Bowl / (balls bowled / 6)` |
| `bowl_average` | bowling | `Runs_Bowl / Wkts_Bowl` |
| `bowl_strike_rate` | bowling | `balls bowled / Wkts_Bowl` |
| `field_catches_per_match` | fielding | `Ct_Field / matches` (includes keeper catches) |
| `field_stumpings_per_match` | fielding | `St_Field / matches` |
| `field_keeper_share` | fielding | `(Ct Wk_Field + St_Field) / Dis_Field`, share of dismissals made as keeper |

**Balls bowled** come from `Overs` using cricket notation:
`whole_overs * 6 + balls_component`. For example, 10.0 → 60, 10.1 → 61, 10.5 → 65.
The function is `overs_to_balls()` in `data_loader.py`, and the loader and
preprocessing both use it.

**Not included:** `field_dismissals_per_match` (`Dis_Field / matches`). In this data
`Dis = Ct + St` for every player, so it equals catches per match + stumpings per
match exactly. It would add no information and would count wicket-keepers twice.

Features that are entirely missing or have a single constant value are dropped
automatically. With the real data at the default threshold, none are dropped.

### Missing-value policy

- **Counts** (runs, fours, sixes, wickets, catches, stumpings, dismissals) are set to
  **0 only** for players whose innings column is `-`, which in the source means they
  never batted, bowled or fielded. Any other missing count stays missing.
- **Rates** (averages, strike rates, economy, keeper share) stay **NaN** when undefined:
  no innings, 0 balls faced, never dismissed, never bowled, 0 wickets, or 0 dismissals.
  They are **not** set to 0, because an undefined average is not a poor average.
  Imputation is decided in Phase 3.
- Division by zero never produces infinity, and infinite inputs are treated as missing.

Missing values at the default threshold (1105 players): 10 players never batted and
316 never bowled; every player fielded.

| Feature | Missing |
|---|---|
| `bat_runs_per_inns`, `bat_fours_per_inns`, `bat_sixes_per_inns` | 10 |
| `bat_strike_rate` | 16 (also 0 balls faced) |
| `bat_average` | 64 (also never dismissed) |
| `bowl_economy` | 316 |
| `bowl_average`, `bowl_strike_rate` | 366 (also 0 wickets) |
| `field_keeper_share` | 107 (0 dismissals) |
| `bowl_wkts_per_match`, `field_catches_per_match`, `field_stumpings_per_match` | 0 |

### Known limitations

- The bowling rates are missing for about a third of players, so Phase 3 needs an
  explicit strategy before K-Means, which cannot use missing values.
- Some features overlap. Catches per match includes keeper catches, which also
  feed keeper share. Runs per innings, batting average and strike rate are related.
- Small samples still produce extreme values even with 5 matches. Examples: a
  batting average of 115, a bowling average of 158, and an economy rate of 24.
- Career totals cannot separate batting position or era. `Span` is not used as a
  feature.

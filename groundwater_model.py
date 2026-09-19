"""
Groundwater Level Forecasting Model — single-file version
============================================================

Everything from the multi-file project (config, data_pipeline, features,
train_model, predict) combined into one script, so it can be dropped
straight into a PyCharm project folder.

INPUT SCHEMA (matches the CGWB "changes in depth to water level" dataset):
    date, state_name, state_code, district_name, district_code,
    station_name, latitude, longitude, basin, sub_basin, source

OUTPUT:
    current_depth_level_m  -> predicted depth to water level (m below ground)
    level_difference_m     -> predicted change vs. the CGWB reference period
    risk_flag              -> Critical / Watch / Normal (bonus classification)

SETUP
-----
1. Put the raw CSV next to this file (or update RAW_CSV_PATH below):
     cgwb-changes-in-depth-to-water-level.csv
2. pip install pandas numpy scikit-learn joblib
3. Run this file directly:
     python groundwater_model.py
   First run: cleans the data, trains both models, saves them, then runs a
   sample prediction. Later runs: if the trained models already exist on
   disk, it skips straight to loading them and predicting (see `main()` at
   the bottom) — delete the `models/` folder to force a full retrain.

USAGE FROM YOUR OWN CODE
-------------------------
    from groundwater_model import predict_one

    result = predict_one({
        "date": "2024-06-01",
        "state_name": "Karnataka",
        "state_code": 29,
        "district_name": "Bengaluru Urban",
        "district_code": 601,
        "station_name": "Some Station",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "basin": "Cauvery Basin",
        "sub_basin": "Cauvery",
        "source": "CGWB",
    })
    print(result)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import TargetEncoder

# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

RAW_CSV_PATH = DATA_DIR / "cgwb-changes-in-depth-to-water-level.csv"
CLEAN_CSV_PATH = DATA_DIR / "gwl_clean.csv"
STATION_LOOKUP_PATH = DATA_DIR / "station_lookup.csv"

MODEL_CURRENTLEVEL_PATH = MODELS_DIR / "model_currentlevel.joblib"
MODEL_LEVELDIFF_PATH = MODELS_DIR / "model_level_diff.joblib"
ENCODERS_PATH = MODELS_DIR / "encoders.joblib"

for d in (DATA_DIR, MODELS_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

INPUT_COLUMNS = [
    "date", "state_name", "state_code", "district_name", "district_code",
    "station_name", "latitude", "longitude", "basin", "sub_basin", "source",
]

CATEGORICAL_COLS = ["state_name", "district_name", "basin", "sub_basin", "source"]
NUMERIC_COLS = [
    "latitude", "longitude",
    "month_sin", "month_cos",
    "gap_days",
    "prev_currentlevel", "prev_level_diff",
    "prev2_currentlevel", "prev2_level_diff",
    "trend_currentlevel", "has_prev2",
    "hist_mean_currentlevel", "hist_std_currentlevel", "n_readings_so_far",
]

# A small hyperparameter grid searched per model at training time (see
# train_models) — kept short enough to still finish on a single laptop.
# loss="absolute_error" trains to directly minimize MAE (matches the
# tolerance-based accuracy metric we report) instead of squared error.
PARAM_GRID = [
    dict(max_iter=300, max_depth=6, learning_rate=0.06, l2_regularization=0.1, random_state=42),
    dict(max_iter=400, max_depth=8, learning_rate=0.05, l2_regularization=0.1, random_state=42),
    dict(max_iter=500, max_depth=None, learning_rate=0.04, l2_regularization=0.2, max_leaf_nodes=63, random_state=42),
    dict(max_iter=400, max_depth=10, learning_rate=0.03, l2_regularization=0.3, random_state=42),
    dict(max_iter=500, max_depth=None, learning_rate=0.04, l2_regularization=0.2, max_leaf_nodes=63,
         loss="absolute_error", random_state=42),
    dict(max_iter=600, max_depth=8, learning_rate=0.03, l2_regularization=0.15, max_leaf_nodes=127,
         loss="absolute_error", random_state=42),
]


# ============================================================================
# STEP 1 — DATA PIPELINE: load & clean the raw CSV
# ============================================================================

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    text_cols = ["state_name", "district_name", "station_name", "basin", "sub_basin", "source"]
    for c in text_cols:
        df[c] = df[c].astype(str).str.strip()

    num_cols = ["state_code", "district_code", "latitude", "longitude", "currentlevel", "level_diff"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=num_cols)

    # Sanity filters: plausible depth range, and lat/long inside India
    df = df[(df["currentlevel"] >= 0) & (df["currentlevel"] <= 300)]
    df = df[df["level_diff"].abs() <= 300]
    df = df[df["latitude"].between(6, 38) & df["longitude"].between(68, 98)]

    df = df.drop_duplicates()
    df = df.sort_values(["station_name", "date"]).reset_index(drop=True)
    return df


def run_data_pipeline() -> pd.DataFrame:
    print(f"Loading raw data from {RAW_CSV_PATH} ...")
    df = load_raw()
    print(f"Raw shape: {df.shape}")

    df_clean = clean(df)
    print(f"Clean shape: {df_clean.shape}")
    print(f"Stations: {df_clean['station_name'].nunique()}  States: {df_clean['state_name'].nunique()}")
    print(f"Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")

    df_clean.to_csv(CLEAN_CSV_PATH, index=False)
    print(f"Saved cleaned data to {CLEAN_CSV_PATH}")

    # Per-station "latest known readings" lookup (last + 2nd-last reading),
    # used at prediction time to supply both the prev_* lag features and the
    # prev2_*/trend_currentlevel features so the model sees a short history.
    def _last_two(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("date")
        last = g.iloc[-1]
        out = {
            "state_name": last["state_name"],
            "state_code": int(last["state_code"]),
            "district_name": last["district_name"],
            "district_code": int(last["district_code"]),
            "basin": last["basin"],
            "sub_basin": last["sub_basin"],
            "latitude": last["latitude"],
            "longitude": last["longitude"],
            "last_date": last["date"],
            "last_currentlevel": last["currentlevel"],
            "last_level_diff": last["level_diff"],
        }
        if len(g) >= 2:
            prev2 = g.iloc[-2]
            out["has_prev2"] = 1
            out["prev2_currentlevel"] = prev2["currentlevel"]
            out["prev2_level_diff"] = prev2["level_diff"]
        else:
            out["has_prev2"] = 0
            out["prev2_currentlevel"] = 0.0
            out["prev2_level_diff"] = 0.0

        levels = g["currentlevel"].to_numpy()
        out["hist_mean_currentlevel"] = float(np.mean(levels))
        out["hist_std_currentlevel"] = float(np.std(levels)) if len(levels) > 1 else 0.0
        out["n_readings_so_far"] = int(len(levels))
        return pd.Series(out)

    latest = (
        df_clean.groupby("station_name")
        .apply(_last_two, include_groups=False)
        .reset_index()
    )
    latest.to_csv(STATION_LOOKUP_PATH, index=False)
    print(f"Saved station lookup table to {STATION_LOOKUP_PATH}")

    return df_clean


# ============================================================================
# STEP 2 — FEATURE ENGINEERING: lag features + target encoding
# ============================================================================

def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    return df


def build_supervised_dataset(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Each consecutive pair of readings at a station becomes one training row:
    (station's context + recent readings) -> (next reading). When a station
    has a reading before `cur` too, it feeds prev2_*/trend_currentlevel;
    otherwise those default to 0 and has_prev2 flags it. Month is encoded
    cyclically (sin/cos) so December and January aren't numerically far apart.
    """
    df = add_time_parts(df_clean)

    records = []
    for station, g in df.groupby("station_name"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < 2:
            continue

        levels_so_far = []  # expanding history, up to and including `cur`
        for i in range(len(g) - 1):
            cur = g.iloc[i]
            nxt = g.iloc[i + 1]
            levels_so_far.append(cur["currentlevel"])
            gap_days = (nxt["date"] - cur["date"]).days
            if gap_days <= 0:
                continue

            has_prev2 = i - 1 >= 0
            if has_prev2:
                prev2 = g.iloc[i - 1]
                prev2_currentlevel = prev2["currentlevel"]
                prev2_level_diff = prev2["level_diff"]
                trend_currentlevel = cur["currentlevel"] - prev2["currentlevel"]
            else:
                prev2_currentlevel = 0.0
                prev2_level_diff = 0.0
                trend_currentlevel = 0.0

            month = int(nxt["month"])
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            hist_mean_currentlevel = float(np.mean(levels_so_far))
            hist_std_currentlevel = float(np.std(levels_so_far)) if len(levels_so_far) > 1 else 0.0
            n_readings_so_far = len(levels_so_far)

            records.append({
                "station_name": station,
                "state_name": cur["state_name"],
                "district_name": cur["district_name"],
                "basin": cur["basin"],
                "sub_basin": cur["sub_basin"],
                "source": cur["source"],
                "latitude": cur["latitude"],
                "longitude": cur["longitude"],
                "month_sin": month_sin,
                "month_cos": month_cos,
                "gap_days": gap_days,
                "prev_currentlevel": cur["currentlevel"],
                "prev_level_diff": cur["level_diff"],
                "prev2_currentlevel": prev2_currentlevel,
                "prev2_level_diff": prev2_level_diff,
                "trend_currentlevel": trend_currentlevel,
                "has_prev2": 1 if has_prev2 else 0,
                "hist_mean_currentlevel": hist_mean_currentlevel,
                "hist_std_currentlevel": hist_std_currentlevel,
                "n_readings_so_far": n_readings_so_far,
                "target_currentlevel": nxt["currentlevel"],
                "target_level_diff": nxt["level_diff"],
                "date_to": nxt["date"],
            })

    return pd.DataFrame(records)


def fit_encoders(sup: pd.DataFrame):
    """Fit one TargetEncoder per categorical column against target_currentlevel."""
    encoders = {}
    for col in CATEGORICAL_COLS:
        # Note: sklearn 1.9+ deprecates shuffle/random_state on TargetEncoder
        # in favor of a `cv` generator — leave both at their defaults so this
        # runs warning-free on old and new sklearn alike.
        te = TargetEncoder(target_type="continuous")
        sup[f"{col}_enc"] = te.fit_transform(sup[[col]], sup["target_currentlevel"]).ravel()
        encoders[col] = te
    return sup, encoders


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col, te in encoders.items():
        df[f"{col}_enc"] = te.transform(df[[col]]).ravel()
    return df


def feature_columns():
    return [f"{c}_enc" for c in CATEGORICAL_COLS] + NUMERIC_COLS


# ============================================================================
# STEP 3 — TRAIN: two HistGradientBoostingRegressor models
# ============================================================================

def accuracy_within(y_true, y_pred, tolerance: float) -> float:
    """% of predictions within `tolerance` metres of the actual value."""
    errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return float((errors <= tolerance).mean() * 100)


def _search_best(X_train, y_train, X_test, y_test, label: str):
    """Try each PARAM_GRID entry, return (model, params, mae, r2, preds) for the best-MAE one."""
    best = None
    print(f"\nSearching hyperparameters for {label} ...")
    for params in PARAM_GRID:
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        print(f"  {params} -> MAE={mae:.3f}  R2={r2:.3f}")
        if best is None or mae < best[2]:
            best = (model, params, mae, r2, pred)
    print(f"  Best for {label}: {best[1]}  (MAE={best[2]:.3f}, R2={best[3]:.3f})")
    return best


def train_models(df_clean: pd.DataFrame):
    print("Building supervised (lag) dataset ...")
    sup = build_supervised_dataset(df_clean)
    print(f"Supervised rows: {len(sup)}  (with a 2nd lag step: {int(sup['has_prev2'].sum())})")

    print("Fitting target encoders ...")
    sup, encoders = fit_encoders(sup)

    feat_cols = feature_columns()
    X = sup[feat_cols]
    y_level = sup["target_currentlevel"]
    y_diff = sup["target_level_diff"]

    # Time-respecting split: hold out each station's most recent transition
    sup["is_latest"] = sup.groupby("station_name")["date_to"].transform("max") == sup["date_to"]
    train_mask = ~sup["is_latest"]

    X_train, X_test = X[train_mask], X[~train_mask]
    y_level_train, y_level_test = y_level[train_mask], y_level[~train_mask]
    y_diff_train, y_diff_test = y_diff[train_mask], y_diff[~train_mask]

    print(f"Train rows: {len(X_train)}   Test rows: {len(X_test)}")

    # ---- Model 1: currentlevel (small hyperparameter search) ----
    model_level, params_level, mae_level, r2_level, pred_level = _search_best(
        X_train, y_level_train, X_test, y_level_test, "currentlevel"
    )
    baseline_level_mae = mean_absolute_error(y_level_test, X_test["prev_currentlevel"])
    acc_level_1m = accuracy_within(y_level_test, pred_level, 1.0)
    acc_level_2m = accuracy_within(y_level_test, pred_level, 2.0)
    print(f"  Accuracy: {acc_level_1m:.1f}% within +/-1m, {acc_level_2m:.1f}% within +/-2m  (naive baseline MAE={baseline_level_mae:.3f} m)")

    # ---- Model 2: level_diff (small hyperparameter search) ----
    model_diff, params_diff, mae_diff, r2_diff, pred_diff = _search_best(
        X_train, y_diff_train, X_test, y_diff_test, "level_diff"
    )
    baseline_diff_mae = mean_absolute_error(y_diff_test, X_test["prev_level_diff"])
    acc_diff_05m = accuracy_within(y_diff_test, pred_diff, 0.5)
    acc_diff_1m = accuracy_within(y_diff_test, pred_diff, 1.0)
    print(f"  Accuracy: {acc_diff_05m:.1f}% within +/-0.5m, {acc_diff_1m:.1f}% within +/-1m  (naive baseline MAE={baseline_diff_mae:.3f} m)")

    # ---- Save models + encoders ----
    joblib.dump(model_level, MODEL_CURRENTLEVEL_PATH)
    joblib.dump(model_diff, MODEL_LEVELDIFF_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"\nSaved models to {MODELS_DIR}")

    # ---- Metrics report ----
    report_path = OUTPUT_DIR / "model_metrics.txt"
    with open(report_path, "w") as f:
        f.write("Groundwater Level Forecasting Model - Metrics (v2, tuned)\n")
        f.write("=" * 58 + "\n\n")
        f.write(f"Train rows: {len(X_train)}   Test rows: {len(X_test)}\n\n")
        f.write("currentlevel model:\n")
        f.write(f"  Best params: {params_level}\n")
        f.write(f"  MAE: {mae_level:.3f} m\n")
        f.write(f"  R2:  {r2_level:.3f}  ({r2_level*100:.1f}% of variance explained)\n")
        f.write(f"  Accuracy: {acc_level_1m:.1f}% of predictions within +/-1m, {acc_level_2m:.1f}% within +/-2m\n")
        f.write(f"  Naive baseline MAE (persistence): {baseline_level_mae:.3f} m\n")
        f.write(f"  Improvement over baseline: {(1 - mae_level/baseline_level_mae)*100:.1f}%\n\n")
        f.write("level_diff model:\n")
        f.write(f"  Best params: {params_diff}\n")
        f.write(f"  MAE: {mae_diff:.3f} m\n")
        f.write(f"  R2:  {r2_diff:.3f}  ({r2_diff*100:.1f}% of variance explained)\n")
        f.write(f"  Accuracy: {acc_diff_05m:.1f}% of predictions within +/-0.5m, {acc_diff_1m:.1f}% within +/-1m\n")
        f.write(f"  Naive baseline MAE (persistence): {baseline_diff_mae:.3f} m\n")
        f.write(f"  Improvement over baseline: {(1 - mae_diff/baseline_diff_mae)*100:.1f}%\n\n")
        f.write("Note on 'accuracy': R2 and MAE describe error in aggregate; the\n")
        f.write("+/-tolerance figures above are the intuitive 'accuracy' number to\n")
        f.write("quote (e.g. 'X% of predictions are within 1 metre of the true depth').\n")
        f.write("R2 as a percentage is a loose paraphrase, not a technically precise\n")
        f.write("accuracy metric -- state it as 'R2 of 0.80' if asked to be exact.\n\n")
        f.write("v2 changes vs. the original model: added a 2nd lag step + trend\n")
        f.write("feature, cyclical (sin/cos) month encoding, and a small per-model\n")
        f.write("hyperparameter search over HistGradientBoostingRegressor configs.\n")
    print(f"Saved metrics report to {report_path}")

    return model_level, model_diff, encoders


# ============================================================================
# STEP 4 — PREDICT: use the saved models on new inputs
# ============================================================================

def _load_prediction_artifacts():
    model_level = joblib.load(MODEL_CURRENTLEVEL_PATH)
    model_diff = joblib.load(MODEL_LEVELDIFF_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    station_lookup = pd.read_csv(STATION_LOOKUP_PATH, parse_dates=["last_date"])
    return model_level, model_diff, encoders, station_lookup.set_index("station_name")


def _resolve_prev_reading(input_row: dict, station_lookup: pd.DataFrame, station_lookup_indexed: pd.DataFrame):
    """
    The model needs the station's recent readings as lag features (last, and
    2nd-last when available). Falls back district -> state -> global average
    for a brand-new station, so prediction never hard-fails on an unseen
    station. A fallback has no real 2nd-last reading, so has_prev2 is 0.
    """
    station = input_row["station_name"]
    if station in station_lookup_indexed.index:
        row = station_lookup_indexed.loc[station]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return (
            float(row["last_currentlevel"]), float(row["last_level_diff"]),
            float(row["prev2_currentlevel"]), float(row["prev2_level_diff"]),
            int(row["has_prev2"]),
            float(row["hist_mean_currentlevel"]), float(row["hist_std_currentlevel"]),
            int(row["n_readings_so_far"]),
        )

    for match in (
        station_lookup[station_lookup["district_name"] == input_row["district_name"]],
        station_lookup[station_lookup["state_name"] == input_row["state_name"]],
        station_lookup,
    ):
        if len(match):
            return (
                float(match["last_currentlevel"].mean()), float(match["last_level_diff"].mean()),
                0.0, 0.0, 0,
                float(match["hist_mean_currentlevel"].mean()), 0.0, 0,
            )

    return 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0


def _classify_risk(level_diff_pred: float) -> str:
    """
    Rising depth-to-water-level (positive level_diff, i.e. the water table
    has dropped further) is the signal. Thresholds are illustrative — tune
    them against your own definition of "critical" once you have domain
    guidance.
    """
    if level_diff_pred >= 2.0:
        return "Critical"
    elif level_diff_pred >= 0.5:
        return "Watch"
    else:
        return "Normal"


def predict_one(input_row: dict) -> dict:
    """
    input_row must contain exactly:
        date, state_name, state_code, district_name, district_code,
        station_name, latitude, longitude, basin, sub_basin, source
    """
    missing = [c for c in INPUT_COLUMNS if c not in input_row]
    if missing:
        raise ValueError(f"Missing required input fields: {missing}")

    model_level, model_diff, encoders, station_lookup_indexed = _load_prediction_artifacts()
    station_lookup = station_lookup_indexed.reset_index()

    (prev_level, prev_diff, prev2_level, prev2_diff, has_prev2,
     hist_mean, hist_std, n_readings) = _resolve_prev_reading(
        input_row, station_lookup, station_lookup_indexed
    )
    trend_currentlevel = (prev_level - prev2_level) if has_prev2 else 0.0

    d = pd.to_datetime(input_row["date"])
    month_sin = np.sin(2 * np.pi * d.month / 12)
    month_cos = np.cos(2 * np.pi * d.month / 12)

    row = pd.DataFrame([{
        "state_name": input_row["state_name"],
        "district_name": input_row["district_name"],
        "basin": input_row["basin"],
        "sub_basin": input_row["sub_basin"],
        "source": input_row["source"],
        "latitude": float(input_row["latitude"]),
        "longitude": float(input_row["longitude"]),
        "month_sin": month_sin,
        "month_cos": month_cos,
        "gap_days": 180,  # CGWB readings are typically ~6 months apart
        "prev_currentlevel": prev_level,
        "prev_level_diff": prev_diff,
        "prev2_currentlevel": prev2_level,
        "prev2_level_diff": prev2_diff,
        "trend_currentlevel": trend_currentlevel,
        "has_prev2": has_prev2,
        "hist_mean_currentlevel": hist_mean,
        "hist_std_currentlevel": hist_std,
        "n_readings_so_far": n_readings,
    }])

    row = apply_encoders(row, encoders)
    X = row[feature_columns()]

    pred_level = float(model_level.predict(X)[0])
    pred_diff = float(model_diff.predict(X)[0])
    pred_level = max(pred_level, 0.0)

    return {
        "station_name": input_row["station_name"],
        "date": input_row["date"],
        "current_depth_level_m": round(pred_level, 2),
        "level_difference_m": round(pred_diff, 2),
        "risk_flag": _classify_risk(pred_diff),
    }


def predict_batch(input_rows: list) -> pd.DataFrame:
    return pd.DataFrame([predict_one(r) for r in input_rows])


# ============================================================================
# INTERACTIVE MODE — type in your own inputs, get a prediction back
# ============================================================================

# Shown as the default for each field when you just press Enter — handy for
# quickly testing without retyping everything. Feel free to change these.
DEFAULTS = {
    "date": "2023-11-15",
    "state_name": "Tamil Nadu",
    "state_code": "33",
    "district_name": "Chennai",
    "district_code": "601",
    "station_name": "Maruthur2",   # a known station -> uses its real history
    "latitude": "13.0827",
    "longitude": "80.2707",
    "basin": "Cauvery Basin",
    "sub_basin": "Cauvery",
    "source": "CGWB",
}

FIELD_PROMPTS = {
    "date": "Date (YYYY-MM-DD)",
    "state_name": "State name",
    "state_code": "State code",
    "district_name": "District name",
    "district_code": "District code",
    "station_name": "Station name",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "basin": "Basin",
    "sub_basin": "Sub-basin",
    "source": "Source",
}


def prompt_for_input() -> dict:
    """Ask the user for each field on the console, pre-filled with a default."""
    print("\nEnter the station details (press Enter to keep the [default]):\n")
    values = {}
    for col in INPUT_COLUMNS:
        default = DEFAULTS[col]
        typed = input(f"  {FIELD_PROMPTS[col]} [{default}]: ").strip()
        values[col] = typed if typed else default

    # state_code / district_code are numeric in the schema
    values["state_code"] = int(float(values["state_code"]))
    values["district_code"] = int(float(values["district_code"]))
    values["latitude"] = float(values["latitude"])
    values["longitude"] = float(values["longitude"])
    return values


def print_result(result: dict):
    print("\n" + "-" * 44)
    print("  PREDICTION")
    print("-" * 44)
    print(f"  Station:             {result['station_name']}")
    print(f"  Date:                {result['date']}")
    print(f"  Current depth level: {result['current_depth_level_m']} m below ground")
    print(f"  Level difference:    {result['level_difference_m']} m")
    print(f"  Risk flag:           {result['risk_flag']}")
    print("-" * 44 + "\n")


def run_interactive():
    print("\nGroundwater Level Prediction — interactive mode")
    print("(Ctrl+C at any time to quit)\n")
    while True:
        input_row = prompt_for_input()
        try:
            result = predict_one(input_row)
            print_result(result)
        except Exception as e:
            print(f"\nCouldn't produce a prediction: {e}\n")

        again = input("Predict another station? (y/n): ").strip().lower()
        if again != "y":
            break
    print("Done.")


# ============================================================================
# MAIN — run everything end-to-end
# ============================================================================

def main():
    models_exist = MODEL_CURRENTLEVEL_PATH.exists() and MODEL_LEVELDIFF_PATH.exists() and ENCODERS_PATH.exists()

    if not models_exist:
        print("No trained models found — running the full pipeline (clean -> train) ...\n")
        df_clean = run_data_pipeline()
        train_models(df_clean)
    else:
        print("Trained models found — skipping straight to prediction.")
        print("(Delete the models/ folder to force a full retrain.)")

    run_interactive()


if __name__ == "__main__":
    main()

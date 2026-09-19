"""
Step 2: Feature engineering.

Builds a supervised "lag" dataset: for each station, each consecutive pair of
readings (reading at time t -> reading at time t+1) becomes one training row.
Features describe the station's location/context plus its recent readings;
the targets are the next reading's currentlevel and level_diff.

v2 additions (accuracy improvement pass):
  - a second lag step (prev2_currentlevel / prev2_level_diff) when a station
    has 3+ readings, so the model sees a short trend, not just a snapshot
  - a trend feature (prev_currentlevel - prev2_currentlevel), 0 when there is
    no second lag available
  - cyclical month encoding (month_sin / month_cos) instead of a raw 1-12
    integer, so December and January are no longer numerically far apart
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder

import config

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


def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    return df


def build_supervised_dataset(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Turns the raw time series into (features at t) -> (targets at t+1) rows,
    one per station per consecutive reading. When a station has a reading
    before `cur` as well, its values feed prev2_* / trend_currentlevel;
    otherwise those default to 0 and has_prev2 flags it.
    """
    df = add_time_parts(df_clean)

    records = []
    for station, g in df.groupby("station_name"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < 2:
            continue

        levels_so_far = []  # expanding history, up to and including `cur` — never `nxt`
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

    sup = pd.DataFrame(records)
    return sup


def fit_encoders(sup: pd.DataFrame):
    """Fit one TargetEncoder per categorical column against target_currentlevel."""
    encoders = {}
    for col in CATEGORICAL_COLS:
        # Note: newer scikit-learn (1.9+) deprecates `shuffle`/`random_state` on
        # TargetEncoder in favor of a `cv` generator; we leave both at their
        # defaults so this runs warning-free on old and new sklearn alike.
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

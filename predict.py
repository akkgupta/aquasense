"""
Step 4: Predict groundwater depth level for a station/date.

Input  (matches the CGWB schema):
    date, state_name, state_code, district_name, district_code,
    station_name, latitude, longitude, basin, sub_basin, source

Output:
    current_depth_level  -> predicted depth to water level (metres below ground)
    level_difference     -> predicted change vs. the CGWB reference period
    risk_flag            -> "Critical" / "Watch" / "Normal" bonus classification
                            based on the predicted level_difference direction/size

Usage as a script (edit the SAMPLE_INPUT dict below, or import predict_one /
predict_batch into your own code):

    python predict.py
"""
import numpy as np
import joblib
import pandas as pd

import config
import features

# ---------------------------------------------------------------------------
# Load trained artifacts once at import time
# ---------------------------------------------------------------------------
_model_level = joblib.load(config.MODEL_CURRENTLEVEL_PATH)
_model_diff = joblib.load(config.MODEL_LEVELDIFF_PATH)
_encoders = joblib.load(config.ENCODERS_PATH)
_station_lookup = pd.read_csv(config.STATION_LOOKUP_PATH, parse_dates=["last_date"])
_station_lookup_indexed = _station_lookup.set_index("station_name")


def _resolve_prev_reading(input_row: dict):
    """
    The model needs the station's recent readings and running history stats
    as features (last, 2nd-last, expanding mean/std/count when available).
    - If the station exists in the lookup table, use its own history.
    - Otherwise (a brand-new station), fall back to the district's average,
      then the state's average, so the pipeline never hard-fails on an
      unseen station. A fallback has no real 2nd-last reading, so has_prev2
      is 0 and the trend/history-stat features default to 0.
    """
    station = input_row["station_name"]
    if station in _station_lookup_indexed.index:
        row = _station_lookup_indexed.loc[station]
        if isinstance(row, pd.DataFrame):  # duplicate station names -> take first
            row = row.iloc[0]
        return (
            float(row["last_currentlevel"]), float(row["last_level_diff"]),
            float(row["prev2_currentlevel"]), float(row["prev2_level_diff"]),
            int(row["has_prev2"]),
            float(row["hist_mean_currentlevel"]), float(row["hist_std_currentlevel"]),
            int(row["n_readings_so_far"]),
        )

    # Fallback: district average, else state average, else global average
    for match in (
        _station_lookup[_station_lookup["district_name"] == input_row["district_name"]],
        _station_lookup[_station_lookup["state_name"] == input_row["state_name"]],
        _station_lookup,
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
    Simple, transparent bonus classification (not part of the original
    schema, but useful for the "identify locations where depletion may
    become critical" framing). A rising depth-to-water-level (positive
    level_diff, i.e. the water table has dropped further) is the signal.
    Thresholds are illustrative — tune them against your own definition of
    "critical" once you have domain guidance.
    """
    if level_diff_pred >= 2.0:
        return "Critical"
    elif level_diff_pred >= 0.5:
        return "Watch"
    else:
        return "Normal"


def predict_one(input_row: dict) -> dict:
    """
    input_row must contain exactly the fields listed in config.INPUT_COLUMNS:
        date, state_name, state_code, district_name, district_code,
        station_name, latitude, longitude, basin, sub_basin, source
    """
    missing = [c for c in config.INPUT_COLUMNS if c not in input_row]
    if missing:
        raise ValueError(f"Missing required input fields: {missing}")

    (prev_level, prev_diff, prev2_level, prev2_diff, has_prev2,
     hist_mean, hist_std, n_readings) = _resolve_prev_reading(input_row)
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
        "gap_days": 180,  # CGWB readings are typically ~6 months apart; adjust if you know the true gap
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

    row = features.apply_encoders(row, _encoders)
    X = row[features.feature_columns()]

    pred_level = float(_model_level.predict(X)[0])
    pred_diff = float(_model_diff.predict(X)[0])
    pred_level = max(pred_level, 0.0)  # depth can't be negative

    return {
        "station_name": input_row["station_name"],
        "date": input_row["date"],
        "current_depth_level_m": round(pred_level, 2),
        "level_difference_m": round(pred_diff, 2),
        "risk_flag": _classify_risk(pred_diff),
    }


def predict_batch(input_rows: list) -> pd.DataFrame:
    return pd.DataFrame([predict_one(r) for r in input_rows])


# ---------------------------------------------------------------------------
# Example usage — edit this and run `python predict.py`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SAMPLE_INPUT = {
        "date": "2023-11-15",
        "state_name": "Tamil Nadu",
        "state_code": 33,
        "district_name": "Chennai",
        "district_code": 601,
        "station_name": "Maruthur2",       # a known station -> uses its real history
        "latitude": 13.0827,
        "longitude": 80.2707,
        "basin": "Cauvery Basin",
        "sub_basin": "Cauvery",
        "source": "CGWB",
    }

    result = predict_one(SAMPLE_INPUT)
    print("Prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")

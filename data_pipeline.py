"""
Step 1: Load and clean the raw CGWB "changes in depth to water level" CSV.

Run directly to (re)generate the cleaned dataset:
    python data_pipeline.py
"""
import pandas as pd
import numpy as np

import config


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Normalize text fields
    text_cols = ["state_name", "district_name", "station_name", "basin", "sub_basin", "source"]
    for c in text_cols:
        df[c] = df[c].astype(str).str.strip()

    # Numeric coercion
    num_cols = ["state_code", "district_code", "latitude", "longitude", "currentlevel", "level_diff"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=num_cols)

    # Sanity filters: depth to water level should be non-negative and within a
    # physically plausible range for the readings in this dataset. Latitude/
    # longitude must fall inside India's bounding box.
    df = df[(df["currentlevel"] >= 0) & (df["currentlevel"] <= 300)]
    df = df[df["level_diff"].abs() <= 300]
    df = df[df["latitude"].between(6, 38) & df["longitude"].between(68, 98)]

    # Drop exact duplicate readings
    df = df.drop_duplicates()

    # Sort for downstream lag-feature construction
    df = df.sort_values(["station_name", "date"]).reset_index(drop=True)

    return df


def main():
    print(f"Loading raw data from {config.RAW_CSV_PATH} ...")
    df = load_raw()
    print(f"Raw shape: {df.shape}")

    df_clean = clean(df)
    print(f"Clean shape: {df_clean.shape}")
    print(f"Stations: {df_clean['station_name'].nunique()}  States: {df_clean['state_name'].nunique()}")
    print(f"Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")

    config.CLEAN_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(config.CLEAN_CSV_PATH, index=False)
    print(f"Saved cleaned data to {config.CLEAN_CSV_PATH}")

    # Save a per-station "latest known readings" lookup table (last + 2nd-last
    # reading). Prediction time needs both: the last reading feeds the
    # prev_* lag features, and the 2nd-last feeds prev2_*/trend_currentlevel
    # so the model sees a short history instead of one snapshot.
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

        # Same expanding-history stats used in training (features.py), but
        # over ALL of this station's known readings — matches what the
        # training-time feature would be for its very last row.
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
    latest.to_csv(config.STATION_LOOKUP_PATH, index=False)
    print(f"Saved station lookup table to {config.STATION_LOOKUP_PATH}")


if __name__ == "__main__":
    main()

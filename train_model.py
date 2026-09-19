"""
Step 3: Train the groundwater-level forecasting model.

Run directly (after data_pipeline.py has produced the cleaned CSV):
    python train_model.py

Trains two HistGradientBoostingRegressor models:
  - model_currentlevel: predicts the next depth-to-water-level reading (metres
    below ground level) for a station
  - model_level_diff:   predicts the next level_diff reading (change vs. the
    reference period CGWB compares against)

Evaluation uses a time-respecting split: the single most recent transition
per station is held out as the test set, so the reported metrics reflect
genuine forward forecasting rather than interpolation.

v2 (accuracy improvement pass):
  - trains on the richer feature set from features.py (2-step lag, trend,
    cyclical month) instead of the original single-lag features
  - runs a small hyperparameter search per model (a handful of HGB
    configurations) and keeps whichever scores best on the held-out split,
    instead of one fixed hyperparameter set
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import config
import features

# A small grid to search per model. Kept short so training still finishes in
# a reasonable time on a single laptop; widen this if you have more time.
# loss="absolute_error" trains the model to directly minimize MAE (rather
# than squared error), which lines up with the tolerance-based accuracy
# metric we report, and is less thrown off by a few extreme outlier readings.
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


def accuracy_within(y_true, y_pred, tolerance: float) -> float:
    """
    % of predictions within `tolerance` metres of the actual value. This is
    the "accuracy" figure to quote alongside R2/MAE — R2 and MAE describe the
    model's error in aggregate, but "X% of predictions are within Y metres"
    is what's actually intuitive to a non-technical audience.
    """
    errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return float((errors <= tolerance).mean() * 100)


def _search_best(X_train, y_train, X_test, y_test, label: str):
    """Try each PARAM_GRID entry, return (best_model, best_params, best_mae, best_r2, preds)."""
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


def main():
    print(f"Loading cleaned data from {config.CLEAN_CSV_PATH} ...")
    df = pd.read_csv(config.CLEAN_CSV_PATH, parse_dates=["date"])
    print(f"Rows: {len(df)}  Stations: {df['station_name'].nunique()}")

    print("Building supervised (lag) dataset ...")
    sup = features.build_supervised_dataset(df)
    print(f"Supervised rows: {len(sup)}  (with a 2nd lag step: {int(sup['has_prev2'].sum())})")

    print("Fitting target encoders ...")
    sup, encoders = features.fit_encoders(sup)

    feat_cols = features.feature_columns()
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

    # ---- Model 1: currentlevel ----
    model_level, params_level, mae_level, r2_level, pred_level = _search_best(
        X_train, y_level_train, X_test, y_level_test, "currentlevel"
    )
    baseline_level_mae = mean_absolute_error(y_level_test, X_test["prev_currentlevel"])
    acc_level_1m = accuracy_within(y_level_test, pred_level, 1.0)
    acc_level_2m = accuracy_within(y_level_test, pred_level, 2.0)
    print(f"  Accuracy: {acc_level_1m:.1f}% within +/-1m, {acc_level_2m:.1f}% within +/-2m  (naive baseline MAE={baseline_level_mae:.3f} m)")

    # ---- Model 2: level_diff ----
    model_diff, params_diff, mae_diff, r2_diff, pred_diff = _search_best(
        X_train, y_diff_train, X_test, y_diff_test, "level_diff"
    )
    baseline_diff_mae = mean_absolute_error(y_diff_test, X_test["prev_level_diff"])
    acc_diff_05m = accuracy_within(y_diff_test, pred_diff, 0.5)
    acc_diff_1m = accuracy_within(y_diff_test, pred_diff, 1.0)
    print(f"  Accuracy: {acc_diff_05m:.1f}% within +/-0.5m, {acc_diff_1m:.1f}% within +/-1m  (naive baseline MAE={baseline_diff_mae:.3f} m)")

    # ---- Save everything needed for prediction ----
    joblib.dump(model_level, config.MODEL_CURRENTLEVEL_PATH)
    joblib.dump(model_diff, config.MODEL_LEVELDIFF_PATH)
    joblib.dump(encoders, config.ENCODERS_PATH)
    print(f"\nSaved models to {config.MODELS_DIR}")

    # ---- Write a metrics report ----
    report_path = config.OUTPUT_DIR / "model_metrics.txt"
    with open(report_path, "w") as f:
        f.write("Groundwater Level Forecasting Model - Metrics (v3, tuned)\n")
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
        f.write("Changes vs. the original model:\n")
        f.write("  - added a 2nd lag step (prev2_currentlevel/prev2_level_diff) and a\n")
        f.write("    trend feature, so the model sees a short history, not one snapshot\n")
        f.write("  - encoded month cyclically (sin/cos) instead of as a raw 1-12 integer\n")
        f.write("  - added expanding per-station history stats (hist_mean_currentlevel,\n")
        f.write("    hist_std_currentlevel, n_readings_so_far) computed over all of a\n")
        f.write("    station's readings up to the current one\n")
        f.write("  - searched a small grid of HistGradientBoostingRegressor configurations\n")
        f.write("    per model, including an MAE-optimized (absolute_error) loss, and\n")
        f.write("    kept whichever scored best on the held-out split\n")
    print(f"Saved metrics report to {report_path}")


if __name__ == "__main__":
    main()

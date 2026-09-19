# Groundwater Level Forecasting Model

An AI model that predicts groundwater depth level and its change, from the
CGWB "changes in depth to water level" dataset (550,850 readings, 23,078
stations, 2013–2023, across 32 Indian states).

## Input schema

| Field | Type | Example |
|---|---|---|
| `date` | string (YYYY-MM-DD) | `2023-11-15` |
| `state_name` | string | `Tamil Nadu` |
| `state_code` | int | `33` |
| `district_name` | string | `Chennai` |
| `district_code` | int | `601` |
| `station_name` | string | `Maruthur2` |
| `latitude` | float | `13.0827` |
| `longitude` | float | `80.2707` |
| `basin` | string | `Cauvery Basin` |
| `sub_basin` | string | `Cauvery` |
| `source` | string | `CGWB` |

## Output

| Field | Meaning |
|---|---|
| `current_depth_level_m` | predicted depth to water level, metres below ground |
| `level_difference_m` | predicted change vs. the CGWB reference period |
| `risk_flag` | bonus label: `Critical` / `Watch` / `Normal`, based on the predicted level_difference (a rising depth = water table dropping further) |

## Project layout

```
gwl_project/
  config.py           file paths & hyperparameters — edit RAW_CSV_PATH if you move the source CSV
  data_pipeline.py     Step 1: load + clean the raw CSV, build the per-station lookup table
  features.py           Step 2: lag-feature engineering + categorical target encoding
  train_model.py        Step 3: trains and saves the two forecasting models
  predict.py             Step 4: load the saved models and predict for new inputs
  data/                  put the raw CSV here (already included)
  models/                 trained models are saved here (created by train_model.py)
  output/                 metrics report is saved here
```

## Running it in PyCharm

1. Open this folder as a PyCharm project.
2. Create a virtual environment (PyCharm will usually offer to do this automatically), then install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the three steps in order (right-click each file → "Run"), or from the terminal:
   ```
   python data_pipeline.py     # ~10 seconds — cleans the raw CSV
   python train_model.py       # ~2 minutes — trains and saves the models
   python predict.py           # instant — runs the sample prediction at the bottom of the file
   ```
4. To predict for your own input, either edit the `SAMPLE_INPUT` dict at the bottom of `predict.py`, or import it into your own script:
   ```python
   from predict import predict_one

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
   ```
   `predict_batch(list_of_input_dicts)` does the same for many rows at once and returns a DataFrame.

## How it works

- **Lag-feature supervised learning.** Every consecutive pair of readings at a
  station becomes one training example: "given the station's location and its
  last known reading, what was the *next* reading?" This is the same
  structure used at prediction time.
- **Unseen stations** fall back to the district average, then the state
  average, then the national average, for the "previous reading" lag feature
  — so the pipeline never hard-fails on a brand-new station, it just has less
  to go on.
- **Two `HistGradientBoostingRegressor` models** (fast on this dataset's
  ~450K training rows): one predicts the next `currentlevel`, the other the
  next `level_diff`.
- **Evaluation** holds out each station's single most recent reading as the
  test set (a time-respecting split), so the reported R²/MAE reflect genuine
  forecasting rather than interpolation. Current results (see
  `output/model_metrics.txt` after training):
  - `currentlevel`: R² ≈ 0.80, ~9% better MAE than a naive "next = same as
    last" baseline.
  - `level_diff`: R² ≈ 0.15, ~49% better MAE than the naive baseline
    (level_diff is inherently noisier — it's a difference, not a level).

## Known limitations

- Readings are irregular per station (some report ~2x/year, some far more
  often via automated loggers), so `gap_days` is a feature but the model
  is not a true continuous-time series model (no ARIMA/state-space
  component).
- No rainfall or extraction-volume data is joined in yet. Adding
  district-level rainfall (IMD) and CGWB dynamic groundwater resource
  assessments as extra features would likely improve accuracy further,
  especially for `level_diff`.
- The `risk_flag` thresholds (±2.0 m / ±0.5 m) are illustrative — tune them
  once you have a domain definition of "critical" depletion for your region.

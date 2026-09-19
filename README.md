# AquaSense — AI Groundwater Level Forecasting

**Team 7 · PRISMTECH Hackathon · Neural/AI Stream**

## Description

AquaSense forecasts groundwater depth levels and their change across India,
trained on real CGWB (Central Ground Water Board) data — 550,850 readings
from 23,078 monitoring stations across 32 states, spanning 2013–2023.

Given a station's location and its most recent reading, the model predicts:

- **`current_depth_level_m`** — the next depth-to-water level (metres below ground)
- **`level_difference_m`** — the predicted change vs. the CGWB reference period
- **`risk_flag`** — `Critical` / `Watch` / `Normal`, derived from the predicted change

The trained model is served through a live website: pick a state, district
and station, and get an instant forecast — no code required to use it.

**Measured accuracy** (held-out forecast split, not interpolation):

| Model | Accuracy | R² |
|---|---|---|
| Depth level | 77.4% within ±2m | 0.80 |
| Level change | 59.7% within ±1m | 0.16 |

## Tech Stack

| Layer | Tool |
|---|---|
| Model | `HistGradientBoostingRegressor` (scikit-learn) |
| Feature encoding | `sklearn.preprocessing.TargetEncoder`, lag features, cyclical month encoding |
| Backend | FastAPI (Python) |
| Frontend | JavaScript, HTML, CSS (no framework, no build step) |
| Database | MongoDB (prediction history), with a local-file fallback when no `MONGO_URI` is set |

## Project Layout

```
gwl_project/
  config.py              File paths & hyperparameters
  data_pipeline.py        Step 1: load + clean the raw CSV, build the per-station lookup table
  features.py              Step 2: lag-feature engineering + categorical target encoding
  train_model.py            Step 3: trains and saves the two forecasting models
  predict.py                 Step 4: load the saved models and predict for new inputs
  groundwater_model.py        Single-file version of the full pipeline (interactive console mode)
  data/                        Raw + cleaned CGWB CSVs, station lookup table
  models/                       Trained models (created by train_model.py)
  output/                       Metrics reports, prediction history log
  webapp/
    backend/main.py             FastAPI app — serves the model + the frontend
    frontend/                    index.html, style.css, app.js
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -r webapp/backend/requirements.txt
```

### 2. Train the model (first time only — trained models are already included, but to retrain from scratch)

```bash
python data_pipeline.py     # cleans the raw CSV, builds the station lookup (~10s)
python train_model.py       # trains and saves both models (~2 min)
```

### 3. Run the website

```bash
cd webapp/backend
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — the form is served from the same server
that answers the prediction API.

Optional: connect a real MongoDB instance for persistent prediction history
(otherwise predictions log to a local file automatically):

```bash
export MONGO_URI="mongodb://localhost:27017"
```

### 4. Or use the model directly in Python

```python
from predict import predict_one

result = predict_one({
    "date": "2024-06-01",
    "state_name": "Karnataka",
    "district_name": "Bengaluru Urban",
    "station_name": "Some Station",
    "latitude": 12.9716,
    "longitude": 77.5946,
    "basin": "Cauvery Basin",
    "sub_basin": "Cauvery",
    "source": "CGWB",
})
print(result)
```

## How It Works

- **Lag-feature supervised learning.** Every consecutive pair of readings at
  a station becomes one training example: "given the station's location and
  its recent history, what was the *next* reading?" — the same structure
  used at prediction time.
- **Expanding per-station history stats** (mean, std, count of readings so
  far) are computed only from readings up to and including the current one,
  never future readings, to avoid leakage.
- **Unseen stations** fall back to the district average, then state, then
  national average for the lag features, so the pipeline never hard-fails
  on a new station.
- **Time-respecting evaluation**: each station's single most recent reading
  is held out as the test set, so the reported accuracy reflects genuine
  forecasting rather than interpolation.

## Known Limitations

- Readings are irregular per station (some report ~2x/year, some more often
  via automated loggers) — `gap_days` is a feature, but this isn't a true
  continuous-time-series model.
- No rainfall or groundwater-extraction data is joined in yet; adding
  district-level rainfall (IMD) would likely improve `level_difference_m`
  accuracy further.
- The `risk_flag` thresholds (±2.0 m / ±0.5 m) are illustrative — tune them
  to a domain definition of "critical" depletion for your region.

## Team

Team 7 — AquaSense — Neural/AI Stream — PRISMTECH Hackathon

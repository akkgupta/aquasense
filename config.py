"""
Central configuration: file paths and model hyperparameters.
Edit RAW_CSV_PATH if you move the source file.
"""
from pathlib import Path

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

# Ensure output/model dirs exist when this module is imported
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The exact input schema this project expects (matches the CGWB source columns)
INPUT_COLUMNS = [
    "date",
    "state_name",
    "state_code",
    "district_name",
    "district_code",
    "station_name",
    "latitude",
    "longitude",
    "basin",
    "sub_basin",
    "source",
]

TARGET_COLUMNS = ["currentlevel", "level_diff"]

# Model hyperparameters (HistGradientBoostingRegressor: scales well to
# hundreds of thousands of rows, unlike plain GradientBoostingRegressor)
HGB_PARAMS = dict(
    max_iter=300,
    max_depth=6,
    learning_rate=0.06,
    l2_regularization=0.1,
    random_state=42,
)

RANDOM_STATE = 42

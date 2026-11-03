
import os
import json
import requests
import numpy as np
import pandas as pd

from pathlib import Path

# ==================================================
# 1. PATHS AND SETTINGS
# ==================================================

# Project root directory (one level above Code)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
DL_DIR = BASE_DIR / "data" / "deep_learning_features"

FRED_FILE = DATA_DIR / "forecast_features.csv"

print("Project directory:", BASE_DIR)
print("Economic data exists:", FRED_FILE.exists())
print("Feature directory exists:", DL_DIR.exists())

if DL_DIR.exists():
    print("\nAvailable deep learning files:")
    for file in DL_DIR.glob("*.csv"):
        print(file.name)

if not FRED_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find {FRED_FILE}. "
        "Check that forecast_features.csv is inside data/processed."
    )
RAW_TEXT_DIR = BASE_DIR / "data" / "raw" / "earnings_releases"
DL_FEATURE_DIR = BASE_DIR / "data" / "deep_learning_features"

OUTPUT_DIR = BASE_DIR / "results" / "deep_learning"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DL_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

FRED_FILE = DATA_DIR / "forecast_features.csv"

# Ollama local model
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


# ==================================================
# 2. LOAD EXISTING AIRLINE AND FUEL DATA
# ==================================================

df = pd.read_csv(FRED_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")

print("Existing economic data loaded:")
print(df.head())

# -----------------------------------------
# MERGE DEEP LEARNING FEATURES
# -----------------------------------------

new_dl_columns = []

feature_files = {
    "audio": DL_DIR / "audio_features.csv",
    "vision": DL_DIR / "vision_features.csv",
    "weather": DL_DIR / "weather_features.csv",
    "movement": DL_DIR / "movement_features.csv",
}

for feature_type, file_path in feature_files.items():

    if not file_path.exists():
        print(f"Skipping {feature_type}: file not found")
        continue

    feature_df = pd.read_csv(file_path)

    feature_df["date"] = pd.to_datetime(feature_df["date"])

    # Merge using monthly dates
    df["date"] = pd.to_datetime(df["date"])

    feature_columns = [
        col for col in feature_df.columns
        if col != "date"
    ]

    df = df.merge(
        feature_df,
        on="date",
        how="left"
    )

    new_dl_columns.extend(feature_columns)

    print(f"Added {feature_type} features:", feature_columns)


# -----------------------------------------
# CREATE LAGS ONLY FOR NEW FEATURES
# -----------------------------------------

for col in new_dl_columns:

    for lag in [1, 2, 3]:

        df[f"{col}_lag{lag}"] = df[col].shift(lag)


# -----------------------------------------
# SAVE FINAL DATASET
# -----------------------------------------

output_dir = BASE_DIR / "results" / "deep_learning"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "combined_deep_learning_data.csv"

df.to_csv(output_file, index=False)

print("\nFinal dataset columns:")
print(df.columns.tolist())

print("\nNew deep learning features:")
print(new_dl_columns)

print("\nSaved dataset to:")
print(output_file)

# ==================================================
# 3. TEXT DATA: AIRLINE EARNINGS RELEASES
# ==================================================

def extract_airline_guidance(text):

    """
    Use Llama to extract airline unit-revenue
    guidance from an earnings release.

    Output: numerical guidance midpoint.
    """

    prompt = f"""
    Analyze this airline earnings release.

    Extract the expected percentage change in
    unit revenue (RASM or PRASM).

    Return JSON with:
    - guidance_found: true or false
    - low: lower percentage guidance
    - high: upper percentage guidance
    - confidence: score from 0 to 1

    If no guidance exists, return null values.

    Do not invent numbers.

    TEXT:
    {text[:10000]}
    """

    schema = {
        "type": "object",
        "properties": {
            "guidance_found": {"type": "boolean"},
            "low": {"type": ["number", "null"]},
            "high": {"type": ["number", "null"]},
            "confidence": {"type": "number"}
        },
        "required": [
            "guidance_found",
            "low",
            "high",
            "confidence"
        ]
    }

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "format": schema,
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=180
    )

    response.raise_for_status()

    result = json.loads(response.json()["response"])

    # Check that extracted numbers appear in source text.
    if not result["guidance_found"]:
        return None

    low = result["low"]
    high = result["high"]

    if low is None or high is None:
        return None

    # Basic validation against hallucinated numbers.
    text_numbers = set(
        pd.to_numeric(
            pd.Series(
                __import__("re").findall(
                    r"[-+]?\d+(?:\.\d+)?", text
                )
            ),
            errors="coerce"
        ).dropna().round(2)
    )

    if (
        round(float(low), 2) not in text_numbers
        or round(float(high), 2) not in text_numbers
    ):
        print("Guidance failed source-number check.")
        return None

    if not 0 <= result["confidence"] <= 1:
        return None

    return {
        "guidance_low": float(low),
        "guidance_high": float(high),
        "guidance_midpoint": (float(low) + float(high)) / 2,
        "confidence": float(result["confidence"])
    }


# Process individual earnings releases
text_records = []

if RAW_TEXT_DIR.exists():

    for file in RAW_TEXT_DIR.glob("*.txt"):

        text = file.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        # Expected filename: YYYY-MM-DD_airline.txt
        date_text = file.name[:10]

        try:
            release_date = pd.to_datetime(date_text)
        except ValueError:
            print(f"Skipping file with invalid date: {file.name}")
            continue

        try:
            extracted = extract_airline_guidance(text)

            if extracted is not None:

                text_records.append({
                    "date": release_date,
                    "airline_guidance": extracted["guidance_midpoint"],
                    "guidance_confidence": extracted["confidence"]
                })

        except Exception as error:
            print(f"Text extraction error: {file.name}: {error}")

else:
    print("No earnings release text folder found.")

text_df = pd.DataFrame(text_records)

if not text_df.empty:

    # Aggregate releases into monthly features.
    text_df["date"] = text_df["date"].dt.to_period("M")

    text_monthly = (
        text_df.groupby("date")
        .agg(
            airline_guidance=("airline_guidance", "mean"),
            guidance_confidence=("guidance_confidence", "mean"),
            guidance_count=("airline_guidance", "count")
        )
        .reset_index()
    )

    text_monthly["date"] = (
        text_monthly["date"].dt.to_timestamp()
    )

    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

    df = df.merge(
        text_monthly,
        on="date",
        how="left"
    )


# ==================================================
# 4. OTHER DEEP-LEARNING DATA SOURCES
# ==================================================

"""
Each additional model should export a monthly CSV.

Required format:
date, feature columns

Examples:

vision_features.csv:
date, dot_complaints, aircraft_count

audio_features.csv:
date, executive_tone

weather_features.csv:
date, storm_hours

movement_features.csv:
date, flights_operated, routes_added

news_features.csv:
date, disruption_count, demand_sentiment

These columns must be generated from actual
source data, not manually invented values.
"""

feature_files = {
    "vision": "vision_features.csv",
    "audio": "audio_features.csv",
    "weather": "weather_features.csv",
    "movement": "movement_features.csv",
    "news": "news_features.csv"
}

for source, filename in feature_files.items():

    file_path = DL_FEATURE_DIR / filename

    if file_path.exists():

        source_df = pd.read_csv(file_path)

        source_df["date"] = pd.to_datetime(
            source_df["date"]
        ).dt.to_period("M").dt.to_timestamp()

        source_df = (
            source_df.groupby("date")
            .mean(numeric_only=True)
            .reset_index()
        )

        df = df.merge(
            source_df,
            on="date",
            how="left",
            validate="one_to_one"
        )

        print(f"Added {source} features.")

    else:
        print(f"Optional source not found: {filename}")


# ==================================================
# 5. CREATE LAGGED DEEP-LEARNING FEATURES
# ==================================================

# Identify newly added deep-learning variables.
economic_columns = {
    "airline_cpi",
    "jet_fuel",
    "unemployment",
    "core_cpi",
    "fare_inflation_yoy",
    "fare_inflation_mom",
    "fuel_change_mom",
    "fuel_change_yoy"
}

date_columns = {"date"}

dl_columns = [
    col for col in df.columns
    if col not in economic_columns | date_columns
]

# Ensure numerical data
for col in dl_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Sort before creating lags
df = df.sort_values("date").reset_index(drop=True)

# Lag all DL features by one month.
# This conservative convention avoids using
# information from the current month.
for col in dl_columns:

    df[f"{col}_lag1"] = df[col].shift(1)

    df[f"{col}_lag2"] = df[col].shift(2)

    df[f"{col}_lag3"] = df[col].shift(3)


# ==================================================
# 6. FINAL COMBINED DATASET
# ==================================================

df = df.replace([np.inf, -np.inf], np.nan)

output_file = OUTPUT_DIR / "combined_deep_learning_data.csv"

df.to_csv(output_file, index=False)

print("\nDeep-learning data pipeline completed.")

print("Output:", output_file)

print("\nFinal dataset columns:")
print(df.columns.tolist())

print("\nDataset preview:")
print(df.tail())

print("\nNumber of observations:", len(df))

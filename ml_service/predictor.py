import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "rf_leakage_classifier.pkl"
model = joblib.load(MODEL_PATH)

WINDOW_SIZE = 8

# Replace with values from your feature engineering notebook
PRESSURE_GRAD_MEAN = 0.0018
PRESSURE_GRAD_STD  = 2.9076
FLOW_DIFF_MEAN     = -0.0018
FLOW_DIFF_STD      = 9.0062

PRESSURE_THRESHOLD = PRESSURE_GRAD_MEAN - 3 * PRESSURE_GRAD_STD
FLOW_THRESHOLD     = FLOW_DIFF_MEAN + 3 * FLOW_DIFF_STD

# Normal operating pressure range in bar
# Update these based on RUJWASCO or LeakDB normal operating values
EXPECTED_PRESSURE_MIN = 3.0
EXPECTED_PRESSURE_MAX = 6.0

LABEL_MAP = {
    0: "No Leak",
    1: "Minor Leak",
    2: "Major Burst"
}

# Mapping ML labels to backend faultType and severity enums
FAULT_MAP = {
    0: {"anomalyType": None,           "severity": None},
    1: {"anomalyType": "LEAK",         "severity": "MEDIUM"},
    2: {"anomalyType": "PIPE_BURST", "severity": "CRITICAL"}
}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["Pressure_Gradient"]     = df["pressure"].diff()
    df["Rolling_Mean_Pressure"] = df["pressure"].rolling(
                                    window=WINDOW_SIZE, min_periods=1).mean()
    df["Rolling_Std_Pressure"]  = df["pressure"].rolling(
                                    window=WINDOW_SIZE, min_periods=1).std().fillna(0)
    df["Pressure_Anomaly"]      = (
        df["Pressure_Gradient"] < PRESSURE_THRESHOLD
    ).astype(int)

    df["Flow_Difference"]   = df["flow"].diff()
    df["Rolling_Mean_Flow"] = df["flow"].rolling(
                                window=WINDOW_SIZE, min_periods=1).mean()
    df["Flow_Variance"]     = df["flow"].rolling(
                                window=WINDOW_SIZE, min_periods=1).var().fillna(0)
    df["Flow_Spike"]        = (
        df["Flow_Difference"] > FLOW_THRESHOLD
    ).astype(int)

    df["Hour_Of_Day"] = df["timestamp"].dt.hour
    df["Day_Of_Week"] = df["timestamp"].dt.dayofweek

    return df


def predict(df: pd.DataFrame) -> dict:
    df_features = engineer_features(df)

    feature_cols = [
        "Pressure_Gradient", "Rolling_Mean_Pressure", "Rolling_Std_Pressure",
        "Pressure_Anomaly", "Flow_Difference", "Rolling_Mean_Flow",
        "Flow_Variance", "Flow_Spike", "Hour_Of_Day", "Day_Of_Week"
    ]

    latest_row    = df_features.iloc[-1]
    latest        = df_features[feature_cols].iloc[[-1]].fillna(0)
    prediction    = int(model.predict(latest)[0])
    probabilities = model.predict_proba(latest)[0].tolist()

    return {
        "label":            prediction,
        "status":           LABEL_MAP[prediction],
        "fault_type":       FAULT_MAP[prediction]["anomalyType"],
        "severity":         FAULT_MAP[prediction]["severity"],
        "pressure_reading": round(float(latest_row["pressure"]), 4),
        "confidence":       round(max(probabilities), 4),
        "probabilities": {
            "no_leak":     round(probabilities[0], 4),
            "minor_leak":  round(probabilities[1], 4),
            "major_burst": round(probabilities[2], 4)
        }
    }
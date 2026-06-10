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

# Normal operating ranges (in converted units: metres water head, m³/h)
# Based on LeakDB training data: pressure ~72–75m, flow ~75–82 m³/h
NORMAL_PRESSURE_MIN = 60.0   # metres water head (~5.9 bar)
NORMAL_PRESSURE_MAX = 85.0   # metres water head (~8.3 bar)
NORMAL_FLOW_MIN     = 50.0   # m³/h (~13.9 L/s)
NORMAL_FLOW_MAX     = 100.0  # m³/h (~27.8 L/s)

# Minimum time span (seconds) for derived features to be meaningful
# Training data used 30-min intervals; below this threshold, Rolling_Std
# and Flow_Variance will be near-zero regardless of system health
MIN_TIMESPAN_FOR_ML = 600  # 10 minutes

# Backward compatibility (used by app.py)
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
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    latest_pressure = float(df.iloc[-1]["pressure"])
    latest_flow = float(df.iloc[-1]["flow"])
    avg_pressure = float(df["pressure"].mean())
    avg_flow = float(df["flow"].mean())

    # --- Rule-based anomaly detection (absolute value check) ---
    # If pressure or flow is far outside normal operating range,
    # report anomaly directly — the RF model can't reliably detect this
    # because it relies on temporal features that are flat at short intervals.

    # Near-zero pressure = pipe burst or major supply failure
    if avg_pressure < 20.0:  # < 20m head (~2 bar) = critically low
        return {
            "label":            2,
            "status":           LABEL_MAP[2],
            "fault_type":       FAULT_MAP[2]["anomalyType"],
            "severity":         FAULT_MAP[2]["severity"],
            "pressure_reading": round(latest_pressure, 4),
            "confidence":       0.92,
            "probabilities": {
                "no_leak":     0.04,
                "minor_leak":  0.04,
                "major_burst": 0.92
            }
        }

    # Very low pressure but not zero = likely leak
    if avg_pressure < NORMAL_PRESSURE_MIN * 0.7:  # < 42m head (~4.1 bar)
        return {
            "label":            1,
            "status":           LABEL_MAP[1],
            "fault_type":       FAULT_MAP[1]["anomalyType"],
            "severity":         FAULT_MAP[1]["severity"],
            "pressure_reading": round(latest_pressure, 4),
            "confidence":       0.88,
            "probabilities": {
                "no_leak":     0.06,
                "minor_leak":  0.88,
                "major_burst": 0.06
            }
        }

    # Near-zero flow with normal pressure = blockage or valve closed
    if avg_flow < 5.0 and avg_pressure > NORMAL_PRESSURE_MIN:
        return {
            "label":            1,
            "status":           LABEL_MAP[1],
            "fault_type":       "FLOW_IRREGULARITY",
            "severity":         "MEDIUM",
            "pressure_reading": round(latest_pressure, 4),
            "confidence":       0.85,
            "probabilities": {
                "no_leak":     0.10,
                "minor_leak":  0.85,
                "major_burst": 0.05
            }
        }

    # --- Normal-range override ---
    # When buffer spans < MIN_TIMESPAN_FOR_ML and all readings are within
    # expected operating range, the temporal mismatch makes derived features
    # unreliable (Rolling_Std ≈ 0, Flow_Variance ≈ 0). Skip ML inference.
    time_span = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
    pressure_in_range = df["pressure"].between(NORMAL_PRESSURE_MIN, NORMAL_PRESSURE_MAX).all()
    flow_in_range = df["flow"].between(NORMAL_FLOW_MIN, NORMAL_FLOW_MAX).all()

    if time_span < MIN_TIMESPAN_FOR_ML and pressure_in_range and flow_in_range:
        return {
            "label":            0,
            "status":           LABEL_MAP[0],
            "fault_type":       None,
            "severity":         None,
            "pressure_reading": round(latest_pressure, 4),
            "confidence":       0.95,
            "probabilities": {
                "no_leak":     0.95,
                "minor_leak":  0.03,
                "major_burst": 0.02
            }
        }

    # --- Standard ML inference (features are meaningful) ---
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
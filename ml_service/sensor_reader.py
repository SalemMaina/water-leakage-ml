import requests
import pandas as pd
from collections import defaultdict
from config import SENSOR_API_URL, SENSOR_API_TOKEN

BUFFER_SIZE = 8

# In-memory buffer per device
device_buffers = defaultdict(list)

HEADERS = {
    "X-API-Key": SENSOR_API_TOKEN,
    "ngrok-skip-browser-warning": "true"  # required for ngrok URLs
}


def fetch_latest_readings() -> list:
    """
    Fetches the latest sensor readings from the backend REST API.
    Returns a list of reading dicts.
    """
    try:
        response = requests.get(
            SENSOR_API_URL,
            headers=HEADERS,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            readings = data.get("content", [])
            return readings
        else:
            print(f"API error: {response.status_code} {response.text}")
            return []

    except requests.exceptions.RequestException as e:
        print(f"Could not reach sensor API: {e}")
        return []


def update_buffers(readings: list):
    """
    Updates in-memory buffers for each device
    with the latest readings fetched from the API.
    """
    for reading in readings:
        device_code = reading.get("deviceCode")
        if not device_code:
            continue

        # Convert bar to metres of water head (1 bar ≈ 10.2 m)
        pressure_m = reading.get("pressureBar", 0) * 10.2

        # Convert Lps to m³/hour (1 Lps = 3.6 m³/hour)
        flow_lps      = reading.get("flowRateLps", 0.0)
        flow_m3h      = flow_lps * 3.6

        entry = {
            "timestamp": reading.get("recordedAt"),
            "pressure":  pressure_m,
            "flow":      flow_m3h,
            "device_code": device_code,
            "location":  reading.get("deviceLocation")
        }

        device_buffers[device_code].append(entry)

        # Keep only last BUFFER_SIZE readings
        if len(device_buffers[device_code]) > BUFFER_SIZE:
            device_buffers[device_code].pop(0)


def get_device_buffer_df(device_code: str) -> pd.DataFrame:
    """
    Returns buffered readings for a device as a DataFrame
    ready for feature engineering.
    """
    buffer = device_buffers.get(device_code, [])

    if len(buffer) < 2:
        raise ValueError(
            f"Not enough readings buffered for {device_code}. "
            f"Have {len(buffer)}, need at least 2."
        )

    df = pd.DataFrame(buffer)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def get_all_device_codes() -> list:
    """Returns all device codes currently in the buffer."""
    return list(device_buffers.keys())
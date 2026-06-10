import time
import threading
import logging
import requests
from datetime import datetime
from flask import Flask, jsonify

from sensor_reader import fetch_latest_readings, update_buffers, get_device_buffer_df, get_all_device_codes
from predictor import predict, EXPECTED_PRESSURE_MIN, EXPECTED_PRESSURE_MAX
from config import BACKEND_URL, ML_TOKEN, POLL_INTERVAL

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

ALERT_HEADERS = {
    "X-API-Key":                  ML_TOKEN,
    "Content-Type":  "application/json",
    "ngrok-skip-browser-warning": "true"

}


def send_alert(device_code, anomaly_type, severity,
               confidence, estimated_flow, location, message):
    """Sends a POST alert to the Spring Boot backend."""

    payload = {
        "deviceCode":            device_code,
        "anomalyType":           anomaly_type,
        "severity":              severity,
        "description":           message,
        "confidenceScore":       confidence,
        "detectedAt":            datetime.now().isoformat(),
        "triggerReadingId":      None,
        "estimatedLeakRateLps":  estimated_flow,
        "faultLatitude":         None,   # placeholder — update when GPS available
        "faultLongitude":        None,   # placeholder — update when GPS available
        "faultPipelineSegment":  location if location else "UNKNOWN"
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/anomalies",
            json=payload,
            headers=ALERT_HEADERS,
            timeout=5
        )
        if response.status_code == 200:
            logging.info(
                f"✅ Alert sent — {device_code} | "
                f"{anomaly_type} | {severity}"
            )
        else:
            logging.error(
                f"Alert failed — {device_code}: "
                f"{response.status_code} {response.text}"
            )
    except requests.exceptions.RequestException as e:
        logging.error(f"Could not reach backend: {e}")


def generate_message(device_code, location, label, pressure, confidence):
    location_str = f" at {location}" if location else ""
    if label == 1:
        return (
            f"Minor leak detected on device {device_code}{location_str}. "
            f"Pressure reading of {pressure} bar indicates gradual pressure loss. "
            f"Model confidence: {confidence}"
        )
    elif label == 2:
        return (
            f"Major pipe burst detected on device {device_code}{location_str}. "
            f"Pressure reading of {pressure} bar indicates catastrophic failure. "
            f"Immediate response required. Model confidence: {confidence}"
        )
    return ""


def processing_loop():
    """
    Main loop — polls the sensor API every POLL_INTERVAL seconds,
    updates buffers, runs predictions, and sends alerts.
    """
    logging.info(f"ML processing loop started — polling every {POLL_INTERVAL}s")

    while True:
        try:
            # ── Step 1: Fetch latest readings from API ──────────
            logging.info("Fetching sensor readings...")
            readings = fetch_latest_readings()

            for reading in readings:
                    print(reading)


            if not readings:
                logging.warning("No readings received — retrying next poll")
                time.sleep(POLL_INTERVAL)
                continue

            logging.info(f"Received {len(readings)} reading(s)")

            # ── Step 2: Update in-memory buffers ────────────────
            update_buffers(readings)

            # ── Step 3: Predict for each device ─────────────────
            for device_code in get_all_device_codes():
                try:
                    df = get_device_buffer_df(device_code)
                    result = predict(df)
                    label  = result["label"]
                    location = df.iloc[-1].get("location", "")

                    logging.info(
                        f"  {device_code} → {result['status']} "
                        f"(confidence: {result['confidence']})"
                    )

                    # ── Step 4: Send alert if leak detected ──────
                    if label in [1, 2] and result["confidence"] > 0.85:
                        latest_flow = float(df.iloc[-1].get("flow", 0.0))

                        message = generate_message(
                            device_code,
                            location,
                            label,
                            result["pressure_reading"],
                            result["confidence"]
                        )
                        send_alert(
                            device_code = device_code,
                            anomaly_type  = result["fault_type"],
                            severity    = result["severity"],
                            confidence     = result["confidence"],
                            estimated_flow = round(latest_flow, 4),
                            location   = location,
                            message     = message
                        )

                except ValueError as e:
                    logging.info(f"  {device_code} — skipped: {e}")
                except Exception as e:
                    logging.error(f"  {device_code} — prediction error: {e}")

        except Exception as e:
            logging.error(f"Processing loop error: {e}")

        # ── Wait before next poll ────────────────────────────────
        logging.info(f"Sleeping {POLL_INTERVAL}s until next poll...")
        time.sleep(POLL_INTERVAL)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ML service running"}), 200


@app.route("/status", methods=["GET"])
def status():
    """Returns current buffer status for all devices."""
    from sensor_reader import device_buffers
    return jsonify({
        "devices_in_buffer": list(device_buffers.keys()),
        "buffer_sizes": {k: len(v) for k, v in device_buffers.items()}
    }), 200


if __name__ == "__main__":
    thread = threading.Thread(target=processing_loop, daemon=True)
    thread.start()
    app.run(host="0.0.0.0", port=5000, debug=False)
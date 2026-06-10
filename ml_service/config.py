# Sensor readings API
SENSOR_API_URL  = "https://370c-129-222-187-199.ngrok-free.app/api/v1/sensor-readings?size=50&sort=receivedAt,desc"
SENSOR_API_TOKEN = "hydromap-ml-key-2026"  #the backend token

# Backend alert API
BACKEND_URL  = "https://370c-129-222-187-199.ngrok-free.app"
ML_TOKEN     = "hydromap-ml-key-2026"

# Polling interval in seconds
POLL_INTERVAL = 30  # matches the 30-minute timestep? adjust as needed

# Rolling window for feature engineering
WINDOW_SIZE = 8

# Expected normal pressure range in bar
EXPECTED_PRESSURE_MIN = 2.0
EXPECTED_PRESSURE_MAX = 6.0
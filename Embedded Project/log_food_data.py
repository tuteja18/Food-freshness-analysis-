import os
import time
import requests
import serial

# --- CONFIGURATION ---
ARDUINO_PORT = 'COM3'                       # Arduino Uno port
BAUD_RATE = 9600
CAMERA_URL = 'http://172.20.10.6/capture'  # Camera capture endpoint
LOG_DIR = r'C:\Users\lenovo\OneDrive\Desktop\Embedded Project'
LOG_INTERVAL_SEC = 60                       # Seconds between snapshots

CAM_TIMEOUT_SEC = 15      # wait longer for the ESP32-CAM (was 4)
CAM_RETRIES = 3           # attempts per snapshot
CAM_RETRY_DELAY_SEC = 3   # pause between attempts (lets the camera recover)

# Ensure output directories exist
os.makedirs(os.path.join(LOG_DIR, "images"), exist_ok=True)
csv_filepath = os.path.join(LOG_DIR, "sensor_log.csv")

# Create CSV header if the file does not exist
if not os.path.exists(csv_filepath):
    with open(csv_filepath, 'w') as f:
        f.write("timestamp,time_ms,mq135,mq3,ds18b20_temp_c,dht_temp_c,humidity_pct,image_filename\n")


def capture_image(session, url):
    """Try to fetch a valid JPEG. Returns (bytes or None, status string)."""
    status = "CAM_TIMEOUT"
    for attempt in range(1, CAM_RETRIES + 1):
        try:
            resp = session.get(url, timeout=CAM_TIMEOUT_SEC)
            data = resp.content
            # A real JPEG starts with FF D8 and is more than a few bytes
            if resp.status_code == 200 and len(data) > 2000 and data[:2] == b"\xff\xd8":
                return data, "OK"
            status = "CAM_ERR"
            print(f"  Camera attempt {attempt}/{CAM_RETRIES}: bad response "
                  f"(status {resp.status_code}, {len(data)} bytes)")
        except requests.exceptions.RequestException as e:
            status = "CAM_TIMEOUT"
            print(f"  Camera attempt {attempt}/{CAM_RETRIES} failed: {e}")
        if attempt < CAM_RETRIES:
            time.sleep(CAM_RETRY_DELAY_SEC)
    return None, status


def read_fresh_sensor_line(ser, max_tries=10):
    """Discard old buffered serial data, then return the next valid sensor line."""
    ser.reset_input_buffer()
    for _ in range(max_tries):
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        # 7 comma-separated values: time_ms,mq135,mq3,ds18b20,dht,humidity ... check >= 5 commas
        if line and ',' in line and not line.startswith("time_ms") and line.count(',') >= 5:
            return line
    return None


print(f"Connecting to Arduino on {ARDUINO_PORT}...")
ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=3)
time.sleep(2)  # Settle serial connection
print("Connected! Starting data collection. Press Ctrl+C to stop.\n")

session = requests.Session()
next_log_time = 0  # log immediately on first loop

try:
    while True:
        if time.time() < next_log_time:
            time.sleep(0.5)
            continue

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        img_name = f"img_{timestamp_str}.jpg"
        img_path = os.path.join(LOG_DIR, "images", img_name)

        # 1. Fetch camera snapshot (with retries)
        data, status = capture_image(session, CAMERA_URL)
        if data:
            with open(img_path, 'wb') as img_f:
                img_f.write(data)
            print(f"Captured: {img_name}")
        else:
            img_name = status  # CAM_TIMEOUT or CAM_ERR

        # 2. Read a FRESH sensor line (after the camera, so it isn't stale)
        line = read_fresh_sensor_line(ser)
        if line is None:
            print("No valid sensor line received, will retry shortly.")
            time.sleep(2)
            continue

        # 3. Append sensor line + snapshot reference to CSV
        log_entry = f"{timestamp_str},{line},{img_name}\n"
        with open(csv_filepath, 'a') as f:
            f.write(log_entry)
        print(f"Logged: {log_entry.strip()}")

        next_log_time = time.time() + LOG_INTERVAL_SEC

except KeyboardInterrupt:
    print("\nStopping data collection.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
    session.close()
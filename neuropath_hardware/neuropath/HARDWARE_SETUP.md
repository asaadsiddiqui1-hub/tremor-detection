# NeuroPath AI — Hardware & Firmware Setup

## The locked data format

Every part of this project depends on the ESP32 sending exactly this:

```
accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, ppg_raw
```

| Field | Unit | Source | Typical range |
|-------|------|--------|---------------|
| accel_x/y/z | m/s² | MPU6050 | -20 to +20 (Z rests near -9.81) |
| gyro_x/y/z | rad/s | MPU6050 | -5 to +5 |
| ppg_raw | integer IR count | MAX30102 | 1000s when finger on sensor |

- **Sample rate:** 50 Hz (stable, via `micros()` scheduling)
- **Baud:** 115200
- **Line ending:** newline (`\n`)

---

## Step 1 — Arduino libraries needed

Install via Arduino IDE → Tools → Manage Libraries:

1. **Adafruit MPU6050** (also pulls in Adafruit Unified Sensor)
2. **Adafruit BusIO** (dependency, usually auto-installed)
3. **SparkFun MAX3010x Pulse and Proximity Sensor Library**

---

## Step 2 — Flash the firmware

1. Open `firmware/neuropath_firmware.ino` in Arduino IDE
2. Board: **ESP32 Dev Module**
3. Confirm `#define USE_MAX30102 true` (line 28) — yours has the sensor
4. Upload

### What changed from your original .ino
| Your version | New version | Why |
|--------------|-------------|-----|
| `delay(10)` | `micros()` scheduling | `delay(10)` gave ~70-85 Hz with jitter; this gives a true, stable 50 Hz |
| No MAX30102 code | I2C MAX30102 with explicit config | The 7th column (PPG) now comes from real sensor setup |
| `setFilterBandwidth` only | + accel/gyro range set | Prevents clipping on big movements |
| 6 values printed | 7 values (adds ppg_raw) | Matches your actual hardware |

---

## Step 3 — Verify the stream

**Close the Arduino Serial Monitor first** (only one program can use the port).

```bash
pip install pyserial numpy
cd app
python check_stream.py            # lists your COM ports
python check_stream.py COM3       # run with your port
```

You want to see:
```
✅ Stream looks healthy. Ready to run the app.
```

If PPG reads low, press the sensor against your wrist/finger — loose
contact gives low IR counts (this is why your screenshot showed ~2000).

---

## Step 4 — Build the app without hardware (optional)

You don't need the glove connected to develop the UI. The
`core/serial_reader.py` module has a `SimulatedReader` that generates
synthetic 7-column data in four scenarios: normal, fall, tremor, stress.

```python
from core.serial_reader import SimulatedReader, SerialReader

# For testing:
reader = SimulatedReader(scenario='tremor')

# For real hardware (just swap this one line):
reader = SerialReader(port='COM3')
```

---

## Files in this folder

```
firmware/
└── neuropath_firmware.ino     ← flash this to ESP32

app/
├── check_stream.py            ← run first to verify hardware
└── core/
    └── serial_reader.py       ← shared reader used by all screens
```

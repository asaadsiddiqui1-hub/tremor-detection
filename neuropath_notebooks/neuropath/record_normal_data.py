"""
Normal Hand Data Recorder — NeuroPath AI
=========================================
Records raw MPU6050 data from the ESP32 glove and saves it as a CSV
with label=0 (normal), ready to merge with imu-hand-tremor-parkinsons.csv
for binary tremor classifier training.

Usage:
    python record_normal_data.py
    python record_normal_data.py --port COM5
    python record_normal_data.py --port COM3 --output data/my_normal_data.csv

Output columns (matches tremor training CSV format exactly):
    xAcc, yAcc, zAcc, xGyro, yGyro, zGyro, label
    (ppg_raw column from the glove is dropped automatically)
    Accel is converted from m/s² → g  (÷ 9.81) to match training data.
"""

import argparse
import csv
import os
import sys
import time
import serial

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_PORT    = "COM3"
BAUD_RATE       = 115200
SAMPLE_RATE_HZ  = 50
G               = 9.81           # m/s² → g
OUTPUT_PATH     = os.path.join(os.path.dirname(__file__), "data", "normal_hand_data.csv")

# Recording sessions — edit durations (seconds) as needed
SESSIONS = [
    ("Resting hand completely still on table",        180),   # 3 min
    ("Slow hand rotation (supination / pronation)",   150),   # 2.5 min
    ("Normal typing on keyboard",                     180),   # 3 min
    ("Pointing, reaching, picking up objects",        180),   # 3 min
    ("Walking with arm swinging naturally",           180),   # 3 min
]

HEADER = ["xAcc", "yAcc", "zAcc", "xGyro", "yGyro", "zGyro", "label"]


def countdown(seconds: int, activity: str):
    """Show a live countdown for the current activity."""
    print(f"\n  Activity: {activity}")
    print(f"  Duration: {seconds} seconds")
    print("  GET READY — starting in 3...", end="", flush=True)
    for i in (2, 1):
        time.sleep(1)
        print(f" {i}...", end="", flush=True)
    time.sleep(1)
    print(" GO!\n")

    start = time.time()
    while True:
        elapsed  = time.time() - start
        remaining = seconds - elapsed
        if remaining <= 0:
            break
        bar_len  = 40
        filled   = int(bar_len * elapsed / seconds)
        bar      = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {int(remaining):3d}s remaining", end="", flush=True)
        time.sleep(0.25)
    print(f"\r  [{'█' * 40}]   0s remaining  ✓")


def record_session(ser: serial.Serial, duration_s: int, writer: csv.writer) -> int:
    """Read serial data for duration_s seconds, write rows, return sample count."""
    deadline = time.time() + duration_s
    count    = 0
    while time.time() < deadline:
        try:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
        except Exception:
            continue

        parts = raw.split(",")
        if len(parts) != 7:      # expect 7 columns from firmware
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue

        # Convert accel m/s² → g  (columns 0-2)
        ax = vals[0] / G
        ay = vals[1] / G
        az = vals[2] / G
        # Firmware sends gyro in rad/s; training CSV uses deg/s — convert
        gx = vals[3] * (180.0 / 3.141592653589793)
        gy = vals[4] * (180.0 / 3.141592653589793)
        gz = vals[5] * (180.0 / 3.141592653589793)
        # ppg_raw (vals[6]) is dropped

        writer.writerow([
            round(ax, 6), round(ay, 6), round(az, 6),
            round(gx, 6), round(gy, 6), round(gz, 6),
            0              # label = normal
        ])
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Record normal hand data from ESP32 glove")
    parser.add_argument("--port",   default=DEFAULT_PORT, help="COM port (default: COM3)")
    parser.add_argument("--output", default=OUTPUT_PATH,  help="Output CSV path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Check if file already exists — ask before overwriting
    append_mode = False
    if os.path.exists(args.output):
        existing = sum(1 for _ in open(args.output)) - 1  # subtract header
        print(f"\n  File already exists: {args.output}")
        print(f"  It contains {existing} rows.")
        choice = input("  [A]ppend to it or [O]verwrite? ").strip().upper()
        append_mode = (choice == "A")

    print("\n" + "="*60)
    print("  NeuroPath AI — Normal Hand Data Recorder")
    print("="*60)
    print(f"  Port    : {args.port}")
    print(f"  Output  : {args.output}")
    print(f"  Rate    : {SAMPLE_RATE_HZ} Hz")
    total_s = sum(d for _, d in SESSIONS)
    print(f"  Total   : {total_s // 60} min {total_s % 60} s across {len(SESSIONS)} activities")
    print("="*60)

    print(f"\nConnecting to {args.port}...", end="", flush=True)
    try:
        ser = serial.Serial(args.port, BAUD_RATE, timeout=1)
        time.sleep(2)          # let ESP32 boot
        ser.reset_input_buffer()
        print(" OK")
    except serial.SerialException as e:
        print(f"\n  ERROR: {e}")
        print("  Make sure the glove is plugged in and the COM port is correct.")
        sys.exit(1)

    file_mode = "a" if append_mode else "w"
    write_header = not append_mode

    total_samples = 0
    try:
        with open(args.output, file_mode, newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(HEADER)

            for i, (activity, duration) in enumerate(SESSIONS, 1):
                print(f"\n── Session {i}/{len(SESSIONS)} ──────────────────────────────")
                countdown(duration, activity)
                n = record_session(ser, duration, writer)
                total_samples += n
                print(f"  Recorded {n} samples  (total so far: {total_samples})")
                f.flush()

                if i < len(SESSIONS):
                    input("\n  Press ENTER when you're ready for the next activity...")

    except KeyboardInterrupt:
        print("\n\n  Recording interrupted by user.")
    finally:
        ser.close()

    print("\n" + "="*60)
    print(f"  Done! Total samples recorded : {total_samples}")
    print(f"  Saved to                      : {os.path.abspath(args.output)}")
    print(f"  Duration at 50 Hz             : ~{total_samples // SAMPLE_RATE_HZ} seconds")
    print("="*60)
    print("\n  Next step: open notebook 02_tremor_classifier_train.ipynb")
    print("  and run the binary retrain cells at the bottom.")


if __name__ == "__main__":
    main()

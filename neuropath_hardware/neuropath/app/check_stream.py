"""
NeuroPath AI — Serial Stream Diagnostic
========================================
Run this FIRST after flashing the firmware, before anything else.
It verifies your ESP32 is sending the correct 7-column format at a
stable 50 Hz.

Usage:
    python check_stream.py COM3          (Windows)
    python check_stream.py /dev/ttyUSB0  (Linux)
    python check_stream.py               (lists available ports)
"""

import sys
import time

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("ERROR: pyserial not installed. Run:  pip install pyserial")
    sys.exit(1)


def list_ports():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found. Is the ESP32 plugged in?")
        return
    print("Available serial ports:")
    for p in ports:
        print(f"  {p.device:15s}  {p.description}")
    print("\nRun again with your port, e.g.:  python check_stream.py", ports[0].device)


def check_stream(port: str, baud: int = 115200, duration: float = 10.0):
    print(f"Opening {port} at {baud} baud...")
    try:
        ser = serial.Serial(port, baud, timeout=1.0)
    except serial.SerialException as e:
        print(f"ERROR: could not open {port}: {e}")
        print("  - Is the port correct? Run with no arguments to list ports.")
        print("  - Is the Arduino Serial Monitor still open? Close it first.")
        return

    time.sleep(2.0)        # ESP32 resets when port opens
    ser.reset_input_buffer()

    print(f"Listening for {duration:.0f} seconds...\n")
    n_good = 0
    n_bad = 0
    t0 = time.time()
    last_print = t0
    sample_example = None

    while time.time() - t0 < duration:
        raw = ser.readline().decode('utf-8', errors='ignore').strip()
        if not raw:
            continue
        parts = raw.split(',')
        if len(parts) == 7:
            try:
                vals = [float(p) for p in parts]
                n_good += 1
                if sample_example is None:
                    sample_example = vals
            except ValueError:
                n_bad += 1
        else:
            n_bad += 1

        # Print a status line every second
        now = time.time()
        if now - last_print >= 1.0:
            rate = n_good / (now - t0)
            print(f"  {n_good:5d} good lines | {n_bad:3d} bad | "
                  f"~{rate:.1f} Hz | latest: {raw[:45]}")
            last_print = now

    ser.close()

    elapsed = time.time() - t0
    rate = n_good / elapsed

    print("\n" + "=" * 55)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 55)
    print(f"  Good lines      : {n_good}")
    print(f"  Bad/malformed   : {n_bad}")
    print(f"  Average rate    : {rate:.1f} Hz  (target: 50 Hz)")

    if sample_example:
        labels = ['accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z','ppg']
        print(f"\n  Example sample:")
        for lbl, val in zip(labels, sample_example):
            print(f"    {lbl:9s}: {val}")

    print("\n  VERDICT:")
    if n_good == 0:
        print("    ❌ No valid data. Check firmware is flashed and format is correct.")
    elif n_bad > n_good * 0.1:
        print("    ⚠️  Too many bad lines. Check baud rate and wiring.")
    elif rate < 40:
        print(f"    ⚠️  Rate is {rate:.0f} Hz, below 50. Check delay/timing in firmware.")
    elif rate > 60:
        print(f"    ⚠️  Rate is {rate:.0f} Hz, above 50. Models expect 50 Hz.")
    else:
        print("    ✅ Stream looks healthy. Ready to run the app.")

        # Check PPG sanity
        if sample_example and sample_example[6] < 1000:
            print("    ℹ️  PPG value is low — make sure finger/wrist contacts the sensor.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        list_ports()
    else:
        check_stream(sys.argv[1])

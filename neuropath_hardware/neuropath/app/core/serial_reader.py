"""
NeuroPath AI — Serial Reader + Rolling Buffer
=============================================
Shared hardware interface used by EVERY screen in the app.

Reads the ESP32 stream:
    accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, ppg_raw
at 50 Hz over USB serial, and maintains a thread-safe rolling
window of the most recent N seconds of data.

Usage
-----
    reader = SerialReader(port='COM3', baud=115200)
    reader.start()
    ...
    window = reader.get_window(seconds=3)   # numpy array (N, 7)
    ...
    reader.stop()

For testing WITHOUT hardware, use SimulatedReader instead — it
generates synthetic 7-column data so you can build/test the UI
before the glove is connected.
"""

from __future__ import annotations
import threading
import time
import collections
import numpy as np

try:
    import serial  # pyserial
    HAS_PYSERIAL = True
except ImportError:
    HAS_PYSERIAL = False


# Column layout — the single source of truth for the whole app
COLUMNS = ['accel_x', 'accel_y', 'accel_z',
           'gyro_x', 'gyro_y', 'gyro_z', 'ppg_raw']
N_COLUMNS = len(COLUMNS)
SAMPLE_RATE_HZ = 50


class SerialReader:
    """Background-threaded reader for the ESP32 glove."""

    def __init__(self, port: str, baud: int = 115200,
                 buffer_seconds: float = 5.0,
                 sample_rate: int = SAMPLE_RATE_HZ):
        if not HAS_PYSERIAL:
            raise RuntimeError(
                "pyserial is not installed. Run: pip install pyserial"
            )
        self.port = port
        self.baud = baud
        self.sample_rate = sample_rate
        self.maxlen = int(buffer_seconds * sample_rate)

        self._buffer = collections.deque(maxlen=self.maxlen)
        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self._serial = None

        # Diagnostics
        self.lines_read = 0
        self.parse_errors = 0
        self.last_sample = None
        self.actual_rate = 0.0
        self._rate_counter = 0
        self._rate_t0 = time.time()

    # ── Lifecycle ─────────────────────────────────────────────────
    def start(self):
        if self._running:
            return
        self._serial = serial.Serial(self.port, self.baud, timeout=1.0)
        time.sleep(2.0)  # let ESP32 reset after opening port
        self._serial.reset_input_buffer()
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if getattr(self, '_serial', None) and self._serial.is_open:
            self._serial.close()

    # ── Background loop ───────────────────────────────────────────
    def _read_loop(self):
        while self._running:
            try:
                raw = self._serial.readline().decode('utf-8', errors='ignore').strip()
                if not raw:
                    continue
                parts = raw.split(',')
                if len(parts) != N_COLUMNS:
                    self.parse_errors += 1
                    continue
                sample = [float(p) for p in parts]
                with self._lock:
                    self._buffer.append(sample)
                self.last_sample = sample
                self.lines_read += 1

                # Measure actual sample rate once per second
                self._rate_counter += 1
                now = time.time()
                if now - self._rate_t0 >= 1.0:
                    self.actual_rate = self._rate_counter / (now - self._rate_t0)
                    self._rate_counter = 0
                    self._rate_t0 = now
            except (ValueError, UnicodeDecodeError):
                self.parse_errors += 1
            except Exception:
                if self._running:
                    time.sleep(0.1)

    # ── Data access ───────────────────────────────────────────────
    def get_window(self, seconds: float = 3.0) -> np.ndarray:
        """Return the most recent `seconds` of data as (N, 7) array."""
        n = int(seconds * self.sample_rate)
        with self._lock:
            data = list(self._buffer)[-n:]
        if not data:
            return np.zeros((0, N_COLUMNS))
        return np.array(data)

    def get_channel(self, col_name: str, seconds: float = 3.0) -> np.ndarray:
        """Return one channel (e.g. 'accel_x') over the window."""
        idx = COLUMNS.index(col_name)
        window = self.get_window(seconds)
        if window.shape[0] == 0:
            return np.zeros(0)
        return window[:, idx]

    def is_ready(self, seconds: float = 3.0) -> bool:
        """True once the buffer holds a full window."""
        with self._lock:
            return len(self._buffer) >= int(seconds * self.sample_rate)

    @property
    def buffer_fill(self) -> float:
        with self._lock:
            return len(self._buffer) / self.maxlen


class SimulatedReader(SerialReader):
    """
    Drop-in replacement that generates synthetic 7-column data.
    Lets you build & test the entire app with no hardware attached.
    """

    def __init__(self, buffer_seconds: float = 5.0,
                 sample_rate: int = SAMPLE_RATE_HZ,
                 scenario: str = 'normal'):
        # Skip parent __init__ (no pyserial needed)
        self.sample_rate = sample_rate
        self.maxlen = int(buffer_seconds * sample_rate)
        self._buffer = collections.deque(maxlen=self.maxlen)
        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self.scenario = scenario       # 'normal', 'fall', 'tremor', 'stress'
        self.lines_read = 0
        self.parse_errors = 0
        self.last_sample = None
        self.actual_rate = float(sample_rate)
        self._t = 0.0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._thread.start()

    def _sim_loop(self):
        dt = 1.0 / self.sample_rate
        while self._running:
            self._t += dt
            sample = self._generate_sample(self._t)
            with self._lock:
                self._buffer.append(sample)
            self.last_sample = sample
            self.lines_read += 1
            time.sleep(dt)

    def _generate_sample(self, t: float) -> list:
        g = 9.81
        # Baseline: hand at rest, gravity on Z, ~72 bpm pulse
        ppg = 2000 + 300 * np.sin(2 * np.pi * 1.2 * t)  # ~72 bpm

        if self.scenario == 'normal':
            ax = np.random.normal(0, 0.2)
            ay = np.random.normal(0, 0.2)
            az = np.random.normal(-g, 0.2)
            gx, gy, gz = np.random.normal(0, 0.05, 3)
        elif self.scenario == 'fall':
            # Sudden spike every ~5 seconds
            phase = t % 5.0
            spike = 25 if 2.0 < phase < 2.3 else 0
            ax = np.random.normal(spike, 1.0)
            ay = np.random.normal(spike, 1.0)
            az = np.random.normal(-g + spike, 1.0)
            gx, gy, gz = np.random.normal(0, 2.0, 3)
        elif self.scenario == 'tremor':
            # 5 Hz Parkinsonian resting tremor
            tremor = 2.0 * np.sin(2 * np.pi * 5.0 * t)
            ax = tremor + np.random.normal(0, 0.1)
            ay = 0.5 * tremor + np.random.normal(0, 0.1)
            az = -g + np.random.normal(0, 0.1)
            gx, gy, gz = np.random.normal(0, 0.3, 3)
        elif self.scenario == 'stress':
            # Fast, uniform pulse ~95 bpm, minimal movement
            ppg = 2000 + 300 * np.sin(2 * np.pi * 1.58 * t)
            ax = np.random.normal(0, 0.05)
            ay = np.random.normal(0, 0.05)
            az = np.random.normal(-g, 0.05)
            gx, gy, gz = np.random.normal(0, 0.02, 3)
        else:
            ax = ay = 0.0
            az = -g
            gx = gy = gz = 0.0

        return [ax, ay, az, gx, gy, gz, ppg]


# ── Standalone test ────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing SimulatedReader (no hardware needed)...")
    reader = SimulatedReader(scenario='tremor')
    reader.start()
    time.sleep(3.5)

    window = reader.get_window(seconds=3)
    print(f"  Window shape : {window.shape}  (expect ~150 rows x 7 cols)")
    print(f"  Columns      : {COLUMNS}")
    print(f"  Buffer fill  : {reader.buffer_fill*100:.0f}%")
    print(f"  Lines read   : {reader.lines_read}")
    print(f"  Last sample  : {[round(v,2) for v in reader.last_sample]}")
    print(f"  Ready (3s)   : {reader.is_ready(3)}")
    reader.stop()
    print("\nSerialReader module OK.")

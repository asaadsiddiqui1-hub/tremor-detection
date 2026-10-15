"""
NeuroPath AI — Serial Reader (threading, no PyQt5)
====================================================
Provides SerialReader (real ESP32) and SimulatedReader (synthetic data).
Both expose the same interface so serial_bridge.py can swap them transparently.

Column contract (ESP32 sends at 50 Hz, 115200 baud):
  0=accel_x (m/s²)  1=accel_y (m/s²)  2=accel_z (m/s²)
  3=gyro_x (rad/s)  4=gyro_y (rad/s)  5=gyro_z (rad/s)
  6=ppg_raw (int)
"""

import threading
import collections
import time
import numpy as np

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

COLUMNS   = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'ppg_raw']
N_COLUMNS = 7
_DEFAULT_BAUD = 115200


# ── Base reader ────────────────────────────────────────────────────────────────

class _BaseReader:
    """Shared ring-buffer logic. Subclasses implement `run()`."""

    def __init__(self, buffer_seconds: float = 65.0, sample_rate: int = 50):
        self.sample_rate   = sample_rate
        self._maxlen       = int(buffer_seconds * sample_rate)
        self._buffer       = collections.deque(maxlen=self._maxlen)
        self._lock         = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running      = False
        self.lines_read    = 0
        self.parse_errors  = 0
        self.last_sample   = None          # list[float] of the most recent row
        self.actual_rate   = float(sample_rate)

    # ── Thread control ─────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def run(self):
        raise NotImplementedError

    # ── Buffer helpers ─────────────────────────────────────────────────────────

    @property
    def buffer_fill(self) -> float:
        """Fraction of ring-buffer that is filled (0.0 – 1.0)."""
        with self._lock:
            return len(self._buffer) / max(1, self._maxlen)

    def buffer_length(self) -> int:
        with self._lock:
            return len(self._buffer)

    def get_window(self, seconds: float) -> np.ndarray:
        """
        Return the last `seconds` worth of samples as (N, 7) float32.
        Zero-pads at the front if the buffer is not yet full.
        """
        n = int(seconds * self.sample_rate)
        with self._lock:
            data = list(self._buffer)
        if not data:
            return np.zeros((n, N_COLUMNS), dtype=np.float32)
        arr = np.array(data, dtype=np.float32)
        if len(arr) >= n:
            return arr[-n:]
        pad = np.zeros((n - len(arr), N_COLUMNS), dtype=np.float32)
        return np.vstack([pad, arr])

    def _append(self, row: np.ndarray):
        with self._lock:
            self._buffer.append(row)
        self.last_sample = row.tolist()
        self.lines_read += 1


# ── Real serial reader ─────────────────────────────────────────────────────────

class SerialReader(_BaseReader):
    """Reads a real ESP32 over USB serial."""

    def __init__(self, port: str = "COM3", baud: int = _DEFAULT_BAUD,
                 buffer_seconds: float = 65.0, sample_rate: int = 50):
        super().__init__(buffer_seconds, sample_rate)
        self.port = port
        self.baud = baud

    def run(self):
        if not SERIAL_AVAILABLE:
            return
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.5)
            t0, count = time.time(), 0
            while self._running:
                try:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                except Exception:
                    break
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != N_COLUMNS:
                    self.parse_errors += 1
                    continue
                try:
                    row = np.array([float(x) for x in parts], dtype=np.float32)
                    self._append(row)
                    count += 1
                    elapsed = time.time() - t0
                    if elapsed >= 2.0:
                        self.actual_rate = count / elapsed
                        count, t0 = 0, time.time()
                except ValueError:
                    self.parse_errors += 1
            ser.close()
        except Exception:
            pass

    @staticmethod
    def available_ports():
        if not SERIAL_AVAILABLE:
            return []
        return [p.device for p in serial.tools.list_ports.comports()]


# ── Simulated reader ────────────────────────────────────────────────────────────

class SimulatedReader(_BaseReader):
    """
    Generates synthetic 50 Hz sensor data for development without hardware.

    Scenarios
    ---------
    'normal'  – quiet baseline motion
    'fall'    – spike at t ≈ 5 s in every 10-second cycle
    'tremor'  – 4.5 Hz sinusoidal tremor on accel axes
    'stress'  – normal motion but elevated PPG heart rate (90 bpm)
    """

    def __init__(self, buffer_seconds: float = 65.0,
                 sample_rate: int = 50, scenario: str = "normal"):
        super().__init__(buffer_seconds, sample_rate)
        self.scenario = scenario
        self._rng = np.random.RandomState(42)

    def run(self):
        dt = 1.0 / self.sample_rate
        t = 0.0
        t0, count = time.time(), 0
        while self._running:
            row = _gen_row(t, self.scenario, self._rng)
            self._append(row)
            count += 1
            elapsed = time.time() - t0
            if elapsed >= 2.0:
                self.actual_rate = count / elapsed
                count, t0 = 0, time.time()
            t += dt
            time.sleep(dt)


# ── Synthetic data generator ──────────────────────────────────────────────────

def _gen_row(t: float, scenario: str, rng: np.random.RandomState) -> np.ndarray:
    if scenario == "fall":
        phase = t % 10.0
        spike = 18.0 if 4.8 < phase < 5.2 else 0.0
        ax = float(rng.normal(0, 0.3)) + spike * 0.5
        ay = float(rng.normal(0, 0.3)) + spike * 0.5
        az = 9.81 + float(rng.normal(0, 0.3)) - spike
        gx, gy, gz = rng.normal(0, 0.5, 3).tolist()

    elif scenario == "tremor":
        freq, amp = 4.5, 0.9
        ax = amp * np.sin(2 * np.pi * freq * t) + float(rng.normal(0, 0.05))
        ay = amp * 0.4 * np.sin(2 * np.pi * freq * t + 0.7) + float(rng.normal(0, 0.05))
        az = 9.81 + float(rng.normal(0, 0.1))
        gx = amp * 0.5 * np.sin(2 * np.pi * freq * t + 1.0)
        gy, gz = rng.normal(0, 0.05, 2).tolist()

    else:   # normal / stress — same motion, stress differs only in PPG rate
        ax, ay = float(rng.normal(0, 0.25)), float(rng.normal(0, 0.25))
        az = 9.81 + float(rng.normal(0, 0.15))
        gx, gy, gz = rng.normal(0, 0.04, 3).tolist()

    hr_bpm = 90.0 if scenario == "stress" else 72.0
    ppg = int(50000 + 20000 * np.sin(2 * np.pi * (hr_bpm / 60.0) * t)
              + float(rng.normal(0, 300)))
    ppg = int(np.clip(ppg, 10000, 100000))

    return np.array([ax, ay, az, gx, gy, gz, ppg], dtype=np.float32)

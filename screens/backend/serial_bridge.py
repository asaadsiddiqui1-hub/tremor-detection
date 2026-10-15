"""
NeuroPath AI — Serial Bridge (singleton)
=========================================
Wraps the existing SerialReader / SimulatedReader into a single
process-wide instance shared by all WebSocket routes.
"""

import sys
import os
import threading

from config import (
    HARDWARE_CORE, SAMPLE_RATE_HZ, BUFFER_SECONDS,
    DEFAULT_COM_PORT, BAUD_RATE
)

# ── Import the existing hardware core ─────────────────────────────────────────
sys.path.insert(0, HARDWARE_CORE)
from serial_reader import SerialReader, SimulatedReader, COLUMNS, N_COLUMNS   # noqa: E402

# ── Singleton state ────────────────────────────────────────────────────────────
_reader = None
_lock   = threading.Lock()
_mode   = "sim"          # "sim" | "live"
_port   = DEFAULT_COM_PORT


def get_reader():
    """Return the active reader (SimulatedReader or SerialReader)."""
    global _reader
    with _lock:
        if _reader is None:
            _reader = _make_sim_reader()
    return _reader


def _make_sim_reader(scenario: str = "normal") -> SimulatedReader:
    r = SimulatedReader(
        buffer_seconds=BUFFER_SECONDS,
        sample_rate=SAMPLE_RATE_HZ,
        scenario=scenario,
    )
    r.start()
    return r


def connect_live(port: str) -> dict:
    """Switch to a real ESP32 serial connection. Returns status dict."""
    global _reader, _mode, _port
    with _lock:
        if _reader is not None:
            _reader.stop()
        try:
            r = SerialReader(
                port=port,
                baud=BAUD_RATE,
                buffer_seconds=BUFFER_SECONDS,
                sample_rate=SAMPLE_RATE_HZ,
            )
            r.start()
            _reader = r
            _mode   = "live"
            _port   = port
            return {"ok": True, "mode": "live", "port": port}
        except Exception as e:
            # Fall back to sim
            _reader = _make_sim_reader()
            _mode   = "sim"
            return {"ok": False, "error": str(e), "mode": "sim"}


def connect_sim(scenario: str = "normal") -> dict:
    """Switch to simulated data."""
    global _reader, _mode
    with _lock:
        if _reader is not None:
            _reader.stop()
        _reader = _make_sim_reader(scenario)
        _mode   = "sim"
    return {"ok": True, "mode": "sim", "scenario": scenario}


def get_status() -> dict:
    r = get_reader()
    return {
        "mode":        _mode,
        "port":        _port if _mode == "live" else "SIM",
        "sample_rate": SAMPLE_RATE_HZ,
        "buffer_fill": round(r.buffer_fill * 100, 1),
        "lines_read":  r.lines_read,
        "parse_errors":r.parse_errors,
        "actual_rate": getattr(r, "actual_rate", SAMPLE_RATE_HZ),
        "columns":     COLUMNS,
    }


def list_ports() -> list[str]:
    """List available serial ports on this machine."""
    try:
        import serial.tools.list_ports
        return [p.device for p in serial.tools.list_ports.comports()]
    except Exception:
        return []

"""
NeuroPath AI — PyQt5 Desktop Application
=========================================
Run:  python app/main.py          (prompts for serial port or simulator)
      python app/main.py --sim    (start immediately with simulated data)

Requires:  PyQt5  pyqtgraph  numpy  scipy  pandas  scikit-learn  tensorflow
"""

import sys
import os
import json
import pickle
import argparse
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QFrame, QProgressBar, QLineEdit, QDialog, QComboBox,
    QSizePolicy, QSpinBox, QDoubleSpinBox, QStatusBar, QGroupBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui  import QFont, QColor

try:
    import pyqtgraph as pg
    pg.setConfigOption("background", "#0d1117")
    pg.setConfigOption("foreground", "#8b949e")
    HAS_PG = True
except ImportError:
    HAS_PG = False

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ── Path setup ────────────────────────────────────────────────────────────────
_APP_DIR    = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR   = os.path.dirname(_APP_DIR)
MODELS_DIR  = os.path.join(_BASE_DIR, "models")
sys.path.insert(0, _APP_DIR)

from core.serial_reader   import SerialReader, SimulatedReader, FS, N_COLS
from core.feature_adapter import (
    adapt_fall, adapt_tremor, adapt_stress, adapt_atrophy, adapt_pt_form,
    adapt_sensor, SENSOR_CHANNEL_LABELS,
    WIN_FALL, WIN_TREMOR, WIN_STRESS, WIN_PT, WIN_SENSOR,
    TREMOR_FEATURES, HRV_FEATURES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════════════════════════════════════════

DARK_STYLE = """
QMainWindow, QDialog { background: #0d1117; }
QWidget     { background: #0d1117; color: #e6edf3;
               font-family: 'Segoe UI', sans-serif; font-size: 13px; }

/* Cards / panels */
QFrame#card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px;
}
QFrame#card:hover { border: 1px solid #58a6ff; }

/* Buttons */
QPushButton {
    background: #21262d; color: #e6edf3;
    border: 1px solid #30363d; border-radius: 6px;
    padding: 6px 16px;
}
QPushButton:hover   { background: #30363d; border-color: #58a6ff; }
QPushButton:pressed { background: #388bfd22; }
QPushButton#primary {
    background: #1f6feb; border-color: #388bfd; color: #ffffff;
}
QPushButton#primary:hover { background: #388bfd; }
QPushButton#danger  {
    background: #da3633; border-color: #f85149; color: #ffffff;
}

/* Input fields */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 4px; padding: 4px 8px; color: #e6edf3;
}
QLineEdit:focus, QComboBox:focus { border-color: #58a6ff; }

/* Progress bar */
QProgressBar {
    background: #21262d; border: 1px solid #30363d;
    border-radius: 4px; height: 10px; text-align: center;
}
QProgressBar::chunk { background: #1f6feb; border-radius: 3px; }

/* Status bar */
QStatusBar { background: #161b22; border-top: 1px solid #30363d;
              color: #8b949e; font-size: 11px; }

/* Group box */
QGroupBox {
    border: 1px solid #30363d; border-radius: 6px;
    margin-top: 8px; padding-top: 6px; color: #8b949e;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; }

/* Labels */
QLabel#big    { font-size: 40px; font-weight: bold; }
QLabel#medium { font-size: 20px; font-weight: 600; }
QLabel#small  { color: #8b949e; font-size: 11px; }
QLabel#metric { font-size: 14px; font-family: 'Consolas', monospace; }
"""


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_json(name):
    path = os.path.join(MODELS_DIR, name)
    with open(path) as f:
        return json.load(f)

def _load_pkl(name):
    path = os.path.join(MODELS_DIR, name)
    with open(path, "rb") as f:
        return pickle.load(f)

def _load_keras(name):
    if not HAS_TF:
        raise RuntimeError("TensorFlow not installed")
    from tensorflow.keras.models import load_model
    return load_model(os.path.join(MODELS_DIR, name))

def _make_plot(title="", y_label="", color="#58a6ff"):
    if not HAS_PG:
        lbl = QLabel(f"[pyqtgraph not installed — pip install pyqtgraph]\n{title}")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color:#8b949e;")
        return lbl, None
    pw = pg.PlotWidget(title=title)
    pw.setLabel("left", y_label)
    pw.getAxis("left").setTextPen("#8b949e")
    pw.getAxis("bottom").setTextPen("#8b949e")
    pw.showGrid(x=True, y=True, alpha=0.15)
    curve = pw.plot(pen=pg.mkPen(color, width=1.5))
    return pw, curve


# ═══════════════════════════════════════════════════════════════════════════════
# BASE SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

class BaseScreen(QWidget):
    go_home = pyqtSignal()

    def __init__(self, title, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── top bar ──────────────────────────────────────────────────────────
        bar = QFrame()
        bar.setFixedHeight(48)
        bar.setStyleSheet("background:#161b22; border-bottom:1px solid #30363d;")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(12, 0, 12, 0)

        back = QPushButton("← Home")
        back.setFixedWidth(90)
        back.clicked.connect(self.go_home)
        bl.addWidget(back)

        lbl = QLabel(title)
        lbl.setFont(QFont("Segoe UI", 14, QFont.Bold))
        lbl.setAlignment(Qt.AlignCenter)
        bl.addWidget(lbl, 1)

        self.conn_dot = QLabel("● SIM")
        self.conn_dot.setStyleSheet("color:#3fb950; font-size:11px;")
        bl.addWidget(self.conn_dot)

        root.addWidget(bar)

        # ── content area (subclasses fill this) ──────────────────────────────
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(16, 12, 16, 12)
        root.addWidget(self.content, 1)

        self._timer = QTimer(self)
        self._reader = None
        self._model_loaded = False

    def attach_reader(self, reader):
        self._reader = reader

    def update_conn_label(self, connected, msg):
        if connected:
            self.conn_dot.setText("● " + msg[:20])
            self.conn_dot.setStyleSheet("color:#3fb950; font-size:11px;")
        else:
            self.conn_dot.setText("✕ " + msg[:20])
            self.conn_dot.setStyleSheet("color:#f85149; font-size:11px;")

    def on_activate(self):
        """Called when this screen becomes visible."""
        if not self._model_loaded:
            self._load_model()
            self._model_loaded = True
        self._timer.start()

    def on_deactivate(self):
        """Called when leaving this screen."""
        self._timer.stop()

    def _load_model(self):
        pass  # override in subclasses


# ═══════════════════════════════════════════════════════════════════════════════
# HOME SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

class HomeScreen(QWidget):
    navigate = pyqtSignal(int)   # emits screen index

    CARDS = [
        ("🫸", "Fall Detection",      "1D-CNN · 10.2 s window",   "#f85149", True),
        ("🤝", "Tremor Severity",     "RandomForest · 2 s window","#d29922", True),
        ("💓", "Stress / HRV",        "RandomForest · 60 s PPG",  "#58a6ff", True),
        ("💪", "Muscle Atrophy",      "GradBoost · live + input", "#3fb950", True),
        ("🏃", "PT Form",             "LSTM · 2 s window",        "#bc8cff", True),
        ("✋", "Gesture Translator",  "Not built — see nb 05/06", "#8b949e", False),
        ("📡", "Live Sensor",         "Raw 7-channel stream",     "#e07b39", True),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        title = QLabel("NeuroPath AI")
        title.setFont(QFont("Segoe UI", 28, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        sub = QLabel("Wearable Diagnostic Glove — Select a Feature")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color:#8b949e; font-size:13px;")
        root.addWidget(sub)

        grid = QGridLayout()
        grid.setSpacing(12)
        for idx, (icon, name, desc, color, ready) in enumerate(self.CARDS):
            card = self._make_card(idx, icon, name, desc, color, ready)
            grid.addWidget(card, idx // 4, idx % 4)
        root.addLayout(grid)
        root.addStretch()

        footer = QLabel("Simulated data active — connect ESP32 to switch to live mode")
        footer.setAlignment(Qt.AlignCenter)
        footer.setObjectName("small")
        footer.setStyleSheet("color:#8b949e; font-size:11px;")
        root.addWidget(footer)
        self._footer = footer

    def set_connection(self, connected, msg):
        if connected and "SIM" not in msg.upper():
            self._footer.setText(f"Live serial: {msg}")
        elif connected:
            self._footer.setText("Simulated data active — connect ESP32 to switch to live mode")
        else:
            self._footer.setText(f"Not connected: {msg}")

    def _make_card(self, idx, icon, name, desc, color, ready):
        frame = QFrame()
        frame.setObjectName("card")
        frame.setFixedSize(200, 130)
        frame.setCursor(Qt.PointingHandCursor if ready else Qt.ForbiddenCursor)

        lay = QVBoxLayout(frame)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(4)

        ic = QLabel(icon)
        ic.setFont(QFont("Segoe UI", 28))
        ic.setAlignment(Qt.AlignCenter)
        lay.addWidget(ic)

        nm = QLabel(name)
        nm.setFont(QFont("Segoe UI", 11, QFont.Bold))
        nm.setAlignment(Qt.AlignCenter)
        nm.setStyleSheet(f"color:{color};")
        lay.addWidget(nm)

        ds = QLabel(desc)
        ds.setFont(QFont("Segoe UI", 9))
        ds.setAlignment(Qt.AlignCenter)
        ds.setStyleSheet("color:#8b949e;")
        ds.setWordWrap(True)
        lay.addWidget(ds)

        if ready:
            frame.mousePressEvent = lambda _e, i=idx: self.navigate.emit(i + 1)
        return frame


# ═══════════════════════════════════════════════════════════════════════════════
# 1 — FALL DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class FallScreen(BaseScreen):

    def __init__(self, parent=None):
        super().__init__("Fall Detection", parent)
        self._model = None
        self._threshold = 0.65
        self._alert_active = False

        cl = self.content_layout
        cl.setSpacing(10)

        # ── prediction panel ─────────────────────────────────────────────────
        top = QHBoxLayout()

        left = QGroupBox("Prediction")
        ll   = QVBoxLayout(left)
        self.pred_lbl = QLabel("NORMAL")
        self.pred_lbl.setObjectName("big")
        self.pred_lbl.setAlignment(Qt.AlignCenter)
        self.pred_lbl.setStyleSheet("color:#3fb950;")
        ll.addWidget(self.pred_lbl)

        self.conf_lbl = QLabel("Confidence: —")
        self.conf_lbl.setAlignment(Qt.AlignCenter)
        self.conf_lbl.setObjectName("medium")
        ll.addWidget(self.conf_lbl)

        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        ll.addWidget(self.conf_bar)

        self.buf_lbl = QLabel("Buffer: 0 / 512 samples")
        self.buf_lbl.setAlignment(Qt.AlignCenter)
        self.buf_lbl.setObjectName("small")
        ll.addWidget(self.buf_lbl)
        left.setFixedWidth(240)
        top.addWidget(left)

        # ── live plot ─────────────────────────────────────────────────────────
        self.plot_w, self.curve = _make_plot(
            "Accel Magnitude (z-scored)", "amplitude", "#58a6ff")
        top.addWidget(self.plot_w, 1)
        cl.addLayout(top, 1)

        # ── info bar ─────────────────────────────────────────────────────────
        info = QLabel(
            "Model: fall_detection_cnn.h5  ·  Window: 512 samples (10.24 s)  ·  "
            "Threshold: 0.65  ·  Missing a fall > false alarm"
        )
        info.setObjectName("small")
        cl.addWidget(info)

        self._timer.setInterval(500)
        self._timer.timeout.connect(self._infer)

    def _load_model(self):
        try:
            self._model = _load_keras("fall_detection_cnn.h5")
            cfg = _load_json("fall_model_config.json")
            self._threshold = cfg.get("threshold", 0.65)
        except Exception as e:
            self.pred_lbl.setText("ERR")
            self.pred_lbl.setStyleSheet("color:#8b949e;")
            self.conf_lbl.setText(str(e)[:60])

    def _infer(self):
        if self._reader is None or self._model is None:
            return
        buf = self._reader.get_window(WIN_FALL)
        have = self._reader.buffer_length()
        self.buf_lbl.setText(f"Buffer: {min(have, WIN_FALL)} / {WIN_FALL} samples")

        x    = adapt_fall(buf).reshape(1, WIN_FALL, 1)
        prob = float(self._model.predict(x, verbose=0)[0][0])
        conf = int(prob * 100)
        fall = prob >= self._threshold

        self.conf_bar.setValue(conf)
        if fall:
            self.pred_lbl.setText("⚠  FALL")
            self.pred_lbl.setStyleSheet("color:#f85149; font-size:40px; font-weight:bold;")
            self.conf_lbl.setText(f"Confidence: {conf}%")
            self.conf_lbl.setStyleSheet("color:#f85149;")
        else:
            self.pred_lbl.setText("NORMAL")
            self.pred_lbl.setStyleSheet("color:#3fb950; font-size:40px; font-weight:bold;")
            self.conf_lbl.setText(f"Confidence (fall): {conf}%")
            self.conf_lbl.setStyleSheet("color:#e6edf3;")

        if self.curve and HAS_PG:
            accel = buf[:, :3]
            mag   = np.sqrt(np.sum(accel ** 2, axis=1))
            mu, s = mag.mean(), mag.std()
            if s > 1e-6:
                mag = (mag - mu) / s
            self.curve.setData(mag)


# ═══════════════════════════════════════════════════════════════════════════════
# 2 — TREMOR SEVERITY
# ═══════════════════════════════════════════════════════════════════════════════

class TremorScreen(BaseScreen):

    COLORS = {"mild": "#3fb950", "moderate": "#d29922", "high": "#f85149"}

    def __init__(self, parent=None):
        super().__init__("Tremor Severity", parent)
        self._clf = self._scaler = self._le = None

        cl = self.content_layout
        cl.setSpacing(10)

        top = QHBoxLayout()

        # ── prediction panel ─────────────────────────────────────────────────
        left = QGroupBox("Prediction")
        ll   = QVBoxLayout(left)
        self.pred_lbl = QLabel("—")
        self.pred_lbl.setObjectName("big")
        self.pred_lbl.setAlignment(Qt.AlignCenter)
        ll.addWidget(self.pred_lbl)
        self.conf_lbl = QLabel("Confidence: —")
        self.conf_lbl.setAlignment(Qt.AlignCenter)
        self.conf_lbl.setObjectName("medium")
        ll.addWidget(self.conf_lbl)
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        ll.addWidget(self.conf_bar)

        # band power metrics
        self.metrics = QLabel("")
        self.metrics.setObjectName("metric")
        self.metrics.setAlignment(Qt.AlignLeft)
        ll.addWidget(self.metrics)
        left.setFixedWidth(240)
        top.addWidget(left)

        # ── live plot ─────────────────────────────────────────────────────────
        self.plot_w, self.curve = _make_plot(
            "Dominant Accel Axis (g)", "g", "#d29922")
        top.addWidget(self.plot_w, 1)
        cl.addLayout(top, 1)

        info = QLabel(
            "Model: tremor_classifier.pkl (RandomForest)  ·  Window: 100 samples (2 s)  ·  "
            "Features: 15 FFT  ·  Accel m/s² → g before extraction"
        )
        info.setObjectName("small")
        cl.addWidget(info)

        self._timer.setInterval(500)
        self._timer.timeout.connect(self._infer)

    def _load_model(self):
        try:
            self._clf    = _load_pkl("tremor_classifier.pkl")
            self._scaler = _load_pkl("tremor_scaler.pkl")
            self._le     = _load_pkl("tremor_label_encoder.pkl")
        except Exception as e:
            self.pred_lbl.setText("ERR")
            self.conf_lbl.setText(str(e)[:60])

    def _infer(self):
        if self._reader is None or self._clf is None:
            return
        buf  = self._reader.get_window(WIN_TREMOR)
        feat = adapt_tremor(buf).reshape(1, -1)
        feat_s = self._scaler.transform(feat)
        idx    = self._clf.predict(feat_s)[0]
        proba  = self._clf.predict_proba(feat_s)[0]
        label  = self._le.classes_[idx]
        severity = {0: "mild", 1: "moderate", 2: "high"}.get(label,
                    ["mild","moderate","high"][idx] if idx < 3 else "mild")
        # map raw label int (1/2/3) to name
        from core.feature_adapter import TREMOR_FEATURES
        cfg_names = ["mild","moderate","high"]
        # use label encoder classes directly
        class_name = str(self._le.classes_[idx])
        severity   = {1:"mild",2:"moderate",3:"high"}.get(int(class_name), class_name)

        conf = int(proba.max() * 100)
        color = self.COLORS.get(severity, "#e6edf3")

        self.pred_lbl.setText(severity.upper())
        self.pred_lbl.setStyleSheet(f"color:{color}; font-size:40px; font-weight:bold;")
        self.conf_lbl.setText(f"Confidence: {conf}%")
        self.conf_bar.setValue(conf)

        # show key band metrics
        raw = adapt_tremor(buf)
        f_names = TREMOR_FEATURES
        p_park = raw[f_names.index("ratio_parkinson")] * 100
        p_ess  = raw[f_names.index("ratio_essential")] * 100
        dom_f  = raw[f_names.index("dominant_freq")]
        self.metrics.setText(
            f"Dominant freq : {dom_f:.1f} Hz\n"
            f"Parkinsonian  : {p_park:.1f} %\n"
            f"Essential     : {p_ess:.1f} %"
        )

        if self.curve and HAS_PG:
            from core.feature_adapter import IDX_AX, IDX_AZ, G
            accel_g = buf[:, IDX_AX:IDX_AZ+1] / G
            var = [np.var(accel_g[:,i]-np.mean(accel_g[:,i])) for i in range(3)]
            self.curve.setData(accel_g[:, int(np.argmax(var))])


# ═══════════════════════════════════════════════════════════════════════════════
# 3 — STRESS / HRV
# ═══════════════════════════════════════════════════════════════════════════════

class StressScreen(BaseScreen):

    COLORS = {"baseline": "#58a6ff", "relaxed": "#3fb950", "stress": "#f85149"}

    def __init__(self, parent=None):
        super().__init__("Stress / HRV", parent)
        self._clf = self._le = None

        cl = self.content_layout
        cl.setSpacing(10)

        top = QHBoxLayout()

        # ── prediction panel ─────────────────────────────────────────────────
        left = QGroupBox("Prediction")
        ll   = QVBoxLayout(left)
        self.pred_lbl = QLabel("—")
        self.pred_lbl.setObjectName("big")
        self.pred_lbl.setAlignment(Qt.AlignCenter)
        ll.addWidget(self.pred_lbl)
        self.conf_lbl = QLabel("Confidence: —")
        self.conf_lbl.setAlignment(Qt.AlignCenter)
        self.conf_lbl.setObjectName("medium")
        ll.addWidget(self.conf_lbl)
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        ll.addWidget(self.conf_bar)

        self.hrv_lbl = QLabel("")
        self.hrv_lbl.setObjectName("metric")
        ll.addWidget(self.hrv_lbl)

        self.buf_lbl = QLabel("Accumulating 60 s of PPG…")
        self.buf_lbl.setObjectName("small")
        self.buf_lbl.setAlignment(Qt.AlignCenter)
        ll.addWidget(self.buf_lbl)
        self.buf_bar = QProgressBar()
        self.buf_bar.setRange(0, WIN_STRESS)
        ll.addWidget(self.buf_bar)

        left.setFixedWidth(240)
        top.addWidget(left)

        # ── PPG plot ──────────────────────────────────────────────────────────
        self.plot_w, self.curve = _make_plot("PPG Signal (last 200 samples)",
                                             "raw count", "#e07b39")
        top.addWidget(self.plot_w, 1)
        cl.addLayout(top, 1)

        info = QLabel(
            "Model: stress_classifier.pkl (RandomForest)  ·  Window: 60 s at 50 Hz  ·  "
            "Features: 8 HRV  ·  Needs 60 s buffer before first inference"
        )
        info.setObjectName("small")
        cl.addWidget(info)

        self._timer.setInterval(2000)
        self._timer.timeout.connect(self._infer)

    def _load_model(self):
        try:
            self._clf = _load_pkl("stress_classifier.pkl")
            self._le  = _load_pkl("stress_label_encoder.pkl")
        except Exception as e:
            self.pred_lbl.setText("ERR")
            self.conf_lbl.setText(str(e)[:60])

    def _infer(self):
        if self._reader is None or self._clf is None:
            return

        have = self._reader.buffer_length()
        self.buf_bar.setValue(min(have, WIN_STRESS))
        pct  = min(have, WIN_STRESS) * 100 // WIN_STRESS
        self.buf_lbl.setText(f"Buffer: {pct}%  ({have}/{WIN_STRESS} samples)")

        # update rolling PPG display regardless of whether we can infer
        ppg_buf = self._reader.get_window(200)
        if self.curve and HAS_PG:
            self.curve.setData(ppg_buf[:, 6].astype(float))

        if have < WIN_STRESS:
            return

        buf   = self._reader.get_window(WIN_STRESS)
        feat  = adapt_stress(buf).reshape(1, -1)
        idx   = self._clf.predict(feat)[0]
        proba = self._clf.predict_proba(feat)[0]
        label = str(self._le.classes_[idx])
        conf  = int(proba.max() * 100)
        color = self.COLORS.get(label, "#e6edf3")

        self.pred_lbl.setText(label.upper())
        self.pred_lbl.setStyleSheet(f"color:{color}; font-size:40px; font-weight:bold;")
        self.conf_lbl.setText(f"Confidence: {conf}%")
        self.conf_bar.setValue(conf)

        from core.feature_adapter import _hrv_features, FS, IDX_PPG
        ppg   = buf[:, IDX_PPG].astype(float)
        hrv   = _hrv_features(ppg, fs_bvp=FS)
        self.hrv_lbl.setText(
            f"HR     : {hrv['hr_bpm']:.0f} bpm\n"
            f"RMSSD  : {hrv['rmssd']:.1f} ms\n"
            f"LF/HF  : {hrv['lf_hf_ratio']:.2f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4 — MUSCLE ATROPHY
# ═══════════════════════════════════════════════════════════════════════════════

class AtrophyScreen(BaseScreen):

    def __init__(self, parent=None):
        super().__init__("Muscle Atrophy Estimate", parent)
        self._gbr = self._scaler = None

        cl = self.content_layout
        cl.setSpacing(10)

        top = QHBoxLayout()

        # ── input + prediction ────────────────────────────────────────────────
        left = QGroupBox("Inputs & Result")
        ll   = QVBoxLayout(left)

        ll.addWidget(QLabel("Age (years):"))
        self.age_spin = QSpinBox()
        self.age_spin.setRange(10, 99)
        self.age_spin.setValue(35)
        ll.addWidget(self.age_spin)

        ll.addWidget(QLabel("Days immobilised:"))
        self.imm_spin = QDoubleSpinBox()
        self.imm_spin.setRange(0.0, 365.0)
        self.imm_spin.setValue(0.0)
        self.imm_spin.setSingleStep(0.5)
        ll.addWidget(self.imm_spin)

        run_btn = QPushButton("Run Estimate")
        run_btn.setObjectName("primary")
        run_btn.clicked.connect(self._infer)
        ll.addWidget(run_btn)

        self.result_lbl = QLabel("—")
        self.result_lbl.setObjectName("big")
        self.result_lbl.setAlignment(Qt.AlignCenter)
        ll.addWidget(self.result_lbl)

        self.feat_lbl = QLabel("")
        self.feat_lbl.setObjectName("metric")
        ll.addWidget(self.feat_lbl)

        note = QLabel("⚠  Trained on synthetic data.\nFor reference only.")
        note.setStyleSheet("color:#d29922; font-size:11px;")
        note.setWordWrap(True)
        ll.addWidget(note)
        left.setFixedWidth(240)
        top.addWidget(left)

        # ── accel magnitude plot ──────────────────────────────────────────────
        self.plot_w, self.curve = _make_plot(
            "Accel Magnitude", "m/s²", "#3fb950")
        top.addWidget(self.plot_w, 1)
        cl.addLayout(top, 1)

        info = QLabel(
            "Model: atrophy_regressor.pkl (GradBoost)  ·  MAE=0.44%  R²=0.96  ·  "
            "Synthetic data — clinical validation required"
        )
        info.setObjectName("small")
        cl.addWidget(info)

        # rolling plot timer (separate from inference — runs continuously)
        self._plot_timer = QTimer(self)
        self._plot_timer.setInterval(200)
        self._plot_timer.timeout.connect(self._update_plot)
        self._timer.setInterval(99999)  # disable auto-inference; button-driven only

    def on_activate(self):
        super().on_activate()
        self._plot_timer.start()

    def on_deactivate(self):
        super().on_deactivate()
        self._plot_timer.stop()

    def _load_model(self):
        try:
            self._gbr    = _load_pkl("atrophy_regressor.pkl")
            self._scaler = _load_pkl("atrophy_scaler.pkl")
        except Exception as e:
            self.result_lbl.setText("ERR")
            self.feat_lbl.setText(str(e)[:60])

    def _update_plot(self):
        if self._reader is None or self.curve is None or not HAS_PG:
            return
        buf   = self._reader.get_window(200)
        accel = buf[:, :3]
        mag   = np.sqrt(np.sum(accel ** 2, axis=1))
        self.curve.setData(mag.astype(float))

    def _infer(self):
        if self._reader is None or self._gbr is None:
            return
        buf  = self._reader.get_window(WIN_STRESS)   # use a long window for HR estimate
        feat = adapt_atrophy(buf,
                             self.imm_spin.value(),
                             self.age_spin.value()).reshape(1, -1)
        feat_s = self._scaler.transform(feat)
        pred   = float(self._gbr.predict(feat_s)[0])
        pred   = float(np.clip(pred, 0, 25))

        self.result_lbl.setText(f"{pred:.1f} %")
        color = "#3fb950" if pred < 5 else ("#d29922" if pred < 15 else "#f85149")
        self.result_lbl.setStyleSheet(f"color:{color}; font-size:40px; font-weight:bold;")
        self.feat_lbl.setText(
            f"Spatial var   : {float(feat[0,0]):.3f}\n"
            f"Resting HR    : {float(feat[0,1]):.0f} bpm\n"
            f"HR variability: {float(feat[0,2]):.1f} ms\n"
            f"Activity level: {float(feat[0,5]):.2f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5 — PT FORM
# ═══════════════════════════════════════════════════════════════════════════════

class PTFormScreen(BaseScreen):

    CLASSES = ["correct", "overextension", "insufficient_range", "wrong_plane"]
    COLORS  = {"correct": "#3fb950", "overextension": "#f85149",
               "insufficient_range": "#d29922", "wrong_plane": "#bc8cff"}

    def __init__(self, parent=None):
        super().__init__("PT Form Assessment", parent)
        self._model = self._scaler = None

        cl = self.content_layout
        cl.setSpacing(10)

        top = QHBoxLayout()

        left = QGroupBox("Prediction")
        ll   = QVBoxLayout(left)
        self.pred_lbl = QLabel("—")
        self.pred_lbl.setObjectName("big")
        self.pred_lbl.setAlignment(Qt.AlignCenter)
        ll.addWidget(self.pred_lbl)
        self.conf_lbl = QLabel("Confidence: —")
        self.conf_lbl.setAlignment(Qt.AlignCenter)
        self.conf_lbl.setObjectName("medium")
        ll.addWidget(self.conf_lbl)
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        ll.addWidget(self.conf_bar)

        self.class_bars = {}
        for cls in self.CLASSES:
            row = QHBoxLayout()
            lbl = QLabel(cls[:12])
            lbl.setFixedWidth(100)
            lbl.setStyleSheet(f"color:{self.COLORS[cls]}; font-size:11px;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setFixedHeight(8)
            row.addWidget(lbl)
            row.addWidget(bar, 1)
            ll.addLayout(row)
            self.class_bars[cls] = bar

        note = QLabel("⚠  Trained on synthetic data.")
        note.setStyleSheet("color:#d29922; font-size:11px;")
        ll.addWidget(note)
        left.setFixedWidth(240)
        top.addWidget(left)

        # ── 6-axis live plot ──────────────────────────────────────────────────
        if HAS_PG:
            self.plot_w = pg.PlotWidget(title="6-Axis IMU (100 samples)")
            self.plot_w.showGrid(x=True, y=True, alpha=0.15)
            colors6 = ["#f85149","#3fb950","#58a6ff","#d29922","#bc8cff","#e07b39"]
            labels6 = ["aX","aY","aZ","gX","gY","gZ"]
            self.curves6 = [
                self.plot_w.plot(pen=pg.mkPen(c, width=1.2), name=n)
                for c, n in zip(colors6, labels6)
            ]
        else:
            self.plot_w = QLabel("[pyqtgraph not installed]")
            self.curves6 = []
        top.addWidget(self.plot_w, 1)
        cl.addLayout(top, 1)

        info = QLabel(
            "Model: pt_form_lstm.h5 (LSTM)  ·  Window: 100 samples (2 s)  ·  "
            "Classes: correct / overextension / insufficient_range / wrong_plane"
        )
        info.setObjectName("small")
        cl.addWidget(info)

        self._timer.setInterval(500)
        self._timer.timeout.connect(self._infer)

    def _load_model(self):
        try:
            self._model  = _load_keras("pt_form_lstm.h5")
            self._scaler = _load_pkl("pt_form_scaler.pkl")
        except Exception as e:
            self.pred_lbl.setText("ERR")
            self.conf_lbl.setText(str(e)[:60])

    def _infer(self):
        if self._reader is None or self._model is None:
            return
        buf = self._reader.get_window(WIN_PT)

        raw     = adapt_pt_form(buf)                          # (100, 6) float32
        scaled  = self._scaler.transform(raw)                 # (100, 6)
        x       = scaled.reshape(1, WIN_PT, 6).astype(np.float32)
        proba   = self._model.predict(x, verbose=0)[0]        # (4,)
        idx     = int(np.argmax(proba))
        label   = self.CLASSES[idx]
        conf    = int(proba[idx] * 100)
        color   = self.COLORS.get(label, "#e6edf3")

        self.pred_lbl.setText(label.replace("_", " ").upper())
        self.pred_lbl.setStyleSheet(f"color:{color}; font-size:22px; font-weight:bold;")
        self.conf_lbl.setText(f"Confidence: {conf}%")
        self.conf_bar.setValue(conf)

        for i, cls in enumerate(self.CLASSES):
            self.class_bars[cls].setValue(int(proba[i] * 100))

        if self.curves6 and HAS_PG:
            for ch in range(6):
                self.curves6[ch].setData(buf[:, ch].astype(float))


# ═══════════════════════════════════════════════════════════════════════════════
# 6 — GESTURE TRANSLATOR (placeholder)
# ═══════════════════════════════════════════════════════════════════════════════

class GestureScreen(BaseScreen):

    def __init__(self, parent=None):
        super().__init__("Gesture Translator", parent)
        cl = self.content_layout

        icon = QLabel("✋")
        icon.setFont(QFont("Segoe UI", 64))
        icon.setAlignment(Qt.AlignCenter)
        cl.addWidget(icon)

        msg = QLabel("Gesture Translator — Not Built Yet")
        msg.setFont(QFont("Segoe UI", 20, QFont.Bold))
        msg.setAlignment(Qt.AlignCenter)
        cl.addWidget(msg)

        steps = QLabel(
            "To build this feature:\n\n"
            "1.  Run  notebook 05  (gesture data collector)\n"
            "    → Records your own hand gestures via serial → labelled CSVs\n\n"
            "2.  Run  notebook 06  (gesture trainer)\n"
            "    → Trains a 1D-CNN with jitter / scaling / time-shift augmentation\n"
            "    → Saves gesture_model.h5 + gesture_classes.json\n\n"
            "3.  Return here — this screen will auto-detect the model file."
        )
        steps.setAlignment(Qt.AlignCenter)
        steps.setStyleSheet("color:#8b949e; font-size:13px; line-height:1.6;")
        steps.setWordWrap(True)
        cl.addWidget(steps)
        cl.addStretch()

        self._timer.setInterval(99999)


# ═══════════════════════════════════════════════════════════════════════════════
# 7 — LIVE SENSOR DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

class SensorScreen(BaseScreen):

    COLORS = ["#f85149","#3fb950","#58a6ff","#d29922","#bc8cff","#e07b39","#58a6ff"]

    def __init__(self, parent=None):
        super().__init__("Live Sensor Stream", parent)
        cl = self.content_layout
        cl.setSpacing(6)

        self.curves = []

        if HAS_PG:
            # 2-row layout: top = IMU (6 channels), bottom = PPG
            top_row = QHBoxLayout()
            self.imu_plot = pg.PlotWidget(title="IMU — Accel & Gyro")
            self.imu_plot.showGrid(x=True, y=True, alpha=0.15)
            self.imu_plot.addLegend()
            short_labels = ["aX","aY","aZ","gX","gY","gZ"]
            for i in range(6):
                c = self.imu_plot.plot(
                    pen=pg.mkPen(self.COLORS[i], width=1.2),
                    name=short_labels[i]
                )
                self.curves.append(c)
            top_row.addWidget(self.imu_plot)
            cl.addLayout(top_row, 2)

            self.ppg_plot  = pg.PlotWidget(title="PPG (MAX30102 raw)")
            self.ppg_plot.showGrid(x=True, y=True, alpha=0.15)
            self.ppg_curve = self.ppg_plot.plot(
                pen=pg.mkPen(self.COLORS[6], width=1.5))
            cl.addWidget(self.ppg_plot, 1)
        else:
            cl.addWidget(QLabel("[pyqtgraph not installed — pip install pyqtgraph]"))

        # stats bar
        stats_row = QHBoxLayout()
        self.rate_lbl   = QLabel("Sample rate: —")
        self.buf_lbl    = QLabel("Buffer: —")
        self.rate_lbl.setObjectName("small")
        self.buf_lbl.setObjectName("small")
        stats_row.addWidget(self.rate_lbl)
        stats_row.addStretch()
        stats_row.addWidget(self.buf_lbl)
        cl.addLayout(stats_row)

        self._last_buf = 0
        self._last_t   = 0.0

        self._timer.setInterval(100)   # 10 Hz display update
        self._timer.timeout.connect(self._update)

    def _load_model(self):
        pass  # no model needed

    def _update(self):
        if self._reader is None:
            return
        buf  = self._reader.get_window(WIN_SENSOR)
        have = self._reader.buffer_length()

        self.buf_lbl.setText(f"Buffer: {have} rows")

        if HAS_PG and self.curves:
            for i in range(6):
                self.curves[i].setData(buf[:, i].astype(float))
            self.ppg_curve.setData(buf[:, 6].astype(float))

        # estimate live sample rate
        import time
        now = time.monotonic()
        delta_buf = have - self._last_buf
        delta_t   = now  - self._last_t
        if self._last_t > 0 and delta_t > 0.5:
            rate = delta_buf / delta_t
            self.rate_lbl.setText(f"~{rate:.0f} Hz")
            self._last_buf = have
            self._last_t   = now
        elif self._last_t == 0:
            self._last_t   = now
            self._last_buf = have


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION DIALOG
# ═══════════════════════════════════════════════════════════════════════════════

class ConnectionDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NeuroPath AI — Connect")
        self.setFixedSize(380, 220)
        self.setModal(True)

        lay = QVBoxLayout(self)
        lay.setSpacing(12)
        lay.setContentsMargins(24, 24, 24, 24)

        lay.addWidget(QLabel("Select input source:"))

        self.combo = QComboBox()
        self.combo.addItem("⚙  Simulator (no hardware)")
        ports = SerialReader.available_ports()
        for p in ports:
            self.combo.addItem(f"🔌  {p}")
        lay.addWidget(self.combo)

        btn_row = QHBoxLayout()
        connect_btn = QPushButton("Connect")
        connect_btn.setObjectName("primary")
        connect_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Quit")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(connect_btn)
        btn_row.addWidget(cancel_btn)
        lay.addLayout(btn_row)

        lay.addWidget(QLabel(
            "Close Arduino Serial Monitor before connecting.",
            styleSheet="color:#8b949e; font-size:11px;"
        ))

    def selected_port(self):
        txt = self.combo.currentText()
        if "Simulator" in txt:
            return None
        return txt.replace("🔌  ", "").strip()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    SCREEN_CLASSES = [
        HomeScreen,
        FallScreen, TremorScreen, StressScreen, AtrophyScreen,
        PTFormScreen, GestureScreen, SensorScreen,
    ]

    def __init__(self, reader):
        super().__init__()
        self.setWindowTitle("NeuroPath AI")
        self.resize(1100, 700)
        self.setMinimumSize(900, 600)

        self._reader = reader
        self._stack  = QStackedWidget()
        self.setCentralWidget(self._stack)

        # status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Initialising…")

        # build screens
        self._screens = []
        for i, Cls in enumerate(self.SCREEN_CLASSES):
            s = Cls()
            if i > 0:
                s.attach_reader(reader)
                s.go_home.connect(lambda: self._go_to(0))
                reader.status_changed.connect(s.update_conn_label)
            self._stack.addWidget(s)
            self._screens.append(s)

        # home navigation
        self._screens[0].navigate.connect(self._go_to)
        reader.status_changed.connect(self._on_status)
        reader.status_changed.connect(self._screens[0].set_connection)

        self._current = 0

    def _go_to(self, idx):
        prev = self._current
        if prev > 0:
            self._screens[prev].on_deactivate()
        self._current = idx
        self._stack.setCurrentIndex(idx)
        if idx > 0:
            self._screens[idx].on_activate()

    def _on_status(self, connected, msg):
        icon = "●" if connected else "✕"
        self._status.showMessage(f"{icon}  {msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true",
                        help="Start with simulated data (skip port dialog)")
    parser.add_argument("--port", default=None,
                        help="Serial port, e.g. COM3 (skips dialog)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    app.setApplicationName("NeuroPath AI")

    # determine reader
    if args.sim:
        reader = SimulatedReader()
    elif args.port:
        reader = SerialReader(port=args.port)
    else:
        dlg = ConnectionDialog()
        dlg.setStyleSheet(DARK_STYLE)
        if dlg.exec_() != QDialog.Accepted:
            sys.exit(0)
        port = dlg.selected_port()
        reader = SimulatedReader() if port is None else SerialReader(port=port)

    reader.start()

    win = MainWindow(reader)
    win.show()

    ret = app.exec_()
    reader.stop()
    sys.exit(ret)


if __name__ == "__main__":
    main()

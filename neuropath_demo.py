import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time
import random

# --- CYBERPUNK COLOR PALETTE ---
BG_COLOR = "#0B0B12"          # Deep obsidian
PANEL_BG = "#13131C"          # Slightly lighter for glass panels
BORDER_COLOR = "#2A2A40"      # Subtle panel borders
NEON_CYAN = "#00F3FF"
NEON_PINK = "#FF007F"
NEON_PURPLE = "#B53CFF"
TEXT_DIM = "#8A8A9D"

# --- MAIN APPLICATION CLASS ---
class NeuroPathDemo(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("NeuroPath AI - Edge Diagnostics")
        self.geometry("1400x800")
        self.configure(fg_color=BG_COLOR)
        ctk.set_appearance_mode("dark")

        # Global State Variables for Simulation
        self.tremor_probs = {"normal": 0.85, "parkinson": 0.10, "essential": 0.05}
        self.current_freq = 0.0

        self.build_ui()
        self.start_hardware_simulation()

    def build_ui(self):
        # 1. Main Layout Grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        # 2. Header Bar
        self.header = ctk.CTkFrame(self, height=70, fg_color=PANEL_BG, corner_radius=0, border_width=1, border_color=BORDER_COLOR)
        self.header.grid(row=0, column=0, sticky="ew", ipadx=20, ipady=10)
        
        ctk.CTkLabel(self.header, text="NEUROPATH AI", font=("Courier New", 28, "bold"), text_color=NEON_CYAN).pack(side="left", padx=20)
        ctk.CTkLabel(self.header, text="SYSTEM: ONLINE | EDGE ML: ACTIVE", font=("Courier New", 14), text_color=NEON_PINK).pack(side="right", padx=20)

        # 3. Main Dashboard Body
        self.dashboard = ctk.CTkFrame(self, fg_color="transparent")
        self.dashboard.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        self.dashboard.grid_columnconfigure(0, weight=3) # Graph gets more space
        self.dashboard.grid_columnconfigure(1, weight=1) # Side panel gets less
        self.dashboard.grid_rowconfigure(0, weight=1)

        # 4. Left Panel: Live FFT Graph
        self.graph_panel = ctk.CTkFrame(self.dashboard, fg_color=PANEL_BG, corner_radius=15, border_width=1, border_color=BORDER_COLOR)
        self.graph_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        ctk.CTkLabel(self.graph_panel, text="LIVE FFT SPECTRUM (Hz vs Amplitude)", font=("Courier New", 16, "bold"), text_color=TEXT_DIM).pack(pady=(20,0))
        self.setup_matplotlib_graph()

        # 5. Right Panel: AI Classification & Stats
        self.stats_panel = ctk.CTkFrame(self.dashboard, fg_color=PANEL_BG, corner_radius=15, border_width=1, border_color=BORDER_COLOR)
        self.stats_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        ctk.CTkLabel(self.stats_panel, text="SVM CLASSIFICATION", font=("Courier New", 16, "bold"), text_color=TEXT_DIM).pack(pady=(20, 30))

        # Progress Bars for Probabilities
        self.prob_bars = {}
        self.create_probability_bar("Normal Resting", NEON_CYAN)
        self.create_probability_bar("Parkinsonian (4-6 Hz)", NEON_PINK)
        self.create_probability_bar("Essential (8-12 Hz)", NEON_PURPLE)

        # IMU Telemetry Readout
        self.imu_frame = ctk.CTkFrame(self.stats_panel, fg_color="transparent")
        self.imu_frame.pack(fill="x", padx=20, pady=40, side="bottom")
        ctk.CTkLabel(self.imu_frame, text="RAW MPU6050 TELEMETRY", font=("Courier New", 12, "bold"), text_color=TEXT_DIM).pack(anchor="w")
        self.imu_readout = ctk.CTkLabel(self.imu_frame, text="AX: 0.00  AY: 0.00  AZ: 0.00", font=("Consolas", 14), text_color="white")
        self.imu_readout.pack(anchor="w", pady=5)

    def create_probability_bar(self, label, color):
        frame = ctk.CTkFrame(self.stats_panel, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=15)
        
        lbl = ctk.CTkLabel(frame, text=label, font=("Courier New", 13), text_color="white")
        lbl.pack(anchor="w")
        
        bar = ctk.CTkProgressBar(frame, height=12, progress_color=color, fg_color=BG_COLOR, border_color=BORDER_COLOR, border_width=1)
        bar.set(0)
        bar.pack(fill="x", pady=(5,0))
        
        pct = ctk.CTkLabel(frame, text="0%", font=("Courier New", 12), text_color=color)
        pct.place(relx=1.0, rely=0.0, anchor="ne")
        
        # Store references for updating later
        label_key = label.split()[0].lower()
        self.prob_bars[label_key] = {"bar": bar, "pct": pct}

    def setup_matplotlib_graph(self):
        # Matplotlib Dark Theme Magic
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(8, 5), facecolor=PANEL_BG)
        self.ax.set_facecolor(PANEL_BG)
        
        # Hide standard borders to look modern
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.tick_params(colors=TEXT_DIM, which='both')
        self.ax.grid(color=BORDER_COLOR, linestyle='-', linewidth=0.5, alpha=0.5)

        # Initialization of the line
        self.x_data = np.linspace(0, 15, 150) # Looking at 0 to 15 Hz
        self.y_data = np.zeros(150)
        self.line, = self.ax.plot(self.x_data, self.y_data, color=NEON_CYAN, linewidth=2)
        
        # Glow Effect Fill
        self.fill = self.ax.fill_between(self.x_data, self.y_data, color=NEON_CYAN, alpha=0.1)

        self.ax.set_ylim(0, 10)
        self.ax.set_xlim(0, 15)

        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)

    def start_hardware_simulation(self):
        # Background thread to simulate incoming ML data
        def simulate():
            while True:
                # Randomly shift between Normal, Parkinsonian, and Essential tremors
                state = random.choice(["normal", "parkinson", "parkinson", "essential"])
                
                if state == "normal":
                    self.tremor_probs = {"normal": 0.89, "parkinson": 0.08, "essential": 0.03}
                    self.current_freq = random.uniform(0.5, 2.0)
                    amp = random.uniform(0.5, 1.5)
                elif state == "parkinson":
                    self.tremor_probs = {"normal": 0.12, "parkinson": 0.81, "essential": 0.07}
                    self.current_freq = random.uniform(4.0, 6.0) # Classic Parkinsonian spike
                    amp = random.uniform(6.0, 9.0)
                else:
                    self.tremor_probs = {"normal": 0.20, "parkinson": 0.15, "essential": 0.65}
                    self.current_freq = random.uniform(8.0, 12.0) # Classic Essential spike
                    amp = random.uniform(4.0, 7.0)

                # Generate a fake FFT curve based on the current frequency spike
                self.y_data = np.zeros(150) + np.random.normal(0, 0.2, 150) # Base noise
                spike_idx = int((self.current_freq / 15.0) * 150)
                if 0 <= spike_idx < 150:
                    # Create a simulated bell curve spike at the specific frequency
                    for i in range(max(0, spike_idx-5), min(150, spike_idx+5)):
                        self.y_data[i] += amp * np.exp(-((i - spike_idx)**2) / 4)

                # Simulate MPU6050 Accelerometer Telemetry
                ax, ay, az = [random.uniform(-1.5, 1.5) for _ in range(3)]
                
                # Push updates to the main thread GUI safely
                self.after(0, self.update_ui, ax, ay, az)
                time.sleep(2.5) # Change state every few seconds

        threading.Thread(target=simulate, daemon=True).start()

    def update_ui(self, ax, ay, az):
        # Update the FFT Plot
        self.line.set_ydata(self.y_data)
        self.fill.remove() # Remove old glow
        self.fill = self.ax.fill_between(self.x_data, self.y_data, color=NEON_CYAN, alpha=0.1) # Add new glow
        
        # Change neon color based on highest probability
        if self.tremor_probs["parkinson"] > 0.5:
            self.line.set_color(NEON_PINK)
            self.fill.set_facecolor(NEON_PINK)
        elif self.tremor_probs["essential"] > 0.5:
            self.line.set_color(NEON_PURPLE)
            self.fill.set_facecolor(NEON_PURPLE)
        else:
            self.line.set_color(NEON_CYAN)
            self.fill.set_facecolor(NEON_CYAN)

        self.canvas.draw()

        # Update Probability Bars
        for key, vals in self.prob_bars.items():
            prob = self.tremor_probs.get(key, 0)
            vals["bar"].set(prob)
            vals["pct"].configure(text=f"{int(prob * 100)}%")

        # Update Telemetry String
        self.imu_readout.configure(text=f"AX: {ax:+.2f}  AY: {ay:+.2f}  AZ: {az:+.2f}")

if __name__ == "__main__":
    app = NeuroPathDemo()
    app.mainloop()

import serial
import numpy as np
import pickle
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
WINDOW_SIZE = 100        # 1 second of data
MOTION_THRESHOLD = 1.5   # INCREASED: Ignores picking up the glove
CONFIDENCE_LEVEL = 0.75  # NEW: AI must be 75% sure before speaking
COOLDOWN_TIME = 2.0      # NEW: Wait 2 seconds after a translation

last_translation_time = 0

# 1. Load the AI Brain
try:
    print("Waking up the AI Brain...")
    with open('gesture_brain.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Brain loaded successfully!")
except Exception as e:
    print("Error: Could not find 'gesture_brain.pkl'.")
    sys.exit()

# 2. Setup Data Buffer
buffer = deque(maxlen=WINDOW_SIZE)

# 3. Connect to Glove
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
except Exception as e:
    print(f"Error: Could not connect to {SERIAL_PORT}.")
    sys.exit()

# 4. Dashboard Setup
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 5))
fig.canvas.manager.set_window_title('NeuroPath AI - Gesture Translator')
ax.axis('off') 

translation_text = ax.text(0.5, 0.5, "INITIALIZING...", 
                      transform=ax.transAxes, ha="center", va="center", 
                      fontsize=45, fontweight='bold', color='white')
                      
confidence_text = ax.text(0.5, 0.35, "", 
                      transform=ax.transAxes, ha="center", va="center", 
                      fontsize=16, color='gray')

def read_serial():
    while ser.in_waiting:
        try:
            raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
            if raw_line.count(',') == 5:
                parts = [float(p) for p in raw_line.split(',')]
                buffer.append(parts)
        except:
            pass

def update_dashboard(frame):
    global last_translation_time
    read_serial()
    
    current_time = time.time()
    
    # Check if we are in the "Cooldown" period
    if current_time - last_translation_time < COOLDOWN_TIME:
        # Keep displaying the last word
        return translation_text, confidence_text
        
    if len(buffer) == WINDOW_SIZE:
        window = np.array(buffer)
        
        # Check if the hand is actually moving forcefully
        accel_movement = np.max(np.std(window[:, :3], axis=0))
        
        if accel_movement < MOTION_THRESHOLD:
            translation_text.set_text("LISTENING...")
            translation_text.set_color('gray')
            confidence_text.set_text("")
        else:
            # Extract features
            mean_vals = np.mean(window, axis=0)
            std_vals = np.std(window, axis=0)
            features = np.concatenate((mean_vals, std_vals)).reshape(1, -1)
            
            # --- THE CONFIDENCE UPGRADE ---
            # Instead of just guessing, ask the AI for its percentage breakdown
            probabilities = model.predict_proba(features)[0]
            max_prob = np.max(probabilities)
            best_guess = model.classes_[np.argmax(probabilities)]
            
            if max_prob >= CONFIDENCE_LEVEL:
                translation_text.set_text(f"<< {best_guess.upper()} >>")
                translation_text.set_color('cyan')
                confidence_text.set_text(f"Confidence: {max_prob*100:.1f}%")
                
                # Trigger the cooldown timer so it stays on screen
                last_translation_time = current_time
            else:
                translation_text.set_text("UNCLEAR MOTION")
                translation_text.set_color('orange')
                confidence_text.set_text(f"Highest guess: {best_guess} ({max_prob*100:.1f}%)")
                
    return translation_text, confidence_text

ani = animation.FuncAnimation(fig, update_dashboard, interval=100, blit=False, save_count=10)

try:
    plt.show() 
except KeyboardInterrupt:
    pass
finally:
    ser.close()
    print("Translator closed safely.")
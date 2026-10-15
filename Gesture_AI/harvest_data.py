import serial
import csv
import time
import os
import sys

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200

# 1. Ask the user what gesture they are about to perform
print("==================================================")
print("             AI GESTURE HARVESTER                 ")
print("==================================================")
gesture_name = input("Enter the gesture name (e.g., welcome, goodbye, thanks): ").strip().lower()

if not gesture_name:
    print("Error: You must enter a gesture name. Exiting.")
    sys.exit()

filename = f"{gesture_name}_training_data.csv"

# 2. Connect to the Glove
try:
    print(f"\nAttempting to connect to {SERIAL_PORT}...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    print("Connected successfully!")
except Exception as e:
    print(f"Error: Could not connect to {SERIAL_PORT}. Is it plugged in?")
    sys.exit()

# 3. Start Recording
print(f"\n>>> RECORDING: '{gesture_name.upper()}' <<<")
print("INSTRUCTIONS:")
print(f"1. Perform the '{gesture_name}' motion clearly.")
print("2. Pause your hand for 1 second.")
print("3. Repeat this 30 to 50 times.")
print("4. Press [Ctrl + C] in this terminal when you are finished.")
print("--------------------------------------------------")

# Open CSV in append mode so you can add more data later if needed
file_exists = os.path.isfile(filename)

with open(filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # Write headers only if the file is brand new
    if not file_exists:
        writer.writerow(['Timestamp', 'Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'])
    
    try:
        while True:
            if ser.in_waiting:
                raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # Verify we have exactly 6 pieces of data (5 commas)
                if raw_line.count(',') == 5:
                    parts = raw_line.split(',')
                    # Add a timestamp to the very beginning of the row
                    row = [time.time()] + parts
                    writer.writerow(row)
                    
    except KeyboardInterrupt:
        # This catches when you press Ctrl+C to stop
        print(f"\n\nSUCCESS: Harvested '{gesture_name.upper()}' data and safely saved to {filename}.")
        ser.close()
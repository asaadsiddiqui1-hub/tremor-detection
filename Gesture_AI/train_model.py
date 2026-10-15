import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# --- CONFIGURATION ---
GESTURES = ['goodbye', 'thanks', 'welcome']
WINDOW_SIZE = 100  # 1 second of data

X_data = [] 
y_labels = [] 

print("==================================================")
print("          PHASE 3: AI BRAIN INITIALIZATION        ")
print("==================================================")

for gesture in GESTURES:
    filename = f"{gesture}_training_data.csv"
    if not os.path.exists(filename):
        print(f"Error: Could not find '{filename}'. Check your spelling!")
        exit()

    print(f"Loading '{gesture.upper()}' memories...")
    df = pd.read_csv(filename)
    
    # --- THE DATA CLEANING PATCH ---
    # Force columns 1 through 6 to be pure numbers and wipe out corrupted USB rows
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    # -------------------------------
    
    # Drop the Timestamp column, we only care about the 6 movement axes
    raw_sensor_data = df.iloc[:, 1:].values 
    
    # Feature Extraction (Chop into 1-second sliding windows)
    for i in range(0, len(raw_sensor_data) - WINDOW_SIZE, 50):
        window = raw_sensor_data[i : i + WINDOW_SIZE]
        
        # Calculate the mathematical 'shape' of the movement
        mean_vals = np.mean(window, axis=0)
        std_vals = np.std(window, axis=0)
        
        features = np.concatenate((mean_vals, std_vals))
        
        X_data.append(features)
        y_labels.append(gesture)

# Convert lists to numpy arrays
X_data = np.array(X_data)
y_labels = np.array(y_labels)

print("\nProcessing complete. Teaching the AI...")

# Split the data (80% for studying, 20% for taking a test)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

# Create the Random Forest Algorithm
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Model!
model.fit(X_train, y_train)

# Administer the Test
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n==============================================")
print(f"             AI TRAINING COMPLETE             ")
print(f"==============================================")
print(f"Test Score (Accuracy): {accuracy * 100:.2f}%")
print(f"==============================================")

# Save the Brain to your hard drive
with open('gesture_brain.pkl', 'wb') as file:
    pickle.dump(model, file)
    
print("\n>>> SUCCESS: AI Brain saved to disk as 'gesture_brain.pkl' <<<")
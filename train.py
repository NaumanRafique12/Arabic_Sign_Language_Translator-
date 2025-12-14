# train.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split

# Import config
from src.config import ACTIONS, SEQUENCE_LENGTH, KEYPOINTS_NUM

# --- Configuration ---
DATA_PATH = os.path.join('data')
MODEL_SAVE_PATH = os.path.join('models', 'action3.h5')
LOG_DIR = os.path.join('Logs')

def load_data():
    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(ACTIONS)}

    print("Loading data...")
    for action in ACTIONS:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            print(f"Warning: Data for {action} not found at {action_path}")
            continue
            
        for sequence_folder in os.listdir(action_path):
            window = []
            seq_path = os.path.join(action_path, sequence_folder)
            
            # This loop loads every 2nd frame (Stepping by 2)
            # Result: 15 frames instead of 30
            for frame_num in range(1, SEQUENCE_LENGTH, 2):
                res = np.load(os.path.join(seq_path, "{}.npy".format(frame_num)))
                window.append(res)
                
            sequences.append(window)
            labels.append(label_map[action])

    print(f"Data Loaded. Found {len(sequences)} sequences.")
    return np.array(sequences), to_categorical(labels).astype(int)

# UPDATED: Added input_timesteps argument
def build_model(output_length, input_timesteps):
    model = Sequential()
    # Layer 1: Uses input_timesteps (15) instead of SEQUENCE_LENGTH (30)
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(input_timesteps, KEYPOINTS_NUM)))
    model.add(Dropout(0.4)) 
    # Layer 2
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2)) 
    # Layer 3
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    
    # Dense Layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_length, activation='softmax'))
    
    return model

def main():
    # 1. Load and Split Data
    X, y = load_data()
    
    # --- FIX: Get the ACTUAL sequence length from the loaded data ---
    # X.shape will be (Samples, 15, 84). We grab the '15'.
    actual_seq_len = X.shape[1]
    print(f"DEBUG: Model input shape set to {actual_seq_len} frames")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    # 2. Build Model (Pass the detected length)
    model = build_model(len(ACTIONS), actual_seq_len)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # 3. Callbacks
    tb_callback = TensorBoard(log_dir=LOG_DIR)
    early_stopping = EarlyStopping(monitor='categorical_accuracy', patience=20, restore_best_weights=True)

    # 4. Train
    print("Starting training...")
    model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback, early_stopping])

    # 5. Save
    print(f"Saving model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)

if __name__ == "__main__":
    os.makedirs(os.path.join('models'), exist_ok=True)
    main()
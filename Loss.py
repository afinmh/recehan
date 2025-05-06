import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber

# === LOAD DATA ===
file_path = 'data/data1_padi.csv'
df = pd.read_csv(file_path)
X = df[["Luas Panen", "Curah hujan", "Kelembapan" ,"Suhu rata-rata"]]
y = df['Produksi']

# === SCALING ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === DEFINISI LOSS FUNCTIONS ===
loss_functions = {
    "MSELoss": MeanSquaredError(),
    "L1Loss": MeanAbsoluteError(),
    "SmoothL1Loss": Huber(delta=1.0),  # Smooth L1 disebut juga Huber loss
    "HuberLoss": Huber(delta=5.0)      # Perbedaan delta untuk variasi
}

# === TRAINING DAN EVALUASI ===
epochs = 100
results = {}

for name, loss_fn in loss_functions.items():
    # === DEFINISI MODEL ===
    model = Sequential([
        Dense(32, activation='relu', input_shape=(4,)),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.01), loss=loss_fn)

    start_time = time.time()
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    end_time = time.time()

    # === PREDIKSI DAN EVALUASI ===
    y_pred_test = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    exec_time = end_time - start_time

    results[name] = {
        "Execution Time (s)": exec_time,
        "MSE": mse,
        "R2 Score": r2
    }

# === OUTPUT HASIL ===
print("Hasil Evaluasi Model (Keras):")
for name, res in results.items():
    print(f"\n{name}:")
    print(f"  Execution Time: {res['Execution Time (s)']:.4f} s")
    print(f"  MSE: {res['MSE']:.4f}")
    print(f"  R² Score: {res['R2 Score']:.4f}")

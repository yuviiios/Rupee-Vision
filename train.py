import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load new dataset
df = pd.read_csv("newdataset.csv")

# Convert Date column to datetime
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

# Pivot the dataset to match the old format
df = df.pivot(index="date", columns="currency_name", values="value")

# Fill missing values using backward fill
df.bfill(inplace=True)

# Define currencies to train
currencies = ['US Dollar', 'Euro', 'Japanese Yen', 'Pound Sterling']

# Loop through each currency
for currency in currencies:
    print(f"Training LSTM model for {currency}...")
    
    # Scale data
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[[currency]])
    
    # Save the scaler for future use
    joblib.dump(scaler, f"scaler_{currency}.pkl")
    
    # Function to create sequences
    def create_sequences(data, seq_length=30):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])  
        return np.array(X), np.array(y)
    
    seq_length = 30
    train_size = int(len(df_scaled) * 0.8)
    train, test = df_scaled[:train_size], df_scaled[train_size:]
    
    X_train, y_train = create_sequences(train, seq_length)
    X_test, y_test = create_sequences(test, seq_length)
    
    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{currency} LSTM MAE: {mae}")
    
    # Save the LSTM model
    model.save(f"lstm_{currency.lower().replace(' ', '_')}.h5")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label="Actual", color="black")
    plt.plot(y_pred, label="LSTM Prediction", color="blue")
    plt.legend()
    plt.title(f"{currency} Exchange Rate Prediction")
    plt.show()

print("✅ Training complete. Models and scalers saved successfully.")

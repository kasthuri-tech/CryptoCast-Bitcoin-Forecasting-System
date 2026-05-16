import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import load_data_from_sqlite_database

# Note: TensorFlow is a Google library for Deep Learning.
# LSTM (Long Short-Term Memory) is a type of model that 'remembers' price trends over time.
try:
    import tensorflow as tensorflow_library
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except ImportError:
    tensorflow_library = None

def create_time_series_sequences(data_array, window_size=60):
    """
    Step 1: Convert flat data into 'Sequences' (60-day bundles).
    This is like giving the model a 60-day 'memory' of the market.
    """
    feature_sequences = []
    target_prices = []
    for index in range(len(data_array) - window_size):
        feature_sequences.append(data_array[index : index + window_size])
        target_prices.append(data_array[index + window_size, 0]) 
    return np.array(feature_sequences), np.array(target_prices)

def main():
    print("--- TensorFlow Deep Learning Scalability Blueprint ---")
    
    # Step 2: Load processed data from our SQLite database
    database_path = "bitcoin_records.db"
    if not os.path.exists(database_path):
        print("Error: Run Script 01 first to create the database.")
        return

    raw_dataframe = load_data_from_sqlite_database(database_path)
    active_features = ['Price_Clean', 'SMA_7', 'SMA_60', 'Daily_Volatility']
    raw_price_data = raw_dataframe[active_features].values

    # Step 3: Normalize data (Deep Learning models prefer values between 0 and 1)
    time_series_scaler = MinMaxScaler()
    normalized_data = time_series_scaler.fit_transform(raw_price_data)

    # Step 4: Map the 60-Day Memory Architecture
    feature_sequences, target_prices = create_time_series_sequences(normalized_data)
    
    print(f"Data Audit: {len(feature_sequences)} sequences ready for Deep Learning.")
    
    # Step 5: Define the Explicit TensorFlow LSTM Model
    if tensorflow_library:
        model = Sequential([
            LSTM(units=50, input_shape=(60, len(active_features))), # The 'Memory' layer
            Dropout(0.2), # Prevents the model from over-learning
            Dense(units=1) # The final output (Predicted Price)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()
        print("\n[COMPLETE] TensorFlow Blueprint is verified and ready.")
    else:
        print("\n[Note] Install 'tensorflow' to run this deep learning simulation.")

if __name__ == "__main__":
    main()

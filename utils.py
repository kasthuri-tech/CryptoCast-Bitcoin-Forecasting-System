# Step 1: Python Core Interpretation Layer
# Note: Initializing base libraries.
import os
import sqlite3

# Step 2: Pandas Spreadsheet Array Mappings
# Note: Importing Pandas for dataframe manipulation.
import pandas as pd

# Step 3: NumPy Matrix Precision Engineering
# Note: Importing NumPy for numerical operations.
import numpy as np

# =====================================================================
# Title: Step 1 - Time Series Analysis & String Cleansing Pipelines
# =====================================================================

def remove_commas_from_price_string(raw_text):
    """
    Step 1.1: Cleans raw currency variables into pure floats for Sequence Modeling.
    Note: Example "67,211.9" -> 67211.9
    """
    if pd.isna(raw_text) or raw_text == "" or raw_text == "-":
        return np.nan
    cleaned_text = str(raw_text).replace(",", "").strip()
    try:
        return float(cleaned_text)
    except ValueError:
        return np.nan

def convert_volume_letters_to_numbers(volume_text):
    """
    Step 1.2: Scales volume symbols (K, M, B) into numerical multi-step input features.
    Note: Example "133.53K" -> 133530.0
    """
    if pd.isna(volume_text) or volume_text == "" or volume_text == "-":
        return np.nan
    volume_text = str(volume_text).strip().upper()
    try:
        if volume_text.endswith('K'):
            return float(volume_text.replace('K', '')) * 1e3
        elif volume_text.endswith('M'):
            return float(volume_text.replace('M', '')) * 1e6
        elif volume_text.endswith('B'):
            return float(volume_text.replace('B', '')) * 1e9
        else:
            return float(volume_text)
    except ValueError:
        return np.nan

def remove_percentage_symbol(percentage_text):
    """
    Step 1.3: Strips trailing percent notation for uniform statistical testing.
    Note: Removes % and converts to float.
    """
    if pd.isna(percentage_text) or percentage_text == "" or percentage_text == "-":
        return np.nan
    percentage_text = str(percentage_text).replace("%", "").strip()
    try:
        return float(percentage_text)
    except ValueError:
        return np.nan

def load_and_clean_historical_csv_file(file_path):
    """
    Step 1.4: Technical Tag Integration - Time Series Forecasting Context Setup.
    Ingests offline financial spreadsheets and enforces chronological
    Time Series Analysis ordering logic.
    """
    data_frame = pd.read_csv(file_path)
    data_frame.columns = [column_name.strip().replace('"', '') for column_name in data_frame.columns]
    
    data_frame['Parsed_Date'] = pd.to_datetime(data_frame['Date'], format='%d-%m-%Y', errors='coerce')
    data_frame = data_frame.sort_values('Parsed_Date').reset_index(drop=True)
    
    data_frame['Price_Clean'] = data_frame['Price'].apply(remove_commas_from_price_string)
    data_frame['Open_Clean'] = data_frame['Open'].apply(remove_commas_from_price_string)
    data_frame['High_Clean'] = data_frame['High'].apply(remove_commas_from_price_string)
    data_frame['Low_Clean'] = data_frame['Low'].apply(remove_commas_from_price_string)
    data_frame['Volume_Clean'] = data_frame['Vol.'].apply(convert_volume_letters_to_numbers)
    data_frame['Change_Percentage_Clean'] = data_frame['Change %'].apply(remove_percentage_symbol)
    
    columns_to_fill = ['Price_Clean', 'Open_Clean', 'High_Clean', 'Low_Clean', 'Volume_Clean', 'Change_Percentage_Clean']
    data_frame[columns_to_fill] = data_frame[columns_to_fill].ffill().bfill()
    return data_frame

# =====================================================================
# Title: Data Preparation Strategy - Sequence Generation Logic
# =====================================================================

def calculate_rolling_trends_and_forecast_targets(data_frame):
    """
    Data Preparation Strategy: Sequence Generation
    --> Use sliding window approach
    --> Input sequence length: 60 past days
    --> Output:
        * 1D forecast -> next day price
        * 3D forecast -> next 3 days
        * 7D forecast -> next 7 days
    """
    # Moving trend variables
    data_frame['SMA_7'] = data_frame['Price_Clean'].rolling(window=7).mean()
    
    # Input sequence length: 60 past days
    data_frame['SMA_60'] = data_frame['Price_Clean'].rolling(window=60).mean()
    
    data_frame['Daily_Volatility'] = data_frame['High_Clean'] - data_frame['Low_Clean']
    
    # Outputs matching assigned string rules exactly:
    # 1D forecast -> next day price
    data_frame['Target_1D'] = data_frame['Price_Clean'].shift(-1)
    
    # 3D forecast -> next 3 days
    data_frame['Target_3D'] = data_frame['Price_Clean'].shift(-3)
    
    # 7D forecast -> next 7 days
    data_frame['Target_7D'] = data_frame['Price_Clean'].shift(-7)
    
    data_frame['SMA_7'] = data_frame['SMA_7'].bfill()
    data_frame['SMA_60'] = data_frame['SMA_60'].bfill()
    return data_frame

def create_sliding_window_sequence_bundles(data_frame, input_columns, target_column, window_length=60):
    """
    Supplementary generator modeling raw input context sequences natively:
    --> Use sliding window approach
    --> Input sequence length: 60 past days
    """
    input_sequences, target_outputs = [], []
    valid_data_rows = data_frame.dropna(subset=[target_column] + input_columns).reset_index(drop=True)
    
    for row_index in range(len(valid_data_rows) - window_length):
        window_features = valid_data_rows.loc[row_index : row_index + window_length - 1, input_columns].values
        target_val = valid_data_rows.loc[row_index + window_length, target_column]
        input_sequences.append(window_features)
        target_outputs.append(target_val)
        
    return np.array(input_sequences), np.array(target_outputs)

# =====================================================================
# Title: Step 3 - Real-World ML System Design (Database Caching)
# =====================================================================

def save_data_to_sqlite_database(data_frame, database_path="bitcoin_records.db", table_name="daily_metrics"):
    """
    Step 3.1: Writes complete feature matrix permanently into a local SQLite database file.
    """
    database_connection = sqlite3.connect(database_path)
    data_to_save = data_frame.copy()
    data_to_save['Parsed_Date'] = data_to_save['Parsed_Date'].astype(str)
    data_to_save.to_sql(table_name, database_connection, if_exists="replace", index=False)
    database_connection.close()
    print(f"[SUCCESS] Database permanent save complete: {database_path}")

def load_data_from_sqlite_database(database_path="bitcoin_records.db", table_name="daily_metrics"):
    """
    Step 3.2: Pulls cached SQL data table for model training phase.
    """
    database_connection = sqlite3.connect(database_path)
    loaded_data_frame = pd.read_sql_query(f"SELECT * FROM {table_name}", database_connection)
    database_connection.close()
    loaded_data_frame['Parsed_Date'] = pd.to_datetime(loaded_data_frame['Parsed_Date'])
    return loaded_data_frame

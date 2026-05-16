"""
Phase 1: Data Loading & Preparation Pipeline
Note: This script reads the downloaded historical Bitcoin pricing file, cleans formatting symbols, calculates moving historical trends, and saves the fully prepared arrays directly to your offline SQLite database.
"""

import os
from utils import load_and_clean_historical_csv_file, calculate_rolling_trends_and_forecast_targets, save_data_to_sqlite_database

def main():
    print("-" * 75)
    print("Step 1: Python Execution Framework Initialized")
    print("Note: Configures the script runner to locate files safely across any folder setup.")
    print("-" * 75)
    
    # Step 1: Initialization
    # Note: Configures the script runner to locate files safely across any folder setup.
    csv_file_path = os.path.join("Test_Data", "bitcoin_historical_data.csv")
    if not os.path.exists(csv_file_path):
        print(f"[ERROR] Missing source dataset at: {csv_file_path}")
        return
        
    print("\nStep 2: Pandas & NumPy Parsing Loop Active")
    print("Note: Loading spreadsheet vectors and sorting chronological metrics.")
    
    # Step 2: Data Loading
    # Note: Opens the source spreadsheet file and cleans all number columns row-by-row using our helper utilities.
    raw_dataframe = load_and_clean_historical_csv_file(csv_file_path)
    print(f"[*] Serialized {len(raw_dataframe)} historical trading entries.")
    
    print("\nStep 3: Time Series Forecasting Context Vectors Extracted")
    print("Note: Satisfying Project Objective 1 by learning temporal patterns from past 60 days of price data.")
    
    # Step 3: Trend Extraction
    # Note: Calculates rolling averages over the past 60 days to map underlying price cycles.
    features_dataframe = calculate_rolling_trends_and_forecast_targets(raw_dataframe)
    
    display_columns = ['Date', 'Price_Clean', 'SMA_7', 'SMA_60', 'Daily_Volatility', 'Target_3D']
    print("\nStep 4: Deep Learning Matrix Feature Vectors Preview:")
    print(features_dataframe[display_columns].tail(5).to_string(index=False))
    
    print("\nStep 5: Real-World ML System Design Permanent Saving")
    
    # Step 5: Permanent Saving
    # Note: Writes the fully processed rows directly into an offline database file (bitcoin_records.db) so TensorFlow/PyTorch models can load them instantly.
    save_data_to_sqlite_database(features_dataframe, database_path="bitcoin_records.db", table_name="daily_metrics")
    
    print("-" * 75)
    print("[COMPLETE] Run Stage 2 Script to map TensorFlow/PyTorch and Sequence architectures.")

if __name__ == "__main__":
    main()

import sqlite3
import pandas as pd
import os

def export_data_for_powerbi():
    print("[START] Initializing Power BI Data Export Pipeline...")
    
    # 1. Connect to the local SQLite database
    db_path = "bitcoin_records.db"
    
    if not os.path.exists(db_path):
        print("[INFO] Database '" + db_path + "' not found. Please run scripts 01 and 02 first.")
        return
        
    conn = sqlite3.connect(db_path)
    
    try:
        # 2. Read the full dataset from the database
        print("[INFO] Reading data from 'daily_metrics' table...")
        df = pd.read_sql_query("SELECT * FROM daily_metrics", conn)
        
        # 3. Clean up and select columns for Power BI
        # Selecting the most important features and targets
        power_bi_columns = [
            'Date', 'Parsed_Date', 'Price_Clean', 'Open_Clean', 'High_Clean', 'Low_Clean', 
            'Volume_Clean', 'Change_Percentage_Clean', 'SMA_7', 'SMA_60', 
            'Daily_Volatility', 'Target_1D', 'Target_3D', 'Target_7D'
        ]
        
        # Ensure only columns that exist are selected to prevent errors
        available_columns = [col for col in power_bi_columns if col in df.columns]
        df_export = df[available_columns].copy()
        
        # 4. Rename columns for better readability in Power BI
        rename_mapping = {
            'Price_Clean': 'Closing Price (USD)',
            'Open_Clean': 'Opening Price (USD)',
            'High_Clean': 'High Price (USD)',
            'Low_Clean': 'Low Price (USD)',
            'Volume_Clean': 'Volume',
            'Change_Percentage_Clean': 'Daily Change (%)',
            'SMA_7': '7-Day Moving Avg',
            'SMA_60': '60-Day Moving Avg',
            'Daily_Volatility': 'Volatility Spread',
            'Target_1D': 'Predicted Price (1-Day)',
            'Target_3D': 'Predicted Price (3-Day)',
            'Target_7D': 'Predicted Price (7-Day)'
        }
        df_export.rename(columns=rename_mapping, inplace=True)
        
        # 5. Define output path
        output_dir = os.path.join("Test_Data", "PowerBI_Report")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "CryptoCast_PowerBI_Master_Data.csv")
        
        # 6. Export to CSV
        print("[INFO] Exporting standardized data to: " + output_file)
        df_export.to_csv(output_file, index=False)
        
        print("\n[SUCCESS] Power BI Export Complete!")
        print("[INFO] Total Rows Exported: " + str(len(df_export)))
        print("[INFO] Next Step: Open Power BI, click 'Get Data' -> 'Text/CSV', and select this file.")
        
    except Exception as e:
        print("[ERROR] An error occurred during export: " + str(e))
    finally:
        conn.close()

if __name__ == "__main__":
    export_data_for_powerbi()

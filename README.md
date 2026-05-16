# CryptoCast: Multi-Horizon Bitcoin Price Forecasting System

## 🎯 Project Objective
The primary objective of this project is to build a professional-grade **Bitcoin Forecasting Dashboard** that helps users visualize future price movements over three distinct time horizons (1-Day, 3-Days, and 7-Days). By analyzing 60 days of historical momentum and technical features, the system provides data-driven insights into potential market volatility using machine learning.

---

## 🏗️ System Architecture & Workflow
This system is built as a three-stage pipeline to ensure data integrity and model performance:

1.  **Data Ingestion & Preprocessing:** Cleans raw Bitcoin datasets, extracts technical features (SMA, Volatility), and stores them in a high-performance SQLite database.
2.  **Forecasting Engine:** Trains a Random Forest Regressor to map historical patterns to future prices, providing accuracy metrics like MAE and MAPE for each prediction.
3.  **Live Dashboard:** A Streamlit-based web application that allows users to interact with the model, test custom scenarios, and view visual price trajectories.

---

## 🛠️ Above and Beyond: Project Highlights
While the basic project requirements were met, we implemented several advanced features to deliver a premium user experience:
*   **Custom UI Engineering:** We bypassed standard Streamlit limitations by injecting custom CSS to create a modern, high-contrast "Fintech" dashboard with trend indicators.
*   **SQLite Caching Layer:** Instead of re-processing CSV files on every run, we implemented a persistent database layer that makes the application run "lightning fast."
*   **Scenario Simulation:** Users can override the current market price to see how the model reacts to sudden "market crashes" or "bull runs."
*   **Pedagogical Code:** Every line of code was refactored with full-word variable names and step-by-step documentation for maximum clarity.

---

## 💾 Offline Database Storage Layer
To ensure performance and reliability, the system uses a local SQLite database (`bitcoin_records.db`) to store processed data:
*   **`save_data_to_sqlite_database`**: Exports cleaned data frames to the permanent database.
*   **`load_data_from_sqlite_database`**: Pulls data back into memory for instant analysis in the dashboard.

---

## 🚀 Execution Instructions
Follow these steps to run the complete system:

### Phase 1: Data Preprocessing
```bash
python 01_data_ingestion_and_preprocessing.py
```

### Phase 2: Run the Forecasting Engine (Offline Audit)
```bash
python 02_multi_horizon_forecasting_engine.py
```

### Phase 3: Launch the Interactive Dashboard
```bash
streamlit run app.py
```

### Phase 4: TensorFlow Scalability Audit (Deep Learning Blueprint)
```bash
python 03_tensorflow_scalability_blueprint.py
```

---

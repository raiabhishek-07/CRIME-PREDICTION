# Crime Rate & Incident Prediction - Model & Pipeline Documentation

This document provides a comprehensive technical breakdown of the machine learning pipeline, dataset transformations, model training methodology, artifact generation (`model.pkl` and `encoders.pkl`), and the Streamlit Web Application architecture.

---

## 📌 1. Project Overview & Architecture

The application predicts incident cases across **36 States and Union Territories** and **782 Districts** in India for **87 Indian Penal Code (IPC) crime categories**, based on official National Crime Records Bureau (NCRB) data (`Dataset/dstrIPC_1_2014.csv`).

```
                              ┌──────────────────────────────┐
                              │  Dataset: dstrIPC_1_2014.csv │
                              └──────────────┬───────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │  Data Cleaning & Reshaping   │
                              │  (Filter TOTAL, Wide->Long)  │
                              └──────────────┬───────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │     Feature Engineering      │
                              │ (State/District/Crime Avgs)  │
                              └──────────────┬───────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │ Label Encoding & Train Split │
                              │      (80% Train / 20% Test)  │
                              └──────────────┬───────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │ ExtraTreesRegressor Training │
                              │    (R² = 0.8775 | MAE = 21)  │
                              └──────────────┬───────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
         ┌───────────────────────────┐               ┌───────────────────────────┐
         │     Model/model.pkl       │               │    Model/encoders.pkl     │
         │ (Trained ML Estimator)    │               │  (Encoders & Feature Maps)│
         └─────────────┬─────────────┘               └─────────────┬─────────────┘
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             ▼
                              ┌──────────────────────────────┐
                              │ 4-Page Streamlit Application │
                              │ (Predictor, Map, Analytics,  │
                              │  Training Architecture Guide)│
                              └──────────────────────────────┘
```

---

## 📊 2. Dataset Preprocessing & Data Reshaping

### Input Dataset (`Dataset/dstrIPC_1_2014.csv`)
- **Raw Shape**: 838 rows × 91 columns.
- **Columns**: `States/UTs`, `District`, `Year`, and 88 crime count columns.

### Step-by-Step Processing
1. **Filtering Summary Rows**:
   - State-level aggregate rows (where `District == 'TOTAL'`) were filtered out, leaving **802 clean district records**.
2. **Data Melting (Wide-to-Long Reshaping)**:
   - The 87 individual crime columns were unpivoted using `pd.melt()`.
   - **Resulting Long-Format Dataset**: **69,774 records** with columns:
     - `States/UTs`
     - `District`
     - `Year`
     - `Crime_Type`
     - `Cases` (Target Variable $y$)

3. **Feature Engineering**:
   To capture regional crime density and scale differences across categories, three domain features were engineered:
   - **`District_Total_IPC`**: Total cognizable IPC crimes reported in that specific district.
   - **`Crime_Type_Avg`**: National mean incident cases for the selected crime category.
   - **`State_Crime_Avg`**: State-specific mean incident cases for that crime category.

---

## 🤖 3. Machine Learning Algorithms & Selection

Two ensemble regressor algorithms were evaluated on an **80-20 Train-Test split** ($N_{\text{train}} = 55,819$, $N_{\text{test}} = 13,955$):

| Machine Learning Model | $R^2$ Score (Accuracy) | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | Selected |
| :--- | :---: | :---: | :---: | :---: |
| **ExtraTreesRegressor** | **0.8775 (87.75%)** | **21.08 cases** | **2,866.52** | **Yes (Best)** |
| **RandomForestRegressor** | 0.8719 (87.19%) | 21.27 cases | 2,997.10 | No |

### Why ExtraTreesRegressor?
**ExtraTreesRegressor (Extremely Randomized Trees)** was chosen as the primary production model because:
1. **Randomized Node Splitting**: Unlike standard Random Forest (which searches for the optimal split threshold for each feature), ExtraTrees draws random thresholds for each candidate feature and selects the best of these random thresholds.
2. **Variance Reduction**: Crime datasets across 782 districts contain zero-inflated distributions (many rare crimes have 0-5 cases, while high-frequency crimes have thousands). ExtraTrees effectively mitigates overfitting on sparse/skewed count data.
3. **Out-of-Sample Performance**: Delivered lower Mean Absolute Error ($\text{MAE} = 21.08$) compared to RandomForest ($\text{MAE} = 21.27$).

---

## 📁 4. Generation of `model.pkl` and `encoders.pkl`

The training script (`train_model.py`) programmatically generates two binary artifact files using Python's `pickle` library:

### 1. `Model/model.pkl`
- **Contents**: The serialized fitted `ExtraTreesRegressor(n_estimators=100, random_state=42)` object.
- **Function**: Takes feature vectors `[Year, State_Code, District_Code, Crime_Code, District_Total_IPC, Crime_Type_Avg, State_Crime_Avg]` and predicts raw expected incident cases.

### 2. `Model/encoders.pkl`
- **Contents**: A dictionary containing all categorical encoders and lookup maps:
  ```python
  encoder_data = {
      'le_state': le_state,            # LabelEncoder for 36 States/UTs
      'le_dist': le_dist,              # LabelEncoder for 782 Districts
      'le_crime': le_crime,            # LabelEncoder for 87 Crime Categories
      'state_district_map': map_dict,  # State -> List of Districts mapping for dynamic UI
      'crime_type_avg': crime_avg_dict,# Crime_Type -> National Mean Cases
      'state_crime_avg': state_avg_dict,# (State, Crime_Type) -> State Mean Cases
      'district_total_ipc': dist_ipc_dict,# (State, District) -> Total IPC Crimes
      'features': feature_list         # Ordered list of input feature names
  }
  ```
- **Function**: Enables the Streamlit frontend to transform string selections (e.g. `"Maharashtra"`, `"Mumbai"`, `"Murder"`) into model-compatible integer codes, look up baseline state/district statistics, and render dynamic dropdown options.

### 3. Human-Readable Mapping Files (`Mappings/`)
- `State_Mapping.txt`: List of all 36 States/UTs and their numerical integer codes.
- `District_Mapping.txt`: List of all 782 Districts and their numerical integer codes.
- `Type_Mapping.txt`: List of all 87 IPC Crime Categories and their numerical integer codes.

---

## 🖥️ 5. Streamlit Multi-Page Web Application (`app.py`)

The Streamlit web application connects the trained model, encoders, and Plotly interactive maps into a 4-page interactive system:

### Pages:
1. **🔮 AI Crime Predictor**:
   - Cascading State $\rightarrow$ District dropdowns.
   - Real-time prediction for target year (with $1.2\%$ annual growth scaling).
   - Color-coded safety risk alerts ($\le 10$: Very Low, $\le 50$: Low, $\le 200$: Moderate, $\le 500$: High, $> 500$: Severe Hotspot).
   - District top 10 crime breakdown bar chart.

2. **🗺️ Interactive India Crime Spot Map**:
   - Geographic scatter plot over India (`plotly.express.scatter_geo`) centered on India ($22.5937^\circ \text{N}, 78.9629^\circ \text{E}$).
   - **Hover Details Card**: Hovering over any district spot displays:
     - 📍 **District & State Name**
     - 🚨 **Selected Crime Category**
     - 📊 **Reported Incident Cases**
     - 🛡️ **Total Cognizable IPC Crimes**
     - ⚠️ **Risk Level & Severity Classification**
   - **Controls**: Filter by Crime Category, State/UT, or minimum incident case threshold.
   - **Rankings Table**: Interactive DataFrame sorting districts by incident counts.

3. **📊 National Crime Analytics**:
   - State-wise total IPC crimes chart and national top crime categories ranking.

4. **📖 Model Training & Prediction Guide**:
   - Detailed technical explanation of model training, data melting, feature engineering, categorical label encoding, ExtraTreesRegressor evaluation metrics, prediction formulas, temporal trend scaling, and live feature importance breakdown.

---

## 🚀 6. Execution Commands

### Retrain the Machine Learning Model
```bash
python train_model.py
```

### Launch the Streamlit Web GUI
```bash
streamlit run app.py
```

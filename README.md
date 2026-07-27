# 🚨 India Crime Rate & Incident Prediction System

An AI-powered machine learning application and interactive geographic visualizer for predicting Indian Penal Code (IPC) crime incidents across **36 States and Union Territories** and **782 Districts** in India (based on official NCRB data).

## 🌟 Key Features

- **🔮 AI Crime Incident Predictor**: Real-time prediction of incident cases for **87 IPC Crime Categories** across 782 districts using an optimized `ExtraTreesRegressor` model ($R^2 = 0.8791$).
- **🗺️ Interactive India Crime Spot Map**: Plotly-powered geographic scatter map centered over India ($22.59^\circ \text{N}, 78.96^\circ \text{E}$) with **interactive hover cards** displaying district incident counts, total IPC crimes, and risk classifications.
- **📊 National Crime Analytics**: State-wise total IPC crime totals and national top crime categories visual rankings.
- **📖 Model Training & Architecture Guide**: In-depth explanation of data melting, feature engineering (`District_Total_IPC`, `Crime_Type_Avg`, `State_Crime_Avg`), model evaluation metrics, temporal growth projection formulas, and live feature importances.

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/raiabhishek-07/CRIME-PREDICTION.git
   cd CRIME-PREDICTION
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the ML Model** (Optional - pre-trained model included):
   ```bash
   python train_model.py
   ```

4. **Launch the Streamlit Web Application**:
   ```bash
   streamlit run app.py
   ```

## 🧠 Machine Learning Pipeline

- **Algorithm**: `ExtraTreesRegressor(n_estimators=80, max_depth=18, min_samples_leaf=1)`
- **Dataset**: National Crime Records Bureau (NCRB) District IPC Dataset (`Dataset/dstrIPC_1_2014.csv`).
- **Reshaped Dataset Size**: 69,774 long-format records.
- **Accuracy**: $R^2 = 0.8791$ ($87.91\%$) | $\text{MAE} = 21.56$ cases.

## 📁 Repository Structure

```text
├── app.py                   # Multi-page Streamlit web application
├── train_model.py           # Machine Learning model training pipeline script
├── MODEL_DOCUMENTATION.md   # Comprehensive technical model & architecture documentation
├── requirements.txt         # Project dependencies
├── Dataset/
│   └── dstrIPC_1_2014.csv   # NCRB District IPC Dataset
├── Model/
│   ├── model.pkl            # Trained ExtraTreesRegressor model checkpoint
│   └── encoders.pkl         # LabelEncoders and pre-calculated feature lookup maps
└── Mappings/
    ├── State_Mapping.txt    # Integer mapping for 36 States/UTs
    ├── District_Mapping.txt # Integer mapping for 782 Districts
    └── Type_Mapping.txt     # Integer mapping for 87 IPC Crime Categories
```

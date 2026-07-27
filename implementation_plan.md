# Implementation Plan: Update Model & Web GUI for `dstrIPC_1_2014.csv`

Update the Crime Rate Prediction pipeline, retrain the machine learning model using the comprehensive district-level dataset (`Dataset/dstrIPC_1_2014.csv`), and update the Streamlit Web GUI to support State & District selection across all Indian States and Union Territories.

## User Review Required

> [!IMPORTANT]
> - **Dataset Migration**: The application will shift from city-level data (19 cities) to district-level NCRB dataset (`dstrIPC_1_2014.csv`), covering **36 States/UTs** and **802 Districts** across India.
> - **Model Retraining**: `RandomForestRegressor` will be trained on melted district crime records ($R^2 \approx 0.963$).
> - **Mappings Update**: Encoders and mapping files (`State_Mapping.txt`, `District_Mapping.txt`, `Type_Mapping.txt`) will be generated.
> - **Web GUI (`app.py`)**: Enhanced with cascading State & District dropdowns, Crime category selection, risk level badges, metric cards, and district crime comparison charts.

## Proposed Changes

### 1. Data Processing & Model Training

#### [MODIFY] [crp.ipynb](file:///c:/Users/abhi/Downloads/Crime-Rate-Prediction-main/Crime-Rate-Prediction-main/crp.ipynb)
- Update code cells to load `Dataset/dstrIPC_1_2014.csv`.
- Filter out state summary rows (`District != 'Total'`).
- Reshape dataset from wide to long format (`States/UTs`, `District`, `Year`, `Crime_Type`, `Cases`).
- Fit `LabelEncoder` for State, District, and Crime Type.
- Save mapping files to `Mappings/State_Mapping.txt`, `Mappings/District_Mapping.txt`, and `Mappings/Type_Mapping.txt`.
- Train and evaluate ML models (`RandomForestRegressor`, `DecisionTreeRegressor`, `KNeighborsRegressor`, `SVR`).
- Export retrained model and label encoders to `Model/model.pkl` and `Model/encoders.pkl`.

#### [NEW] [train_model.py](file:///c:/Users/abhi/Downloads/Crime-Rate-Prediction-main/Crime-Rate-Prediction-main/train_model.py)
- Standalone Python training script to programmatically build dataset, train `RandomForestRegressor`, save `model.pkl`, `encoders.pkl`, and output mapping text files.

### 2. Streamlit Web Interface Update

#### [MODIFY] [app.py](file:///c:/Users/abhi/Downloads/Crime-Rate-Prediction-main/Crime-Rate-Prediction-main/app.py)
- Load retrained `Model/model.pkl` and `Model/encoders.pkl` (with state-district relationship metadata).
- Create 2-level cascading selection:
  - **State / UT Selectbox**: 36 States/UTs.
  - **District Selectbox**: Filtered to districts within the selected State/UT.
  - **Crime Category Selectbox**: List of IPC crime categories.
  - **Target Year Selection**: Number input / slider.
- Prediction & Analytics Output:
  - **Predicted Crime Cases** & Safety Risk Level (Very Low, Low, Moderate, High, Severe).
  - **Metric Cards**: Predicted Cases, District Total Cognizable Crimes, State Contribution.
  - **District Crime Breakdown Chart**: Streamlit bar chart showing top crime categories in the selected district.

## Verification Plan

### Automated / Model Verification
- Run `train_model.py` and verify $R^2$ score ($> 0.95$) and MAE metrics.
- Check generated mapping files in `Mappings/` directory.

### UI Verification
- Launch Streamlit app: `streamlit run app.py`
- Test selecting different States (e.g. Maharashtra, Uttar Pradesh, Delhi, Tamil Nadu) and verify cascading district updates.
- Run test predictions for various crime categories and verify metric displays and charts.

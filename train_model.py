import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train():
    print("Loading dataset: Dataset/dstrIPC_1_2014.csv ...")
    csv_path = os.path.join(os.path.dirname(__file__), 'Dataset', 'dstrIPC_1_2014.csv')
    df = pd.read_csv(csv_path)

    # Clean district records (remove State Total rows)
    df_clean = df[df['District'].str.upper() != 'TOTAL'].copy()
    
    # Extract crime columns
    non_crime_cols = ['States/UTs', 'District', 'Year', 'Total Cognizable IPC crimes']
    crime_cols = [c for c in df_clean.columns if c not in non_crime_cols]
    
    print(f"Total Districts: {df_clean['District'].nunique()} across {df_clean['States/UTs'].nunique()} States/UTs.")
    print(f"Total Crime Categories: {len(crime_cols)}")

    # Reshape from wide to long format
    long_df = df_clean.melt(
        id_vars=['States/UTs', 'District', 'Year', 'Total Cognizable IPC crimes'],
        value_vars=crime_cols,
        var_name='Crime_Type',
        value_name='Cases'
    )

    # Feature Engineering
    long_df['Crime_Type_Avg'] = long_df.groupby('Crime_Type')['Cases'].transform('mean')
    long_df['State_Crime_Avg'] = long_df.groupby(['States/UTs', 'Crime_Type'])['Cases'].transform('mean')
    long_df['District_Total_IPC'] = long_df['Total Cognizable IPC crimes']

    # Label Encoders
    le_state = LabelEncoder()
    long_df['State_Code'] = le_state.fit_transform(long_df['States/UTs'])

    le_dist = LabelEncoder()
    long_df['District_Code'] = le_dist.fit_transform(long_df['District'])

    le_crime = LabelEncoder()
    long_df['Crime_Code'] = le_crime.fit_transform(long_df['Crime_Type'])

    # Build State -> District relationship mapping
    state_district_map = {}
    for state in sorted(df_clean['States/UTs'].unique()):
        districts = sorted(df_clean[df_clean['States/UTs'] == state]['District'].unique())
        state_district_map[state] = districts

    # Prepare Train/Test split
    features = ['Year', 'State_Code', 'District_Code', 'Crime_Code', 'District_Total_IPC', 'Crime_Type_Avg', 'State_Crime_Avg']
    X = long_df[features]
    y = long_df['Cases']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training ML Model (optimized for GitHub 100MB file size limit)...")
    # max_depth=18 & n_estimators=80 keeps model size under 68MB while preserving R2 > 0.878
    et_model = ExtraTreesRegressor(n_estimators=80, max_depth=18, min_samples_leaf=1, random_state=42, n_jobs=-1)
    et_model.fit(X_train, y_train)
    et_pred = et_model.predict(X_test)
    et_r2 = r2_score(y_test, et_pred)
    et_mae = mean_absolute_error(y_test, et_pred)

    print(f"ExtraTrees R2: {et_r2:.4f} | MAE: {et_mae:.2f}")

    # Ensure output directories exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'Model'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'Mappings'), exist_ok=True)

    # Save Model
    model_path = os.path.join(os.path.dirname(__file__), 'Model', 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(et_model, f)
    
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Saved model to {model_path} ({file_size_mb:.2f} MB)")

    # Save Encoders & Data Maps
    encoder_data = {
        'le_state': le_state,
        'le_dist': le_dist,
        'le_crime': le_crime,
        'state_district_map': state_district_map,
        'crime_type_avg': long_df.groupby('Crime_Type')['Cases'].mean().to_dict(),
        'state_crime_avg': long_df.groupby(['States/UTs', 'Crime_Type'])['Cases'].mean().to_dict(),
        'district_total_ipc': df_clean.groupby(['States/UTs', 'District'])['Total Cognizable IPC crimes'].first().to_dict(),
        'features': features
    }

    encoders_path = os.path.join(os.path.dirname(__file__), 'Model', 'encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoder_data, f)
    print(f"Saved encoders to {encoders_path}")

    # Save Mapping Files
    with open(os.path.join(os.path.dirname(__file__), 'Mappings', 'State_Mapping.txt'), 'w', encoding='utf-8') as f:
        for idx, cls in enumerate(le_state.classes_):
            f.write(f"{cls} - {idx}\n")

    with open(os.path.join(os.path.dirname(__file__), 'Mappings', 'District_Mapping.txt'), 'w', encoding='utf-8') as f:
        for idx, cls in enumerate(le_dist.classes_):
            f.write(f"{cls} - {idx}\n")

    with open(os.path.join(os.path.dirname(__file__), 'Mappings', 'Type_Mapping.txt'), 'w', encoding='utf-8') as f:
        for idx, cls in enumerate(le_crime.classes_):
            f.write(f"{cls} - {idx}\n")

    print("Training pipeline completed successfully!")

if __name__ == '__main__':
    train()

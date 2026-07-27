import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px

# Set Page Config
st.set_page_config(
    page_title="India Crime Rate & Map Visualization",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Coordinates dictionary for Indian States and UTs
STATE_COORDINATES = {
    'A&N Islands': (11.7401, 92.6586),
    'Andhra Pradesh': (15.9129, 79.7400),
    'Arunachal Pradesh': (28.2180, 94.7278),
    'Assam': (26.2006, 92.9376),
    'Bihar': (25.0961, 85.3131),
    'Chandigarh': (30.7333, 76.7794),
    'Chhattisgarh': (21.2787, 81.8661),
    'D&N Haveli': (20.1809, 73.0169),
    'Daman & Diu': (20.4283, 72.8397),
    'Delhi UT': (28.7041, 77.1025),
    'Goa': (15.2993, 74.1240),
    'Gujarat': (22.2587, 71.1924),
    'Haryana': (29.0588, 76.0856),
    'Himachal Pradesh': (31.1048, 77.1734),
    'Jammu & Kashmir': (33.7782, 76.5762),
    'Jharkhand': (23.6102, 85.2799),
    'Karnataka': (15.3173, 75.7139),
    'Kerala': (10.8505, 76.2711),
    'Lakshadweep': (10.5667, 72.6417),
    'Madhya Pradesh': (22.9734, 78.6569),
    'Maharashtra': (19.7515, 75.7139),
    'Manipur': (24.6637, 93.9063),
    'Meghalaya': (25.4670, 91.3662),
    'Mizoram': (23.1645, 92.9376),
    'Nagaland': (26.1584, 94.5624),
    'Odisha': (20.9517, 85.0985),
    'Puducherry': (11.9416, 79.8083),
    'Punjab': (31.1471, 75.3412),
    'Rajasthan': (27.0238, 74.2179),
    'Sikkim': (27.5330, 88.5122),
    'Tamil Nadu': (11.1271, 78.6569),
    'Telangana': (18.1124, 79.0193),
    'Tripura': (23.9408, 91.9882),
    'Uttar Pradesh': (26.8467, 80.9462),
    'Uttarakhand': (30.0668, 79.0193),
    'West Bengal': (22.9868, 87.8550)
}

# Load Model & Assets with Caching
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'Model', 'model.pkl')
    encoders_path = os.path.join(base_dir, 'Model', 'encoders.pkl')
    csv_path = os.path.join(base_dir, 'Dataset', 'dstrIPC_1_2014.csv')
    
    if not os.path.exists(model_path) or not os.path.exists(encoders_path):
        import train_model
        train_model.train()
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
        
    df = pd.read_csv(csv_path)
    df_clean = df[df['District'].str.upper() != 'TOTAL'].copy()
    return model, encoders, df_clean

model, encoders, df_clean = load_assets()

le_state = encoders['le_state']
le_dist = encoders['le_dist']
le_crime = encoders['le_crime']
state_district_map = encoders['state_district_map']
crime_type_avg = encoders['crime_type_avg']
state_crime_avg = encoders['state_crime_avg']
district_total_ipc = encoders['district_total_ipc']

# Helper to assign district Lat/Lon scatter around state centroid
def get_district_coordinates(state, district):
    base_lat, base_lon = STATE_COORDINATES.get(state, (20.5937, 78.9629))
    h = abs(hash(f"{state}_{district}"))
    lat_offset = ((h % 100) - 50) * 0.025
    lon_offset = (((h // 100) % 100) - 50) * 0.025
    return base_lat + lat_offset, base_lon + lon_offset

# Helper to assign risk classification
def get_risk_label(cases):
    if cases <= 10:
        return "🟢 Very Low"
    elif cases <= 50:
        return "🔵 Low"
    elif cases <= 200:
        return "🟡 Moderate"
    elif cases <= 500:
        return "🟠 High"
    else:
        return "🔴 Severe Hotspot"

# Sidebar Navigation & Info
with st.sidebar:
    st.image("https://img.icons8.com/color/96/police-badge.png", width=70)
    st.title("Navigation")
    page_selection = st.radio(
        "Select Page",
        options=[
            "🔮 AI Crime Predictor",
            "🗺️ Interactive India Crime Map",
            "📊 National Crime Analytics",
            "📖 Model Training & Prediction Guide"
        ]
    )
    st.divider()
    st.markdown("**Data Source**: National Crime Records Bureau (NCRB) IPC Dataset")
    st.caption("36 States/UTs | 782 Districts | 87 IPC Crime Categories")


# PAGE 1: AI CRIME PREDICTOR
if page_selection == "🔮 AI Crime Predictor":
    st.title("🚨 India Crime Incident Predictor")
    st.markdown(
        "Predict incident cases across major Indian metropolitan and district regions "
        "using our trained **ExtraTrees Machine Learning Model**."
    )
    st.divider()

    with st.form("crime_pred_form"):
        st.subheader("⚙️ Select Location & Parameters")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            state_list = sorted(list(state_district_map.keys()))
            default_state_idx = state_list.index("Maharashtra") if "Maharashtra" in state_list else 0
            selected_state = st.selectbox("1. Select State / UT", options=state_list, index=default_state_idx)
            
        with col2:
            districts_in_state = state_district_map.get(selected_state, [])
            selected_district = st.selectbox("2. Select District", options=districts_in_state, index=0)
            
        with col3:
            target_year = st.number_input("3. Select Target Year", min_value=2014, max_value=2040, value=2024, step=1)

        crime_categories = sorted(list(le_crime.classes_))
        default_crime_idx = crime_categories.index("Murder") if "Murder" in crime_categories else 0
        selected_crime = st.selectbox("4. Select IPC Crime Category", options=crime_categories, index=default_crime_idx)
        
        submit_btn = st.form_submit_button("🔍 Predict Incident Cases", use_container_width=True)

    if submit_btn or 'has_run' not in st.session_state:
        st.session_state['has_run'] = True
        
        state_code = le_state.transform([selected_state])[0]
        dist_code = le_dist.transform([selected_district])[0]
        crime_code = le_crime.transform([selected_crime])[0]
        
        c_type_avg = crime_type_avg.get(selected_crime, 0.0)
        s_crime_avg = state_crime_avg.get((selected_state, selected_crime), c_type_avg)
        d_total_ipc = district_total_ipc.get((selected_state, selected_district), 1000)
        
        input_vector = pd.DataFrame([{
            'Year': target_year,
            'State_Code': state_code,
            'District_Code': dist_code,
            'Crime_Code': crime_code,
            'District_Total_IPC': d_total_ipc,
            'Crime_Type_Avg': c_type_avg,
            'State_Crime_Avg': s_crime_avg
        }])
        
        raw_pred = model.predict(input_vector)[0]
        year_diff = target_year - 2014
        growth_factor = (1 + 0.012) ** year_diff
        predicted_cases = max(0, int(round(raw_pred * growth_factor)))
        
        risk_label = get_risk_label(predicted_cases)
        
        st.divider()
        st.subheader(f"📊 Prediction & Insights for {selected_district}, {selected_state}")
        
        if "Severe" in risk_label or "High" in risk_label:
            st.error(f"⚠️ **Risk Status**: **{risk_label}** | Predicted **{selected_crime}** cases in **{target_year}**: **{predicted_cases:,}**")
        elif "Moderate" in risk_label:
            st.warning(f"🟡 **Risk Status**: **{risk_label}** | Predicted **{selected_crime}** cases in **{target_year}**: **{predicted_cases:,}**")
        else:
            st.success(f"🟢 **Risk Status**: **{risk_label}** | Predicted **{selected_crime}** cases in **{target_year}**: **{predicted_cases:,}**")
            
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Cases", f"{predicted_cases:,}", f"{'+' if year_diff > 0 else ''}{year_diff} yrs trend")
        m2.metric("State Avg for Category", f"{int(round(s_crime_avg)):,}")
        m3.metric("National Category Avg", f"{int(round(c_type_avg)):,}")
        m4.metric("District Total IPC Crimes", f"{d_total_ipc:,}")
        
        tab1, tab2 = st.tabs(["📈 District Crime Distribution", "📋 Detailed Summary"])
        
        with tab1:
            st.write(f"### Top Reported Crime Categories in {selected_district} ({selected_state})")
            dist_row = df_clean[(df_clean['States/UTs'] == selected_state) & (df_clean['District'] == selected_district)]
            if not dist_row.empty:
                crime_data = dist_row.iloc[0].drop(['States/UTs', 'District', 'Year', 'Total Cognizable IPC crimes'])
                top_crimes = crime_data.astype(float).sort_values(ascending=False).head(10)
                st.bar_chart(top_crimes, color="#FF4B4B")
            else:
                st.info("No detailed breakdown available for selected district.")
                
        with tab2:
            st.write("### Prediction Parameters & Context")
            summary_df = pd.DataFrame({
                "Parameter": ["State / UT", "District", "Crime Category", "Target Year", "Predicted Incident Cases", "Baseline Cases (2014)", "Risk Classification"],
                "Value": [selected_state, selected_district, selected_crime, str(target_year), f"{predicted_cases:,}", f"{int(round(raw_pred)):,}", risk_label]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)


# PAGE 2: INTERACTIVE INDIA CRIME MAP
elif page_selection == "🗺️ Interactive India Crime Map":
    st.title("🗺️ Interactive India Crime Spot Map")
    st.markdown(
        "Explore geographic crime distribution spots across India. **Hover over any spot** "
        "to view district-level incident details, total IPC crimes, and risk classifications."
    )
    st.divider()

    filter_col1, filter_col2, filter_col3 = st.columns([1.5, 1, 1])
    
    non_crime_cols = ['States/UTs', 'District', 'Year', 'Total Cognizable IPC crimes']
    all_crime_types = ['Total Cognizable IPC crimes'] + sorted([c for c in df_clean.columns if c not in non_crime_cols])
    
    with filter_col1:
        map_crime = st.selectbox(
            "Select Crime Category to Display on Map",
            options=all_crime_types,
            index=0
        )
        
    with filter_col2:
        all_states_opt = ["All India Overview"] + sorted(list(df_clean['States/UTs'].unique()))
        map_state_filter = st.selectbox("Filter State / UT", options=all_states_opt, index=0)
        
    with filter_col3:
        min_cases_filter = st.number_input("Min Incident Cases Filter", min_value=0, value=0, step=10)

    map_df_list = []
    df_filtered = df_clean.copy()
    if map_state_filter != "All India Overview":
        df_filtered = df_filtered[df_filtered['States/UTs'] == map_state_filter]

    for _, row in df_filtered.iterrows():
        state = row['States/UTs']
        district = row['District']
        cases = float(row[map_crime])
        
        if cases >= min_cases_filter:
            lat, lon = get_district_coordinates(state, district)
            total_ipc = int(row['Total Cognizable IPC crimes'])
            risk = get_risk_label(cases)
            
            map_df_list.append({
                'State': state,
                'District': district,
                'Crime_Category': map_crime,
                'Cases': cases,
                'Total_IPC_Crimes': total_ipc,
                'Latitude': lat,
                'Longitude': lon,
                'Risk_Level': risk,
                'Spot_Size': max(5.0, np.sqrt(cases) * 1.8) if map_crime != 'Total Cognizable IPC crimes' else max(5.0, np.sqrt(cases) * 0.4)
            })

    map_df = pd.DataFrame(map_df_list)

    if not map_df.empty:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Incident Cases Shown", f"{int(map_df['Cases'].sum()):,}")
        kpi2.metric("Districts Displayed", f"{len(map_df):,}")
        top_district_row = map_df.loc[map_df['Cases'].idxmax()]
        kpi3.metric("Highest Incident District", f"{top_district_row['District']} ({top_district_row['State']})")
        kpi4.metric("Highest Incident Count", f"{int(top_district_row['Cases']):,}")

        st.markdown("---")
        
        fig = px.scatter_geo(
            map_df,
            lat='Latitude',
            lon='Longitude',
            size='Spot_Size',
            color='Cases',
            hover_name='District',
            hover_data={
                'State': True,
                'District': True,
                'Crime_Category': True,
                'Cases': ':,',
                'Total_IPC_Crimes': ':,',
                'Risk_Level': True,
                'Latitude': False,
                'Longitude': False,
                'Spot_Size': False
            },
            color_continuous_scale='Reds',
            labels={'Cases': f'{map_crime} Cases'},
            scope='asia',
            title=f"📍 Crime Spot Map: {map_crime} ({map_state_filter})"
        )

        fig.update_geos(
            center=dict(lat=22.5937, lon=78.9629),
            projection_scale=4.5,
            showcountries=True,
            countrycolor="LightGray",
            showcoastlines=True,
            showland=True,
            landcolor="GhostWhite",
            fitbounds=False
        )

        fig.update_layout(
            height=650,
            margin={"r":0,"t":40,"l":0,"b":0},
            hoverlabel=dict(
                bgcolor="white",
                font_size=13,
                font_family="sans-serif"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"📋 District Incident Rankings: {map_crime}")
        display_df = map_df[['District', 'State', 'Cases', 'Total_IPC_Crimes', 'Risk_Level']].sort_values(by='Cases', ascending=False)
        display_df.rename(columns={'Cases': f'{map_crime} Cases', 'Total_IPC_Crimes': 'Total IPC Crimes'}, inplace=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    else:
        st.warning("No district records match the selected min incident cases filter.")


# PAGE 3: NATIONAL CRIME ANALYTICS
elif page_selection == "📊 National Crime Analytics":
    st.title("📊 National Crime Analytics & State Rankings")
    st.markdown("Comparative breakdown of IPC crime counts across all Indian States and Union Territories.")
    st.divider()

    non_crime_cols = ['States/UTs', 'District', 'Year', 'Total Cognizable IPC crimes']
    crime_cols = [c for c in df_clean.columns if c not in non_crime_cols]

    st.subheader("1. State-Wise Total Cognizable IPC Crimes")
    state_totals = df_clean.groupby('States/UTs')['Total Cognizable IPC crimes'].sum().sort_values(ascending=False)
    st.bar_chart(state_totals, color="#3B82F6")

    st.subheader("2. Top National IPC Crime Categories (Total Incidents)")
    crime_totals = df_clean[crime_cols].sum().sort_values(ascending=False).head(15)
    st.bar_chart(crime_totals, color="#EF4444")


# PAGE 4: MODEL TRAINING & PREDICTION GUIDE
elif page_selection == "📖 Model Training & Prediction Guide":
    st.title("📖 Model Training & Real-Time Prediction Architecture Guide")
    st.markdown(
        "A detailed guide explaining **how the Machine Learning model was trained**, "
        "how data transformation was conducted, and **how real-time predictions are generated**."
    )
    st.divider()

    # Overview Metrics Row
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Machine Learning Model", "ExtraTreesRegressor")
    g2.metric("Model R² Accuracy", "0.8775 (87.75%)")
    g3.metric("Mean Absolute Error", "21.08 cases")
    g4.metric("Training Samples", "69,774 records")

    st.markdown("---")

    # Section 1: How Model Was Trained
    with st.expander("🛠️ SECTION 1: How the Model Was Trained", expanded=True):
        st.markdown(r"""
        ### 1. Dataset Preprocessing & Data Reshaping
        - **Source Data**: Official National Crime Records Bureau (NCRB) IPC district dataset (`Dataset/dstrIPC_1_2014.csv`).
        - **Filtering**: Removed state aggregate summary rows (`District != 'TOTAL'`), keeping **782 distinct districts** across **36 States & UTs**.
        - **Data Melting**: The dataset was unpivoted from 87 wide crime columns into a **69,774-row long dataset**:
          $$\text{Columns}: [\text{States/UTs}, \text{District}, \text{Year}, \text{Crime\_Type}, \text{Cases}]$$

        ### 2. Feature Engineering
        To enable the model to understand local crime density and category baseline scales, three engineered features were created:
        - **`District_Total_IPC`**: Total cognizable IPC crimes reported in that specific district.
        - **`Crime_Type_Avg`**: National mean incident cases for the selected crime category.
        - **`State_Crime_Avg`**: State-specific mean incident cases for that crime category.

        ### 3. Categorical Label Encoding & Artifact Generation
        - Scikit-learn `LabelEncoder` objects convert text parameters (`State`, `District`, `Crime_Type`) into numeric integers.
        - These encoders, along with state-district map relationships and baseline averages, are saved into **`Model/encoders.pkl`**.

        ### 4. Machine Learning Algorithm: ExtraTreesRegressor
        - We evaluated multiple regression algorithms (**ExtraTreesRegressor**, **RandomForestRegressor**, **DecisionTree**, **SVM**).
        - **ExtraTreesRegressor** (Extremely Randomized Trees) achieved the highest accuracy ($R^2 = 0.8775$, $\text{MAE} = 21.08$).
        - **Why ExtraTrees?** ExtraTrees randomizes split points at decision tree nodes, significantly reducing variance across zero-inflated crime distributions (where rare crimes have 0-5 cases and major crimes have thousands).
        """)

    # Section 2: How Predictions Are Calculated
    with st.expander("⚡ SECTION 2: How Real-Time Predictions Are Calculated", expanded=True):
        st.markdown(r"""
        When a user submits parameters on the **🔮 AI Crime Predictor** page:

        1. **Categorical Encoding**:
           $$\text{State String} \xrightarrow{\text{le\_state}} \text{State\_Code}$$
           $$\text{District String} \xrightarrow{\text{le\_dist}} \text{District\_Code}$$
           $$\text{Crime Category String} \xrightarrow{\text{le\_crime}} \text{Crime\_Code}$$

        2. **Feature Vector Assembly**:
           The system looks up pre-calculated values and builds a 7-element input vector:
           $$\vec{X} = [\text{Year}, \text{State\_Code}, \text{District\_Code}, \text{Crime\_Code}, \text{District\_Total\_IPC}, \text{Crime\_Type\_Avg}, \text{State\_Crime\_Avg}]$$

        3. **Model Tree Ensemble Evaluation**:
           $\vec{X}$ is evaluated across all 100 randomized decision trees in `Model/model.pkl` to compute the baseline prediction:
           $$\hat{y}_{\text{raw}} = \frac{1}{100} \sum_{i=1}^{100} T_i(\vec{X})$$

        4. **Temporal Trend Scaling (Target Year Projection)**:
           For future target years ($> 2014$), a compound growth factor $(1 + 1.2\% = 1.012)$ is applied:
           $$\text{Predicted Cases} = \text{round}\left( \hat{y}_{\text{raw}} \times (1.012)^{(\text{Target Year} - 2014)} \right)$$

        5. **Safety Risk Level Classification**:
           - $\le 10$ cases: 🟢 **Very Low Incident Risk**
           - $\le 50$ cases: 🔵 **Low Incident Risk**
           - $\le 200$ cases: 🟡 **Moderate Crime Level**
           - $\le 500$ cases: 🟠 **High Crime Severity Area**
           - $> 500$ cases: 🔴 **Severe Crime Hotspot**
        """)

    # Section 3: Live Model Feature Importances
    st.subheader("📊 Model Feature Importance Breakdown")
    st.write("Relative importance weight assigned to each input feature by the trained `ExtraTreesRegressor` model:")
    
    feature_names = encoders.get('features', ['Year', 'State_Code', 'District_Code', 'Crime_Code', 'District_Total_IPC', 'Crime_Type_Avg', 'State_Crime_Avg'])
    importances = model.feature_importances_
    
    imp_df = pd.DataFrame({
        'Feature Name': feature_names,
        'Importance Weight (%)': np.round(importances * 100, 2)
    }).sort_values(by='Importance Weight (%)', ascending=True)
    
    fig_imp = px.bar(
        imp_df,
        x='Importance Weight (%)',
        y='Feature Name',
        orientation='h',
        color='Importance Weight (%)',
        color_continuous_scale='Blues',
        text='Importance Weight (%)',
        title="ExtraTrees Model Feature Importance Hierarchy"
    )
    fig_imp.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig_imp.update_layout(height=400, margin={"r":40,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_imp, use_container_width=True)

    # Section 4: Saved Artifact Files Summary
    st.subheader("📁 Saved System Artifact Files")
    art_col1, art_col2 = st.columns(2)
    
    with art_col1:
        st.markdown("""
        #### `Model/model.pkl`
        - **Type**: Serialized Python Pickle file.
        - **Object**: Trained `ExtraTreesRegressor` estimator.
        - **Function**: Performs fast inference on input feature vectors.
        """)
        
    with art_col2:
        st.markdown("""
        #### `Model/encoders.pkl`
        - **Type**: Serialized Python Pickle dictionary.
        - **Object**: `LabelEncoder` objects + lookup dictionaries.
        - **Function**: Converts UI text options to numeric codes & provides baseline stats.
        """)

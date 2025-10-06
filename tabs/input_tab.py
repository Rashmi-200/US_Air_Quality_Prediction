import streamlit as st
import pandas as pd
import joblib

@st.cache_data
def load_dataset():
    df = pd.read_csv("data/US_air_pollution_dataset_2000_2023.csv")
    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))

    # Keep only recent years
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"].dt.year >= 2015]
    return df

@st.cache_resource
def load_models():
    reg_model = joblib.load("models/aqi_regressor.pkl")
    clf_model = joblib.load("models/air_quality_model.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return reg_model, clf_model, label_encoder

def show_input_tab():
    st.markdown("""
    <div style="background: #2c3e50; padding: 20px; border-radius: 10px; margin-bottom: 25px; border-left: 6px solid #3498db;">
        <h2 style="color: white; margin: 0; font-family: 'Arial', sans-serif;">📥 Input Features</h2>
        <p style="color: #ecf0f1; margin: 5px 0 0 0; font-size: 14px;">Select State → County → City and Date to generate prediction inputs</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_dataset()
    reg_model, clf_model, le = load_models()

    # Simple container without fancy gradients
    st.markdown("""
    <div style="background: white; padding: 20px; border-radius: 8px; border: 2px solid #bdc3c7; margin-bottom: 20px;">
    """, unsafe_allow_html=True)
    
    # Basic two-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        states = sorted(df["State"].unique())
        state = st.selectbox(
            "🌎 Select State", 
            states,
            help="Choose the state for air quality analysis"
        )
        
        df_state = df[df["State"] == state]
        counties_raw = sorted(df_state["County"].unique())
        counties = [f"{c} County" for c in counties_raw]
        county = st.selectbox(
            "🏞️ Select County", 
            counties,
            help="Select the county within the chosen state"
        )
        county_clean = county.replace(" County", "")
    
    with col2:
        df_filtered = df_state[df_state["County"] == county_clean]
        cities_raw = sorted(df_filtered["City"].unique())
        cities = [f"{c} City" for c in cities_raw]
        city = st.selectbox(
            "🏙️ Select City", 
            cities,
            help="Choose the specific city for prediction"
        )
        city_clean = city.replace(" City", "")
        
        date = st.date_input(
            "📅 Select Date",
            help="Choose the date for air quality prediction"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Center the button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "🔮 Generate Prediction", 
            use_container_width=True,
            key="prediction_btn"
        ):
            with st.spinner("Calculating air quality prediction..."):
                input_row = pd.DataFrame({
                    "Date": [pd.to_datetime(date)],
                    "State": [state],
                    "County": [county_clean],
                    "City": [city_clean]
                })

                input_row["Year"] = input_row["Date"].dt.year
                input_row["Month"] = input_row["Date"].dt.month
                input_row["Day"] = input_row["Date"].dt.day
                input_row = input_row.drop(columns=["Date"])

                # --- Predict pollutant metrics (regression) ---
                y_pred_reg = reg_model.predict(input_row)

                # Map predictions to column names
                pollutant_targets = [
                    "O3 Mean", "O3 1st Max Value", "O3 AQI",
                    "CO Mean", "CO 1st Max Value", "CO AQI",
                    "SO2 Mean", "SO2 1st Max Value", "SO2 AQI",
                    "NO2 Mean", "NO2 1st Max Value", "NO2 AQI"
                ]
                predicted_metrics = dict(zip(pollutant_targets, y_pred_reg[0]))

                # Save for prediction tab
                st.session_state.input_values = predicted_metrics
                st.session_state.prediction_made = True
                st.session_state.location_info = {
                    "region": state,
                    "city": city_clean
                }

            # Simple success message
            st.success("✅ Prediction generated! Please check the **Prediction tab** for results.")
            
            # Basic location summary
            st.info(f"📍 **Selected Location:** {city_clean}, {county_clean}, {state} | **Date:** {date}")

    # Simple footer note
    st.markdown("---")
    st.caption("Note: Predictions are based on historical data and machine learning models trained on US air quality data.")
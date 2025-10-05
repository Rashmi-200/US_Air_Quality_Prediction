import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Function to categorize AQI values
def categorize_aqi(aqi_value):
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy_Sensitive"
    elif aqi_value <= 200:
        return "Unhealthy"
    else:
        return "Very_Unhealthy"

def load_and_preprocess_data(filepath, start_year=2015):
    df = pd.read_csv(filepath)

    # --- Data Cleaning ---
    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))

    # --- Use only recent years for future prediction ---
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"].dt.year >= start_year]

    # --- Compute Overall AQI ---
    df["Overall_AQI"] = df[["O3 AQI", "CO AQI", "SO2 AQI", "NO2 AQI"]].max(axis=1)

    # --- Create Category ---
    df["AQI_Category"] = df["Overall_AQI"].apply(categorize_aqi)

    # Encode Category
    le = LabelEncoder()
    df["AQI_Category"] = le.fit_transform(df["AQI_Category"])

    return df, le

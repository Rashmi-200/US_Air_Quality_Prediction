import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Use Streamlit theme + Seaborn style
plt.style.use("seaborn-v0_8")

# Cache data for faster loading
@st.cache_data
def load_data():
    df = pd.read_csv("data/US_air_pollution_dataset_2000_2023.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    return df

def show_eda_tab():
    st.header("✨ Exploratory Data Analysis (EDA)")
    st.markdown("""
    <div style="background:#3498db;padding:12px;border-radius:8px;color:white;">
        <h4 style="margin:0;">Understanding the Dataset</h4>
        <p style="margin:0;">Explore pollutant behavior, seasonal patterns, and long-term air quality trends 
        using interactive filters.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load dataset
    df = load_data()

    # ----------------------------
    # Sidebar Filters
    # ----------------------------
    st.sidebar.header("🔎 EDA Filters")
    year = st.sidebar.selectbox("📆 Select Year", options=["All"] + sorted(df["Year"].unique().tolist()))
    state = st.sidebar.selectbox("🌎 Select State", options=["All"] + sorted(df["State"].unique().tolist()))
    county = st.sidebar.selectbox("🏛 Select County", options=["All"] + sorted(df["County"].unique().tolist()))
    city = st.sidebar.selectbox("🏙 Select City", options=["All"] + sorted(df["City"].unique().tolist()))

    # Apply filters
    filtered_df = df.copy()
    if year != "All":
        filtered_df = filtered_df[filtered_df["Year"] == year]
    if state != "All":
        filtered_df = filtered_df[filtered_df["State"] == state]
    if county != "All":
        filtered_df = filtered_df[filtered_df["County"] == county]
    if city != "All":
        filtered_df = filtered_df[filtered_df["City"] == city]

    st.subheader("📄 Dataset Preview (After Filtering)")
    st.dataframe(filtered_df.head(10), use_container_width=True)
    st.info(f"📊 Total Rows after filtering: **{filtered_df.shape[0]}**")

    # ----------------------------
    # Pollutant Distributions
    # ----------------------------
    st.subheader("🌈 Pollutant Distributions")
    pollutants = ["O3 AQI", "CO AQI", "SO2 AQI", "NO2 AQI"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ["#1abc9c", "#e67e22", "#9b59b6", "#3498db"]
    for i, pollutant in enumerate(pollutants):
        sns.histplot(filtered_df[pollutant], bins=30, ax=axes[i//2, i%2], kde=True, color=colors[i])
        axes[i//2, i%2].set_title(f"Distribution of {pollutant}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)

    # ----------------------------
    # Correlation Heatmap
    # ----------------------------
    st.subheader("🔥 Correlation Between Pollutants")
    corr = filtered_df[pollutants].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, cbar_kws={"shrink": 0.8})
    plt.title("Pollutant Correlation Heatmap", fontsize=12, fontweight="bold")
    st.pyplot(fig)

    # ----------------------------
    # Seasonal Trends
    # ----------------------------
    st.subheader("🌤 Seasonal Trends (Monthly Average)")
    monthly_avg = filtered_df.groupby("Month")[pollutants].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_avg.plot(ax=ax, marker="o", linewidth=2)
    plt.title("Average Pollutant Levels by Month", fontsize=12, fontweight="bold")
    plt.xlabel("Month")
    plt.ylabel("Pollutant Level (AQI)")
    st.pyplot(fig)

    # ----------------------------
    # Long-term Trends
    # ----------------------------
    st.subheader("📈 Long-Term Trends (Yearly Average)")
    yearly_avg = filtered_df.groupby("Year")[pollutants].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    yearly_avg.plot(ax=ax, marker="o", linewidth=2)
    plt.title("Average Pollutant Levels by Year", fontsize=12, fontweight="bold")
    plt.xlabel("Year")
    plt.ylabel("Pollutant Level (AQI)")
    st.pyplot(fig)

    # ----------------------------
    # Download Option
    # ----------------------------
    st.subheader("📥 Download Filtered Data")
    st.download_button(
        label="⬇️ Download as CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_air_quality_data.csv",
        mime="text/csv"
    )

    # ----------------------------
    # Insights Box
    # ----------------------------
    st.markdown("""
    <div style="background:#2ecc71;padding:12px;border-radius:8px;color:white;">
        <h4 style="margin:0;">📌 Quick Insights</h4>
        <ul>
            <li>📊 Distributions highlight pollutant variability and possible outliers.</li>
            <li>🔥 Correlations show how pollutants influence each other (e.g., traffic-related pollutants).</li>
            <li>🌤 Seasonal plots reveal higher O3 in summer and more CO in winter.</li>
            <li>📈 Long-term trends show whether air quality is improving or worsening over the years.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

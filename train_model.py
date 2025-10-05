import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.multiclass import unique_labels
from preprocess import load_and_preprocess_data

# --- Load Data (all rows from 2015 onwards) ---
df, le = load_and_preprocess_data(
    "data/US_air_pollution_dataset_2000_2023.csv",
    start_year=2015
)

# -------------------------------
# Part 1: Regression (predict pollutant metrics)
# -------------------------------
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

X_reg = df[["Year", "Month", "Day", "State", "County", "City"]]

# ✅ Targets (exclude 1st Max Hour columns)
y_reg = df[[
    "O3 Mean", "O3 1st Max Value", "O3 AQI",
    "CO Mean", "CO 1st Max Value", "CO AQI",
    "SO2 Mean", "SO2 1st Max Value", "SO2 AQI",
    "NO2 Mean", "NO2 1st Max Value", "NO2 AQI"
]]

categorical_features = ["State", "County", "City"]
numeric_features = ["Year", "Month", "Day"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
reg_pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                               ("regressor", regressor)])
reg_pipeline.fit(X_reg, y_reg)

joblib.dump(reg_pipeline, "models/aqi_regressor.pkl")
print("✅ Regression model saved: models/aqi_regressor.pkl")

# -------------------------------
# Part 2: Classification (AQI Category)
# -------------------------------
feature_cols = ["O3 AQI", "CO AQI", "SO2 AQI", "NO2 AQI"]
X_cls = df[feature_cols]
y_cls = df["AQI_Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, dt_model.predict(X_test)))

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Test Accuracy:", accuracy_score(y_test, y_pred_rf))

# ✅ Classification Report
labels_in_data = unique_labels(y_test, y_pred_rf)
class_names = le.inverse_transform(labels_in_data)

print("\nClassification Report:\n", classification_report(
    y_test,
    y_pred_rf,
    labels=labels_in_data,
    target_names=class_names,
    zero_division=0
))

# Feature Importance Plot
importances = rf_model.feature_importances_
plt.bar(feature_cols, importances)
plt.title("Feature Importance (Random Forest)")
plt.ylabel("Importance")
plt.show()

# Save classifier + label encoder
joblib.dump(rf_model, "models/air_quality_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
print("✅ Classifier & label encoder saved in 'models/'")

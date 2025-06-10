import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Load Data
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data

df, data = load_data()

# Sidebar - Input sliders
st.sidebar.header("Input Tumor Features")

def user_input_features():
    input_data = {}
    for feature in data.feature_names:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        input_data[feature] = st.sidebar.slider(feature, min_val, max_val, mean_val)
    return pd.DataFrame([input_data])

user_input = user_input_features()

# Title
st.title("🩺 Breast Cancer Diagnosis Predictor")
st.markdown("Predict whether a tumor is **malignant** or **benign** based on diagnostic features.")

# Train Model
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Scale user input
user_input_scaled = scaler.transform(user_input)

# Prediction
prediction = model.predict(user_input_scaled)
prediction_proba = model.predict_proba(user_input_scaled)

# Output Result
st.subheader("Prediction Result")
target_names = {0: "Malignant", 1: "Benign"}
st.write(f"**Prediction:** {target_names[prediction[0]]}")
st.write(f"**Confidence:** {prediction_proba[0][prediction[0]] * 100:.2f}%")

# Show Feature Importance
st.subheader("Top 10 Important Features")
importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)

st.bar_chart(importance_df.set_index('Feature'))

# Optional: Display raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Breast Cancer Dataset")
    st.write(df.head())
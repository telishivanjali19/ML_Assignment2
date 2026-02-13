import streamlit as st
import pandas as pd
import joblib

st.title("📱 Mobile Price Classification App")

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    scaler = joblib.load("scaler.pkl")
    model = joblib.load(f"{model_name}.pkl")

    # Scale features
    X_scaled = scaler.transform(data)

    # Predict
    predictions = model.predict(X_scaled)

    data["Predicted Price Range"] = predictions

    st.subheader("Predictions")
    st.dataframe(data)

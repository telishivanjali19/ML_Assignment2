import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

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

    X = data.drop("price_range", axis=1)
    y = data["price_range"]

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))

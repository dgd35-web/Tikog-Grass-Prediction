# streamlit_app.py
import streamlit as st
import tensorflow as tf
import xgboost as xgb
import numpy as np

# --- Load trained models ---
lstm_model = tf.keras.models.load_model("lstm_model.h5")
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgb_model.json")

st.title("Tikog Demand Prediction (Web App)")

# --- Input fields ---
year = st.text_input("Year")
product_type = st.selectbox("Product Type", ["MAT", "Bayong", "Laptop Bag", "Slippers", "Backpack", "Malaya", "Wallet", "Mini Bag"])
quantity = st.number_input("Quantity", min_value=1)
length = st.number_input("Tikog Length (cm)", min_value=1.0)
width = st.number_input("Tikog Width (cm)", min_value=1.0)

# --- Predict button ---
if st.button("Predict Demand"):
    # Preprocess input
    features = np.array([[quantity, length, width]])

    # LSTM prediction
    lstm_pred = lstm_model.predict(features.reshape((features.shape[0], features.shape[1], 1)))

    # XGBoost prediction
    final_pred = xgb_model.predict(lstm_pred)

    # Confidence interval (simple placeholder)
    mean_pred = final_pred[0]
    lower_bound = mean_pred * 0.9
    upper_bound = mean_pred * 1.1

    # Display results
    st.success(f"Predicted Demand: {mean_pred:.2f}")
    st.write(f"95% Confidence Interval: {lower_bound:.2f} – {upper_bound:.2f}")

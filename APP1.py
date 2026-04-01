# streamlit_app.py
import streamlit as st
import tensorflow as tf
import xgboost as xgb
import numpy as np

# Load models
lstm_model = tf.keras.models.load_model("lstm_model.h5")
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgb_model.json")

st.title("Tikog Demand Prediction")

year = st.text_input("Year")
product_type = st.selectbox("Product Type", ["MAT", "Bayong", "Wallet", "Backpack"])
quantity = st.number_input("Quantity", min_value=1)
length = st.number_input("Tikog Length (cm)", min_value=1.0)
width = st.number_input("Tikog Width (cm)", min_value=1.0)

if st.button("Predict Demand"):
    features = np.array([[quantity, length, width]])
    lstm_pred = lstm_model.predict(features.reshape((features.shape[0], features.shape[1], 1)))
    final_pred = xgb_model.predict(lstm_pred)
    st.success(f"Predicted Demand: {final_pred[0]:.2f}")

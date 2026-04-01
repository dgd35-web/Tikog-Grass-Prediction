import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb

# Load models
lstm_model = tf.keras.models.load_model("lstm_model.keras")
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgb_model.json")

st.title("Tikog Requirement Prediction Application")
st.write("Enter the following details to predict the required Tikog for your product:")

product_sides = {"Basket": 1, "Mat": 1, "Bag": 2, "Slippers": 2, "Wallet": 2, "Others": 1}

dimension_options = {
    "27 inches x 16 inches": (27, 16),
    "11 inches x 14 ½ inches": (11, 14.5),
    "12 inches x 7 ½ inches x 3 ½ inches": (12, 7.5),
    "Body = 17 ½ x 2, packet (11 ½ x 11 ½), side (5 x 6)": (17.5, 2),
    "29 inches x 22 inches": (29, 22)
}

dimension = st.selectbox("Dimension", options=list(dimension_options.keys()) + ["Custom"])
if dimension != "Custom":
    length, width = dimension_options[dimension]
else:
    length = st.number_input("Length (in inches)", min_value=0.0, step=0.1)
    width = st.number_input("Width (in inches)", min_value=0.0, step=0.1)

quantity = st.text_input("Quantity", "10")
product_type = st.selectbox("Product Type", ["Basket", "Mat", "Bag", "Slippers", "Wallet", "Others"])
sales_trend = st.selectbox("Sales Trend", ["Increasing", "Stable", "Decreasing"])

if st.button("Predict"):
    try:
        quantities = [int(q.strip()) for q in quantity.split(",")]
        total_quantity = sum(quantities)

        # Build features
        features = np.array([[total_quantity, length, width]])

        # Predictions
        lstm_pred = lstm_model.predict(features.reshape((features.shape[0], features.shape[1], 1)))
        final_pred = xgb_model.predict(lstm_pred)

        sides = product_sides.get(product_type, 1)
        final_tikog_needed = final_pred[0] * sides

        st.success(f"Prediction: {final_tikog_needed:.2f} units of Tikog required")

        st.write("### Breakdown")
        st.write(f"LSTM raw prediction: {lstm_pred[0][0]:.2f}")
        st.write(f"XGBoost adjusted prediction: {final_pred[0]:.2f}")
        st.write(f"Number of sides: {sides}")
        st.write(f"Total quantity: {total_quantity}")

        st.write("### Details")
        st.write(f"Dimension: {dimension}")
        st.write(f"Length: {length} inches")
        st.write(f"Width: {width} inches")
        st.write(f"Product Type: {product_type}")
        st.write(f"Sales Trend: {sales_trend}")

    except ValueError:
        st.error("Please enter valid integers separated by commas in the Quantity field.")

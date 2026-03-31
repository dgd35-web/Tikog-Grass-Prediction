import streamlit as st
import pandas as pd
import numpy as np

# --- Inputs ---
length = st.number_input("Length")
width = st.number_input("Width")
quantity = st.number_input("Quantity")
number_of_sides = st.number_input("Number of Sides", value=1)

product_type = st.selectbox("Product Type", ["mat", "bag", "hat"])

# Encode product type (example)
product_map = {"mat": 0, "bag": 1, "hat": 2}
product_type_encoded = product_map[product_type]

# --- Deterministic Calculation ---
base_tikog_per_side = 216  # or your formula
tikog_per_product = base_tikog_per_side * number_of_sides
deterministic_tikog = tikog_per_product * quantity

# --- Display breakdown ---
st.subheader("Breakdown")
st.write(f"Base Tikog per side: {base_tikog_per_side}")
st.write(f"Tikog per product: {tikog_per_product}")
st.write(f"Deterministic total: {deterministic_tikog}")

# --- ML Prediction ---
if st.button("Predict with Hybrid Model"):

    # Prepare features
    features = np.array([
        length,
        width,
        quantity,
        product_type_encoded,
        tikog_per_product,
        deterministic_tikog
    ])

    # --- LSTM (optional) ---
    lstm_output = lstm_model.predict(features.reshape(1, 1, -1))[0]

    # Combine
    final_features = np.append(features, lstm_output).reshape(1, -1)

    # --- XGBoost Residual ---
    predicted_residual = xgb_model.predict(final_features)[0]

    # --- Final Tikog ---
    final_tikog = deterministic_tikog + predicted_residual

    st.subheader("Final Prediction")
    st.success(f"Estimated Tikog Needed: {round(final_tikog, 2)}")
    except ValueError:
        st.error(
            "Please enter valid integers separated by commas in the Quantity field."
        )

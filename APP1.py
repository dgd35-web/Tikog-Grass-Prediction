import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# Try to import TensorFlow only if available
try:
    from tensorflow.keras.models import load_model
    tensorflow_available = True
except ImportError:
    tensorflow_available = False

st.set_page_config(page_title="Tikog Grass Prediction", page_icon="🌿")

st.title("Tikog Grass Requirement Prediction")
st.write("Enter the product details to estimate the tikog grass needed.")

# --- Inputs ---
length = st.number_input("Length", min_value=0.0, value=1.0)
width = st.number_input("Width", min_value=0.0, value=1.0)
quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
number_of_sides = st.number_input("Number of Sides", min_value=1, value=1, step=1)

product_type = st.selectbox("Product Type", ["mat", "bag", "hat"])

# --- Encode product type ---
product_map = {"mat": 0, "bag": 1, "hat": 2}
product_type_encoded = product_map[product_type]

# --- Density by product type ---
if product_type == "mat":
    density = 0.50
elif product_type == "bag":
    density = 0.70
else:  # hat
    density = 0.60

# --- Deterministic Calculation ---
base_tikog_per_side = length * width * density
tikog_per_product = base_tikog_per_side * number_of_sides
deterministic_tikog = tikog_per_product * quantity

# --- Display breakdown ---
st.subheader("Breakdown")
st.write(f"Length: {length}")
st.write(f"Width: {width}")
st.write(f"Quantity: {quantity}")
st.write(f"Number of Sides: {number_of_sides}")
st.write(f"Product Type: {product_type}")
st.write(f"Density Used: {density}")
st.write(f"Base Tikog per Side: {round(base_tikog_per_side, 2)}")
st.write(f"Tikog per Product: {round(tikog_per_product, 2)}")
st.write(f"Deterministic Total: {round(deterministic_tikog, 2)}")

# --- Optional dataset preview ---
if os.path.exists("Filtered_Data.csv"):
    try:
        df = pd.read_csv("Filtered_Data.csv")
        with st.expander("Preview Dataset"):
            st.dataframe(df.head())
    except Exception as e:
        st.warning(f"Could not load Filtered_Data.csv: {e}")

# --- Load models if available ---
lstm_model = None
xgb_model = None

if tensorflow_available and os.path.exists("lstm_model.h5"):
    try:
        lstm_model = load_model("lstm_model.h5")
    except Exception as e:
        st.warning(f"Could not load LSTM model: {e}")

if os.path.exists("xgb_model.pkl"):
    try:
        xgb_model = joblib.load("xgb_model.pkl")
    except Exception as e:
        st.warning(f"Could not load XGBoost model: {e}")

# --- Prediction Button ---
if st.button("Predict Tikog Needed"):
    try:
        features = np.array([
            length,
            width,
            quantity,
            product_type_encoded,
            tikog_per_product,
            deterministic_tikog
        ], dtype=float)

        # If both models exist, use hybrid prediction
        if lstm_model is not None and xgb_model is not None:
            lstm_output = lstm_model.predict(features.reshape(1, 1, -1), verbose=0)[0]

            if np.isscalar(lstm_output):
                lstm_output = np.array([lstm_output])

            final_features = np.append(features, lstm_output).reshape(1, -1)
            predicted_residual = xgb_model.predict(final_features)[0]
            final_tikog = deterministic_tikog + predicted_residual

            st.subheader("Final Hybrid Prediction")
            st.success(f"Estimated Tikog Needed: {round(final_tikog, 2)}")
            st.info("Hybrid model used: Deterministic + LSTM + XGBoost")

        else:
            # Fallback if models are missing
            st.subheader("Final Prediction")
            st.success(f"Estimated Tikog Needed: {round(deterministic_tikog, 2)}")
            st.info("Only deterministic calculation was used because model files are not available.")

    except ValueError:
        st.error("Please enter valid numeric input values.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

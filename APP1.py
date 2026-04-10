import streamlit as st
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
from sklearn.metrics import r2_score

# --- Load models and encoder ---
lstm_model = tf.keras.models.load_model("lstm_model.keras")

xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgb_model.json")

encoder = joblib.load("product_encoder.pkl")

st.title("Tikog Requirement Prediction Application")
st.write("Enter the following details to predict the required Tikog demand (pieces) for your product:")

# Define product sides (original logic kept)
product_sides = {
    "Basket": 1,
    "Mat": 1,
    "Bag": 2,
    "Slippers": 1,   # no side
    "Wallet": 1,     # no side override
    "Others": 1
}

# Map simplified product types to encoder categories
product_mapping = {
    "Mat": "mat",
    "Basket": "bag",            # kept as in your original code
    "Bag": "bag",
    "Slippers": "slippers",
    "Wallet": "divider_wallet",
    "Others": "pouch"
}

# Dimension options
dimension_options = {
    "27 inches x 16 inches": (27, 16),
    "11 inches x 14 ½ inches": (11, 14.5),
    "12 inches x 7 ½ inches x 3 ½ inches": (12, 7.5),
    "Body = 17 ½ x 2, packet (11 ½ x 11 ½), side (5 x 6)": (17.5, 2),
    "29 inches x 22 inches": (29, 22)
}

# --- Input form ---
dimension = st.selectbox("Dimension", options=list(dimension_options.keys()) + ["Custom"])

if dimension != "Custom":
    length, width = dimension_options[dimension]
else:
    length = st.number_input("Length (in inches)", min_value=0.0, step=0.1)
    width = st.number_input("Width (in inches)", min_value=0.0, step=0.1)

quantity = st.number_input("Quantity", min_value=1, step=1)
product_type = st.selectbox("Product Type", ["Basket", "Mat", "Bag", "Slippers", "Wallet", "Others"])

if st.button("Predict"):
    try:
        total_quantity = int(quantity)

        # --------------------------------------------------
        # ONLY override Slippers and Wallet
        # --------------------------------------------------
        if product_type in ["Slippers", "Wallet"]:
            tikog_per_unit = 40
            total_tikog = tikog_per_unit * total_quantity
            conversion_factor = 2.46
            strand_length_ft = total_tikog * conversion_factor

        else:
            # --------------------------------------------------
            # ORIGINAL computation for all other products
            # --------------------------------------------------

            # Map user-friendly product type to encoder category
            mapped_type = product_mapping.get(product_type, product_type)

            # Encode product type safely
            product_encoded = encoder.transform([[mapped_type]])
            product_encoded = np.array(product_encoded)

            # Numeric features reshaped to 2D
            numeric_features = np.array([[length, width, total_quantity]])

            # Combine in same order as training: product_encoded first, then numeric
            features = np.hstack([product_encoded, numeric_features])

            # Predictions
            nn_pred = lstm_model.predict(features, verbose=0)
            final_pred = xgb_model.predict(nn_pred)

            # Keep original side logic for non-wallet/non-slippers
            sides = product_sides.get(product_type, 1)
            tikog_per_unit = final_pred[0] * sides

            # Keep original scaling for non-wallet/non-slippers
            scaling_factor = 130.4
            tikog_per_unit = tikog_per_unit * scaling_factor

            # Keep original total computation
            total_tikog = tikog_per_unit * total_quantity

            # Conversion factor
            conversion_factor = 2.46
            strand_length_ft = total_tikog * conversion_factor

        # --- Display results ---
        st.success(f"Total Tikog Needed: {int(round(total_tikog))} pieces")

        results = {
            "Product Type": product_type,
            "Dimension": f"{length} x {width} inches",
            "Tikog per Unit (pieces)": int(round(tikog_per_unit)),
            "Quantity": total_quantity,
            "Total Tikog (pieces)": int(round(total_tikog)),
            "Total Strand Length (ft)": int(round(strand_length_ft))
        }
        st.table(results)

        # --- Compute and display R² accuracy ---
        try:
            X_test, y_test = joblib.load("test_data.pkl")
            y_pred_test = xgb_model.predict(X_test)
            r2 = r2_score(y_test, y_pred_test)
            st.info(f"Model Prediction Accuracy (R²): {r2:.3f}")
        except Exception:
            st.warning("R² score could not be computed. Ensure test_data.pkl exists.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

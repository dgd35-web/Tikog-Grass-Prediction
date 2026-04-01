# regenerate_encoder.py
from sklearn.preprocessing import OneHotEncoder
import joblib

# Define all product types you want supported
product_types = [["Basket"], ["Mat"], ["Bag"], ["Slippers"], ["Wallet"], ["Others"]]

# Create encoder with handle_unknown="ignore"
encoder = OneHotEncoder(handle_unknown="ignore")

# Fit encoder on all categories
encoder.fit(product_types)

# Save updated encoder
joblib.dump(encoder, "product_encoder.pkl")

print("✅ Encoder updated and saved as product_encoder.pkl")

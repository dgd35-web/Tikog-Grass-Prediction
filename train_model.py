import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.constraints import MaxNorm
from xgboost import XGBRegressor

# --- STEP 1: Load demand data ---
detailed_df = pd.read_excel(r"C:\Users\User\Desktop\detailed_demand.xlsx")

# --- STEP 2: Preprocessing ---
# Percentile capping on demand
lower_cap = np.percentile(detailed_df["Tikog Demand (pieces)"], 1)
upper_cap = np.percentile(detailed_df["Tikog Demand (pieces)"], 99)
detailed_df["Tikog Demand (pieces)"] = np.clip(detailed_df["Tikog Demand (pieces)"], lower_cap, upper_cap)

# Yeo-Johnson transformation on target
pt = PowerTransformer(method="yeo-johnson")
detailed_df["Demand_transformed"] = pt.fit_transform(detailed_df[["Tikog Demand (pieces)"]])

# --- STEP 3: Feature Engineering ---
# Encode product type
encoder = OneHotEncoder(sparse_output=False)
product_encoded = encoder.fit_transform(detailed_df[["Product Type"]])

# Combine features: product type + dimensions + quantity
X = np.hstack([
    product_encoded,
    detailed_df[["Tikog Length (cm)", "Tikog Width (cm)", "Quantity"]].values
])
y = detailed_df["Demand_transformed"].values

# --- STEP 4: Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# --- STEP 5: Neural Network Training (Dense instead of LSTM) ---
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],), kernel_constraint=MaxNorm(3)))
model.add(Dropout(0.3))
model.add(Dense(32, activation="relu", kernel_constraint=MaxNorm(3)))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1, verbose=1)

# --- STEP 6: NN Predictions ---
nn_preds_train = model.predict(X_train)
nn_preds_test = model.predict(X_test)

# --- STEP 7: XGBoost Integration ---
xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
xgb.fit(nn_preds_train, y_train)

final_preds = xgb.predict(nn_preds_test)

# --- STEP 8: Evaluation ---
rmse_train = np.sqrt(mean_squared_error(y_train, xgb.predict(nn_preds_train)))
mae_train = mean_absolute_error(y_train, xgb.predict(nn_preds_train))
r2_train = r2_score(y_train, xgb.predict(nn_preds_train))

rmse_test = np.sqrt(mean_squared_error(y_test, final_preds))
mae_test = mean_absolute_error(y_test, final_preds)
r2_test = r2_score(y_test, final_preds)

results_df = pd.DataFrame({
    "Dataset": ["Training", "Testing"],
    "RMSE": [rmse_train, rmse_test],
    "MAE": [mae_train, mae_test],
    "R²": [r2_train, r2_test]
})

print("\nEvaluation Results:")
print(results_df)

# --- STEP 9: Save trained models ---
model.save("lstm_model.keras")        # modern format
xgb.save_model("xgb_model.json")

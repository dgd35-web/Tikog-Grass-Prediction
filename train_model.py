import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.constraints import MaxNorm
from xgboost import XGBRegressor

# --- STEP 1: Load demand data ---
detailed_df = pd.read_excel(r"C:\Users\User\Desktop\detailed_demand.xlsx")

# --- STEP 2: Preprocessing ---
# Percentile capping
lower_cap = np.percentile(detailed_df["Tikog Demand (pieces)"], 1)
upper_cap = np.percentile(detailed_df["Tikog Demand (pieces)"], 99)
detailed_df["Tikog Demand (pieces)"] = np.clip(detailed_df["Tikog Demand (pieces)"], lower_cap, upper_cap)

# Yeo-Johnson transformation
pt = PowerTransformer(method="yeo-johnson")
detailed_df["Demand_transformed"] = pt.fit_transform(detailed_df[["Tikog Demand (pieces)"]])

# --- STEP 3: Train/Test Split ---
train, test = train_test_split(detailed_df, test_size=0.2, shuffle=False)

# --- STEP 4: Data Augmentation ---
def sliding_window(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

def bootstrap_samples(X, y, n_samples):
    idx = np.random.randint(0, len(X), n_samples)
    return X[idx], y[idx]

window_size = 5
train_series = train["Demand_transformed"].values
X_train, y_train = sliding_window(train_series, window_size)

# Bootstrapping augmentation
X_train, y_train = bootstrap_samples(X_train, y_train, n_samples=len(X_train)*2)

test_series = test["Demand_transformed"].values
X_test, y_test = sliding_window(test_series, window_size)

# Reshape for LSTM [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# --- STEP 5: LSTM Training with MaxNorm ---
model = Sequential()
model.add(LSTM(64, input_shape=(window_size, 1),
               kernel_constraint=MaxNorm(3)))   # MaxNorm regularization
model.add(Dropout(0.3))                         # Dropout for regularization
model.add(Dense(32, activation="relu",
                kernel_constraint=MaxNorm(3)))  # MaxNorm also applied here
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
history = model.fit(X_train, y_train, epochs=200, batch_size=16,
                    validation_split=0.1, verbose=1)

# --- STEP 6: LSTM Predictions ---
lstm_preds_train = model.predict(X_train)
lstm_preds_test = model.predict(X_test)

# --- STEP 7: XGBoost Integration ---
xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
xgb.fit(lstm_preds_train, y_train)

final_preds = xgb.predict(lstm_preds_test)

# --- STEP 8: Evaluation ---
rmse_train = np.sqrt(mean_squared_error(y_train, xgb.predict(lstm_preds_train)))
mse_train = mean_squared_error(y_train, xgb.predict(lstm_preds_train))
mae_train = mean_absolute_error(y_train, xgb.predict(lstm_preds_train))
r2_train = r2_score(y_train, xgb.predict(lstm_preds_train))
wape_train = np.sum(np.abs(y_train - xgb.predict(lstm_preds_train))) / np.sum(np.abs(y_train))

rmse_test = np.sqrt(mean_squared_error(y_test, final_preds))
mse_test = mean_squared_error(y_test, final_preds)
mae_test = mean_absolute_error(y_test, final_preds)
r2_test = r2_score(y_test, final_preds)
wape_test = np.sum(np.abs(y_test - final_preds)) / np.sum(np.abs(y_test))

# Put results into a DataFrame with dataset info
results_df = pd.DataFrame({
    "Dataset": ["Training", "Testing"],
    "Data Used": [
        "80% of Tikog demand dataset",
        "20% of Tikog demand dataset"
    ],
    "RMSE": [rmse_train, rmse_test],
    "MSE": [mse_train, mse_test],
    "MAE": [mae_train, mae_test],
    "R²": [r2_train, r2_test],
    "WAPE": [wape_train, wape_test]
})

print("\nEvaluation Results with Dataset Info:")
print(results_df)

# Recommended modern format
model.save("lstm_model.keras")

# OR legacy format
model.save("lstm_model.h5")

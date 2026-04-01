# APP1.py
from flask import Flask, request, jsonify
import tensorflow as tf
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# --- Load trained models ---
lstm_model = tf.keras.models.load_model("lstm_model.h5")
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgb_model.json")

def prediction_interval(model, X, n_bootstrap=100, alpha=0.05):
    preds = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X), len(X), replace=True)
        X_resampled = X[idx]
        preds.append(model.predict(X_resampled))
    preds = np.array(preds)
    mean_pred = preds.mean(axis=0)
    lower = np.percentile(preds, 100*alpha/2, axis=0)
    upper = np.percentile(preds, 100*(1-alpha/2), axis=0)
    return mean_pred, lower, upper

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    quantity = float(data.get("quantity"))
    length = float(data.get("length"))
    width = float(data.get("width"))

    # --- Preprocess input ---
    features = np.array([[quantity, length, width]])

    # --- LSTM prediction ---
    lstm_pred = lstm_model.predict(features.reshape((features.shape[0], features.shape[1], 1)))

    # --- XGBoost prediction ---
    mean_pred, lower, upper = prediction_interval(xgb_model, lstm_pred, n_bootstrap=200)

    return jsonify({
        "predicted_demand": float(mean_pred[0]),
        "lower_bound": float(lower[0]),
        "upper_bound": float(upper[0]),
        "confidence_level": "95%"
    })

if __name__ == "__main__":
    app.run(debug=True)

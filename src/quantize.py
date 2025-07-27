import joblib
import os
import numpy as np
import json
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

model = joblib.load("model.joblib")
coef = model.coef_
intercept = model.intercept_

with open("config/config.json") as f:
    config = json.load(f)

unquant_params = {
    "coef": coef,
    "intercept": intercept
}
joblib.dump(unquant_params, "unquant_params.joblib")

# Perform manual quantization
def quantize(x, scale, zero_point):
    return np.clip(np.round(x / scale + zero_point), 0, 255).astype(np.uint8)

def dequantize(qx, scale, zero_point):
    return (qx.astype(np.float32) - zero_point) * scale

scale = config["quant_scale"]
zero_point = config["quant_zero_point"]

q_coef = quantize(coef, scale, zero_point)
q_intercept = quantize(np.array([intercept]), scale, zero_point)

quant_params = {
    "q_coef": q_coef,
    "q_intercept": q_intercept,
    "scale": scale,
    "zero_point": zero_point
}
joblib.dump(quant_params, "quant_params.joblib")

# Dequantize for inference
dq_coef = dequantize(q_coef, scale, zero_point)
dq_intercept = dequantize(q_intercept, scale, zero_point)[0]

data = fetch_california_housing()
X = data.data
y = data.target

# Inference using dequantized weights
y_pred = np.dot(X, dq_coef) + dq_intercept

print("Inference complete using manually quantized model.")
print("First 5 predictions:", y_pred[:5])

# Evaluate R² score
quant_r2 = r2_score(y, y_pred)
print(f"R² Score (Quantized Model): {quant_r2:.4f}")

# File size reporting
print("Model Size:")
print(f" - unquant_params.joblib: {os.path.getsize('unquant_params.joblib') / 1024:.2f} KB")
print(f" - quant_params.joblib:   {os.path.getsize('quant_params.joblib') / 1024:.2f} KB")

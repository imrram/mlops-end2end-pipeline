import joblib
import os
import json
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def train_model():
    # Load config
    with open("config/config.json") as f:
        config = json.load(f)

    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=config["test_size"], random_state=config["random_state"]
    )

    model = LinearRegression(fit_intercept=config["fit_intercept"])
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "model.joblib")
    print("Model saved to model.joblib")

    # Evaluate and print R² score on test data
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"R² Score (Sklearn Model): {score:.4f}")

if __name__ == "__main__":
    train_model()

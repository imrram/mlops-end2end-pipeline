import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import json
    
def train_model():
    # loading config
    with open("config/config.json") as f:
        config = json.load(f)

    data = fetch_california_housing()
    X_train, _, y_train, _ = train_test_split(data.data, data.target, test_size=config["test_size"], random_state=config["random_state"])

    model = LinearRegression(
        fit_intercept=config["fit_intercept"],
        normalize=config["normalize"]
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "model.joblib")
    print("Model saved to model.joblib")

if __name__ == "__main__":
    train_model()

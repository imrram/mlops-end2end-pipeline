import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def train_model():
    data = fetch_california_housing()
    X_train, _, y_train, _ = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "model.joblib")
    print("Model saved to model.joblib")

if __name__ == "__main__":
    train_model()

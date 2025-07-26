import joblib
from sklearn.datasets import fetch_california_housing

def main():
    model = joblib.load("model.joblib")
    data = fetch_california_housing()
    sample = data.data[0].reshape(1, -1)

    prediction = model.predict(sample)
    print("Sample prediction:", prediction)

if __name__ == "__main__":
    main()

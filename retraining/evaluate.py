import pandas as pd
import joblib
from sklearn.metrics import classification_report

def evaluate_model(data_path):
    model = joblib.load("retraining/model.pkl")
    df = pd.read_csv(data_path)
    df['label'] = (df['sensor_1'] > 11).astype(int)
    X = df[["sensor_1", "sensor_2"]]
    y = df["label"]
    preds = model.predict(X)
    print(classification_report(y, preds))

if __name__ == "__main__":
    evaluate_model("data/sensor_drifted.csv")

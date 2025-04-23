import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def retrain_model(data_path):
    df = pd.read_csv(data_path)
    df['label'] = (df['sensor_1'] > 11).astype(int)
    X = df[["sensor_1", "sensor_2"]]
    y = df["label"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, "retraining/model.pkl")

if __name__ == "__main__":
    retrain_model("data/sensor_drifted.csv")

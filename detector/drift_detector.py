import pandas as pd
import numpy as np
from alibi_detect.cd import KSDrift
import joblib

def load_data(baseline_path, test_path):
    baseline = pd.read_csv(baseline_path).drop(columns=["timestamp"])
    test = pd.read_csv(test_path).drop(columns=["timestamp"])
    return baseline.values, test.values

def detect_drift(baseline, test):
    detector = KSDrift(p_val=0.05)
    detector.fit(baseline)
    preds = detector.predict(test)
    return preds, detector

if __name__ == "__main__":
    baseline, test = load_data("data/sensor_baseline.csv", "data/sensor_drifted.csv")
    result, detector = detect_drift(baseline, test)
    print("Drift Detected:", result['data']['is_drift'])
    print("Drift Scores:", result['data']['p_val'])
    joblib.dump(detector, "detector/ks_drift_detector.pkl")

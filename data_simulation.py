import pandas as pd
import numpy as np

def generate_sensor_data(filename, drift=False):
    timestamps = pd.date_range("2025-01-01", periods=100, freq="H")
    sensor_1 = np.random.normal(10, 0.5, size=100)
    sensor_2 = np.random.normal(100, 2, size=100)

    if drift:
        sensor_1[50:] += 2     
        sensor_2[50:] += 10

    df = pd.DataFrame({
        "timestamp": timestamps,
        "sensor_1": sensor_1,
        "sensor_2": sensor_2
    })

    df.to_csv(filename, index=False)

generate_sensor_data("sensor_baseline.csv", drift=False)
generate_sensor_data("sensor_drifted.csv", drift=True)

print("Generated sensor data saved to sensor_baseline.csv and sensor_drifted.csv")
# This script generates two CSV files with simulated sensor data.
# The first file contains baseline data, while the second file includes drifted data.
# The drift is introduced by adding a constant value to the sensor readings after a certain point in time.
# The generated data can be used for testing and validation of the drift detection algorithm.


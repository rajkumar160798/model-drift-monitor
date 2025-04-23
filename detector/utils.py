import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_distributions(baseline_path, test_path):
    base = pd.read_csv(baseline_path)
    drift = pd.read_csv(test_path)
    for col in ["sensor_1", "sensor_2"]:
        plt.figure(figsize=(8,4))
        sns.kdeplot(base[col], label="Baseline", fill=True)
        sns.kdeplot(drift[col], label="Drifted", fill=True)
        plt.title(f"Distribution of {col} Before vs After Drift")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"notebooks/{col}_distribution.png")
        plt.close()
    print("Distribution plots saved to notebooks/")
plot_feature_distributions("data/sensor_baseline.csv", "data/sensor_drifted.csv")
# This script generates distribution plots for the features in the baseline and drifted datasets.
# It uses seaborn to create kernel density estimates (KDE) for each feature.
# The plots are saved in the "notebooks" directory.
# The generated plots can be used to visually inspect the differences in feature distributions before and after drift.
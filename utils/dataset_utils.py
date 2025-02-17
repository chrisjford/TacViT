import pandas as pd
import os

def split_dataset(csv_path, test_sensor_id=5):
    df = pd.read_csv(csv_path)
    train_df = df[df["sensor_id"] != test_sensor_id]
    test_df = df[df["sensor_id"] == test_sensor_id]

    train_path = "tactile_dataset/train_labels.csv"
    test_path = "tactile_dataset/test_labels.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, test_path

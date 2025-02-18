import pandas as pd
import numpy as np
import config

def split_dataset(csv_file, test_sensor_id, train_ratio=0.8, tradaboost_ratio=0.1, seed=42):
    """
    Dynamically assigns train/val/test splits, ensuring that 10% of the test sensor data is included in training for TrAdaBoost.

    Args:
        csv_file (str): Path to the dataset CSV file.
        test_sensor_id (int): The sensor ID to reserve as test data.
        train_ratio (float): Ratio of train data (default 80% train, 20% val).
        tradaboost_ratio (float): Ratio of test sensor data to be included in training (default 10%).
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Updated DataFrame with assigned "split" column.
    """
    # Load dataset
    df = pd.read_csv(csv_file)

    # Assign test set (all data from test_sensor_id)
    df["split"] = "train"  # Default to train
    df.loc[df["sensor_id"] == test_sensor_id, "split"] = "test"

    # Get only trainable sensors (excluding the test sensor)
    trainable_df = df[df["sensor_id"] != test_sensor_id]

    # Shuffle data for randomness
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(trainable_df))
    split_idx = int(len(trainable_df) * train_ratio)

    # Assign train/val split
    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    df.loc[trainable_df.iloc[val_indices].index, "split"] = "val"  # Assign validation

    # Select 10% of test domain data for TrAdaBoost
    test_data = df[df["sensor_id"] == test_sensor_id]  # Get test domain data
    tradaboost_size = int(len(test_data) * tradaboost_ratio)

    if tradaboost_size > 0:
        tradaboost_samples = test_data.sample(n=tradaboost_size, random_state=seed)
        df.loc[tradaboost_samples.index, "split"] = "train"  # Assign to training

    return df

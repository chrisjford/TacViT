import torch
import pandas as pd
import cv2
from torch.utils.data import Dataset
import config
from utils.split_dataset import split_dataset  # Import the split function

class TacTipDataset(Dataset):
    """
    Custom dataset class for TacTip tactile images and 6DoF pose labels.
    Handles different sensor domains for domain adaptation.
    """
    def __init__(self, csv_file, split="train", transform=None, test_sensor_id=None):
        """
        Args:
            csv_file (str): Path to the dataset CSV file.
            split (str): Which dataset split to load ("train", "val", "test").
            transform (callable, optional): Optional transforms.
            test_sensor_id (int, optional): Sensor ID to reserve as test data.
        """
        # Apply split function to assign splits dynamically
        if test_sensor_id is not None:
            df = split_dataset(csv_file, test_sensor_id)
        else:
            df = pd.read_csv(csv_file)

        self.data = df[df["split"] == split]
        self.transform = transform

        # Compute normalization values for labels
        label_columns = ["x", "Rx", "Ry", "Fx", "Fy", "Fz"]
        self.label_means = torch.tensor(self.data[label_columns].mean().values, dtype=torch.float32)
        self.label_stds = torch.tensor(self.data[label_columns].std().values, dtype=torch.float32)
        self.label_stds[self.label_stds == 0] = 1.0  # Avoid division by zero

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["image_path"]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  
        image = cv2.resize(image, (224, 224))
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  

        label = self.data.iloc[idx][["x", "Rx", "Ry", "Fx", "Fy", "Fz"]].values.astype(float)
        label = torch.tensor(label, dtype=torch.float32)
        label = (label - self.label_means) / self.label_stds

        sensor_id = int(self.data.iloc[idx]["sensor_id"])
        return image, label, sensor_id
    
    def get_label_stats(self):
        """
        Returns the label means & stds for denormalization.
        """
        return self.label_means, self.label_stds

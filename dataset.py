import torch
import pandas as pd
import cv2
from torch.utils.data import Dataset
import config  # Import global config settings

class TacTipDataset(Dataset):
    """
    Custom dataset class for TacTip tactile images and 6DoF pose labels.
    Handles different sensor domains for domain adaptation.
    """
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to the dataset CSV file.
            transform (callable, optional): Optional transforms to apply to images.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Preprocessed TacTip grayscale image.
            label (Tensor): 6DoF pose (X, Y, Z, Rx, Ry, Rz).
            sensor_id (int): Sensor domain ID (for DA methods like DANN/MMD).
        """
        # Load Image
        img_path = self.data.iloc[idx]["image_path"]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load grayscale
        image = cv2.resize(image, (224, 224))  # Resize for ViT
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # Apply optional transformations
        if self.transform:
            image = self.transform(image)

        # Load 6DoF Pose Labels
        label = self.data.iloc[idx][["X", "Y", "Z", "Rx", "Ry", "Rz"]].values.astype(float)
        label = torch.tensor(label, dtype=torch.float32)

        # Load Sensor ID
        sensor_id = int(self.data.iloc[idx]["sensor_id"])

        return image, label, sensor_id

import torch
import torch.nn as nn
import torch.optim as optim
from models.vit6dof import ViT6DoF
from dataset import TacTipDataset
from torch.utils.data import DataLoader
import config

def train_model():
    """
    Trains the ViT model for pose estimation, with optional DANN-based domain adaptation.
    """
    # Load dataset
    train_dataset = TacTipDataset(config.DATASET_PATH)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Initialize Model
    model = ViT6DoF(num_domains=6).to(config.DEVICE)
    
    # Define Loss Functions
    pose_loss_fn = nn.MSELoss()
    domain_loss_fn = nn.NLLLoss() if config.USE_DANN else None  # Negative Log Likelihood for DANN
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    model.train()
    for epoch in range(config.EPOCHS):
        total_loss = 0
        for images, labels, sensor_ids in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE), sensor_ids.to(config.DEVICE)

            optimizer.zero_grad()

            # Forward pass
            pose_pred, domain_pred = model(images, alpha=1.0)

            # Compute pose loss
            pose_loss = pose_loss_fn(pose_pred, labels)

            # Compute domain classification loss (if DANN is enabled)
            domain_loss = domain_loss_fn(domain_pred, sensor_ids) if config.USE_DANN else 0
            
            # Total loss (weighted sum)
            loss = pose_loss + (config.DANN_LAMBDA * domain_loss if config.USE_DANN else 0)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{config.EPOCHS}, Loss: {total_loss:.4f}")
    
    torch.save(model.state_dict(), "experiments/results/trained_model.pth")

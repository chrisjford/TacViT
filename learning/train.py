import torch
import torch.nn as nn
import torch.optim as optim
from models.vit6dof import ViT6DoF
from dataset import TacTipDataset
from torch.utils.data import DataLoader
import config
import os
from learning.validate import validate_model  # Import modular validation

# Import domain adaptation methods if enabled
if config.USE_MMD:
    from domain_adaptation.mmd import compute_mmd_loss
if config.USE_TRADABOOST:
    from domain_adaptation.tradaboost import TrAdaBoost
if config.USE_DANN:
    from domain_adaptation.dann import DomainAdversarialLoss

def train_model():
    """
    Trains the ViT model for pose estimation, with optional domain adaptation methods (DANN, MMD, TrAdaBoost).
    """
    # Load datasets
    train_dataset = TacTipDataset(config.DATASET_PATH, split="train", test_sensor_id=config.TEST_SENSOR_ID)
    val_dataset = TacTipDataset(config.DATASET_PATH, split="val", test_sensor_id=config.TEST_SENSOR_ID)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    # Initialize Model
    model = ViT6DoF(num_domains=config.NUM_DOMAINS).to(config.DEVICE)

    # Initialize domain adaptation techniques
    if config.USE_TRADABOOST:
        tradaboost = TrAdaBoost(target_sensor=config.TEST_SENSOR_ID, n_iters=config.TRADABOOST_ITERS)
    if config.USE_DANN:
        dann_loss_fn = DomainAdversarialLoss(num_domains=config.NUM_DOMAINS).to(config.DEVICE)

    # Define Loss Functions
    pose_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.5, patience=3,
                                                       verbose=True)

    best_val_loss = float("inf")

    model.train()
    for epoch in range(config.EPOCHS):
        total_loss = 0
        total_pose_loss = 0
        total_dann_loss = 0
        total_mmd_loss = 0
        total_tradaboost_loss = 0

        for batch_idx, (images, labels, sensor_ids) in enumerate(train_loader):
            images, labels, sensor_ids = images.to(config.DEVICE), labels.to(config.DEVICE), sensor_ids.to(config.DEVICE)

            optimizer.zero_grad()

            # Forward pass
            if config.USE_MMD or config.USE_DANN:
                pose_pred, domain_pred, features = model(images, alpha=1.0, return_features=True)
            else:
                pose_pred, domain_pred = model(images, alpha=1.0)
                features = None  # Not needed if MMD & DANN are off

            # Compute pose loss
            pose_loss = pose_loss_fn(pose_pred, labels)

            # Compute domain classification loss (DANN)
            dann_loss = torch.tensor(0.0, device=config.DEVICE)
            if config.USE_DANN and features is not None:
                dann_loss, domain_pred = dann_loss_fn(features, sensor_ids, alpha=1.0)

            # Compute MMD loss
            mmd_loss = torch.tensor(0.0, device=config.DEVICE)
            if config.USE_MMD and features is not None:
                mmd_loss = compute_mmd_loss(features, sensor_ids)

            # Apply TrAdaBoost sample weighting
            tradaboost_loss = torch.tensor(0.0, device=config.DEVICE)
            if config.USE_TRADABOOST:
                tradaboost_loss = tradaboost.update_weights(images, labels, pose_pred)

            # Compute total loss
            loss = (
                pose_loss
                + (config.DANN_LAMBDA * dann_loss if config.USE_DANN else 0)
                + (config.MMD_LAMBDA * mmd_loss if config.USE_MMD else 0)
                + (config.TRADABOOST_LAMBDA * tradaboost_loss if config.USE_TRADABOOST else 0)
            )

            loss.backward()
            optimizer.step()

            # Track losses
            total_loss += loss.item()
            total_pose_loss += pose_loss.item()
            total_dann_loss += dann_loss.item() if config.USE_DANN else 0
            total_mmd_loss += mmd_loss.item() if config.USE_MMD else 0
            total_tradaboost_loss += tradaboost_loss.item() if config.USE_TRADABOOST else 0

            # Print loss every 100 batches
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        print(
            f"Epoch {epoch+1}/{config.EPOCHS}, Total Train Loss: {total_loss:.4f}, "
            f"Pose Loss: {total_pose_loss:.4f}, DANN Loss: {total_dann_loss:.4f}, "
            f"MMD Loss: {total_mmd_loss:.4f}, TrAdaBoost Loss: {total_tradaboost_loss:.4f}"
        )

        # Run validation step after every epoch
        val_loss = validate_model(model, val_loader, pose_loss_fn)
        print(f"Validation Loss after Epoch {epoch+1}: {val_loss:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0  # Reset counter
            os.makedirs("experiments/results", exist_ok=True)
            torch.save(model.state_dict(), "experiments/results/best_model.pth")
            print("✅ Saved new best model.")
        else:
            early_stop_counter += 1  # Increment counter
        
        # Adapt learning rate
        scheduler.step(val_loss)

        # Stop training if validation loss keeps increasing
        if early_stop_counter >= config.EARLY_STOPPING_PATIENCE:
            print("⏹ Early stopping triggered. Training stopped.")
            break  # Stop training

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("experiments/results", exist_ok=True)
            torch.save(model.state_dict(), "experiments/results/best_model.pth")
            print("✅ Saved new best model.")

    # Save final model
    torch.save(model.state_dict(), "experiments/results/trained_model_final.pth")
    print("✅ Training complete. Final model saved.")

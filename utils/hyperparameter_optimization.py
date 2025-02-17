import torch
import torch.nn as nn
import torch.optim as optim
from models.vit6dof import ViT6DoF
from dataset import TacTipDataset
from torch.utils.data import DataLoader
import config
from skopt import gp_minimize
from skopt.space import Integer, Real
import logging

def train_dann_with_params(dann_hidden_dim, dann_layers, dann_lambda):
    """
    Trains and evaluates a DANN model using the given hyperparameters.

    Args:
        dann_hidden_dim (int): Hidden layer size for the domain classifier.
        dann_layers (int): Number of layers in domain classifier.
        dann_lambda (float): Weight for domain classification loss.

    Returns:
        float: Final test loss (lower is better).
    """
    # Update config with new hyperparameters
    config.DANN_HIDDEN_DIM = int(dann_hidden_dim)
    config.DANN_LAYERS = int(dann_layers)
    config.DANN_LAMBDA = float(dann_lambda)

    # Load dataset
    train_dataset = TacTipDataset(config.DATASET_PATH)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Initialize Model with new hyperparameters
    model = ViT6DoF(num_domains=6, use_lora=True, dann_hidden_dim=config.DANN_HIDDEN_DIM, dann_layers=config.DANN_LAYERS).to(config.DEVICE)

    # Loss Functions
    pose_loss_fn = nn.MSELoss()
    domain_loss_fn = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Training Loop
    model.train()
    for epoch in range(config.EPOCHS // 2):  # Shorter training for optimization
        total_loss = 0
        for images, labels, sensor_ids in train_loader:
            images, labels, sensor_ids = images.to(config.DEVICE), labels.to(config.DEVICE), sensor_ids.to(config.DEVICE)

            optimizer.zero_grad()
            pose_pred, domain_pred = model(images, alpha=1.0)
            pose_loss = pose_loss_fn(pose_pred, labels)
            domain_loss = domain_loss_fn(domain_pred, sensor_ids)
            loss = pose_loss + config.DANN_LAMBDA * domain_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Evaluate the Model
    from learning.test import evaluate_model
    test_loss = evaluate_model(model)
    
    return test_loss  # Minimization objective

def optimize_dann():
    """
    Runs Bayesian Optimization to find the best DANN hyperparameters.
    """
    logging.info("Starting DANN Hyperparameter Optimization")

    # Define Search Space
    search_space = [
        Integer(64, 512, name="dann_hidden_dim"),  # Hidden layer size
        Integer(1, 4, name="dann_layers"),  # Number of layers
        Real(0.01, 1.0, name="dann_lambda")  # Weight of domain loss
    ]

    # Run Bayesian Optimization
    result = gp_minimize(
        func=lambda params: train_dann_with_params(*params),
        dimensions=search_space,
        n_calls=15,  # Number of evaluations
        random_state=42
    )

    # Best Parameters
    best_params = {
        "dann_hidden_dim": result.x[0],
        "dann_layers": result.x[1],
        "dann_lambda": result.x[2],
        "best_loss": result.fun
    }

    # Save Results
    log_path = "experiments/logs/dann_optimization_results.json"
    import json
    with open(log_path, "w") as file:
        json.dump(best_params, file, indent=4)

    logging.info(f"Best DANN Hyperparameters: {best_params}")

    return best_params

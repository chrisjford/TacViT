import torch
from dataset import TacTipDataset
from models.vit6dof import ViT6DoF
from utils.visualizations import plot_tsne_features
import config

def evaluate_model():
    model = ViT6DoF().to(config.DEVICE)
    model.load_state_dict(torch.load("experiments/results/trained_model.pth"))
    model.eval()

    test_dataset = TacTipDataset(config.DATASET_PATH)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    loss_fn = torch.nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            total_loss += loss.item()

    print(f"Test Loss: {total_loss:.4f}")

    # Visualize Feature Alignment
    plot_tsne_features(model, test_loader, "Feature Space Visualization - Test Data")

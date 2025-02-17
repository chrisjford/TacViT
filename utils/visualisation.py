import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np

def plot_loss_curve(training_losses, validation_losses):
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("experiments/results/loss_curve.png")
    plt.show()

def plot_tsne_features(model, data_loader, experiment_name):
    """
    Generates and saves a t-SNE plot of the feature space.

    Args:
        model (torch.nn.Module): Trained model.
        data_loader (torch.utils.data.DataLoader): Data loader for test set.
        experiment_name (str): Name of the adaptation method (e.g., "DANN", "MMD", "No DA").
    """
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for images, _, sensor_ids in data_loader:
            images = images.to("cuda" if torch.cuda.is_available() else "cpu")
            extracted_features = model.vit(images).last_hidden_state[:, 0, :].cpu().numpy()

            features.append(extracted_features)
            labels.append(sensor_ids.cpu().numpy())

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    # t-SNE Reduction
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.colorbar()
    plt.title(f"t-SNE Feature Space Visualization - {experiment_name}")
    
    # Save plot
    save_path = f"experiments/results/tsne_{experiment_name.replace(' ', '_')}.png"
    plt.savefig(save_path)
    plt.show()

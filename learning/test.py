import torch
from dataset import TacTipDataset
from models.vit6dof import ViT6DoF
from torch.utils.data import DataLoader
from utils.visualisation import plot_tsne_features
import config

def evaluate_model():
    # 1. Load trained model
    model = ViT6DoF().to(config.DEVICE)
    model.load_state_dict(torch.load("experiments/results/trained_model_final.pth"))
    model.eval()

    # 2. Load test dataset
    test_dataset = TacTipDataset(
        csv_file=config.DATASET_PATH,
        split="test",
        test_sensor_id=config.TEST_SENSOR_ID
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    # 3. Retrieve label stats for denormalization
    label_means, label_stds = test_dataset.get_label_stats()
    label_means = label_means.to(config.DEVICE)
    label_stds = label_stds.to(config.DEVICE)

    # 4. MSE Loss in the real (denormalized) scale
    loss_fn = torch.nn.MSELoss()
    total_loss = 0
    total_samples = 0

    # Simple denormalize function
    def denormalize(predictions_norm):
        # predictions_norm shape: (batch_size, 6)
        return predictions_norm * label_stds + label_means

    with torch.no_grad():
        for images, labels_norm, _ in test_loader:
            images = images.to(config.DEVICE)
            labels_norm = labels_norm.to(config.DEVICE)

            # 5. Forward pass
            output = model(images)

            # Model could return tuple
            if isinstance(output, tuple):
                pose_pred_norm = output[0]  # (pose_pred, domain_pred, features)
            else:
                pose_pred_norm = output  # single tensor

            # 6. Denormalize predictions
            pose_pred_real = denormalize(pose_pred_norm)

            # 7. If your dataset loaded normalized labels, we must also denormalize labels
            #    or if your dataset returned them already in real scale, skip this
            labels_real = denormalize(labels_norm)  # if dataset yields normalized labels

            # 8. Compute MSE on real scale
            loss = loss_fn(pose_pred_real, labels_real)
            total_loss += loss.item() * labels_norm.shape[0]
            total_samples += labels_norm.shape[0]

    # Final average loss on real scale
    avg_loss = total_loss / total_samples
    print(f"Test Loss (Denormalized): {avg_loss:.4f}")

    # 9. Visualize Feature Alignment (if needed)
    plot_tsne_features(model, test_loader, "Feature Space Visualization - Test Data")

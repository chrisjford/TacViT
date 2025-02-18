import torch
import config

# Import domain adaptation methods only when needed
if config.USE_DANN:
    from domain_adaptation.dann import DomainAdversarialLoss
if config.USE_MMD:
    from domain_adaptation.mmd import compute_mmd_loss
if config.USE_TRADABOOST:
    from domain_adaptation.tradaboost import TrAdaBoost

def validate_model(model, val_loader, pose_loss_fn):
    """
    Runs validation step after each epoch.

    Args:
        model (nn.Module): The trained ViT6DoF model.
        val_loader (DataLoader): Validation data loader.
        pose_loss_fn (nn.Module): Loss function for pose estimation.

    Returns:
        float: Average validation loss over all batches.
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    total_batches = 0

    with torch.no_grad():
        for images, labels, sensor_ids in val_loader:
            images, labels, sensor_ids = images.to(config.DEVICE), labels.to(config.DEVICE), sensor_ids.to(config.DEVICE)

            if config.USE_MMD or config.USE_DANN:
                pose_pred, domain_pred, features = model(images, alpha=1.0, return_features=True)
            else:
                pose_pred, domain_pred = model(images, alpha=1.0)
                features = None

            pose_loss = pose_loss_fn(pose_pred, labels)

            dann_loss = torch.tensor(0.0, device=config.DEVICE)
            if config.USE_DANN and features is not None:
                dann_loss_fn = DomainAdversarialLoss(num_domains=config.NUM_DOMAINS).to(config.DEVICE)
                dann_loss, domain_pred = dann_loss_fn(features, sensor_ids, alpha=1.0)

            mmd_loss = torch.tensor(0.0, device=config.DEVICE)
            if config.USE_MMD and features is not None:
                mmd_loss = compute_mmd_loss(features, sensor_ids)

            tradaboost_loss = torch.tensor(0.0, device=config.DEVICE)
            if config.USE_TRADABOOST:
                tradaboost = TrAdaBoost(target_sensor=config.TEST_SENSOR_ID, n_iters=config.TRADABOOST_ITERS)
                tradaboost_loss = tradaboost.update_weights(images, labels, pose_pred)

            loss = (
                pose_loss
                + (config.DANN_LAMBDA * dann_loss if config.USE_DANN else 0)
                + (config.MMD_LAMBDA * mmd_loss if config.USE_MMD else 0)
                + (config.TRADABOOST_LAMBDA * tradaboost_loss if config.USE_TRADABOOST else 0)
            )

            val_loss += loss.item()
            total_batches += 1

    model.train()  # Set model back to training mode
    return val_loss / total_batches

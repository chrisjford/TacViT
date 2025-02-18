import torch
import torch.nn.functional as F

def compute_mmd_loss(features, domain_labels):
    """
    Computes the Maximum Mean Discrepancy (MMD) loss between different sensor domains.
    
    Args:
        features (Tensor): Extracted feature representations from ViT.
        domain_labels (Tensor): Domain labels (sensor IDs).
    
    Returns:
        Tensor: MMD loss value.
    """
    # Split source and target features
    source_features = features[domain_labels != domain_labels.max()]  # Training domains
    target_features = features[domain_labels == domain_labels.max()]  # Unseen test domain

    if len(source_features) == 0 or len(target_features) == 0:
        return torch.tensor(0.0, device=features.device)  # Avoid NaNs when domains are empty

    # Compute squared Euclidean distance
    mean_source = torch.mean(source_features, dim=0)
    mean_target = torch.mean(target_features, dim=0)
    mmd_loss = F.mse_loss(mean_source, mean_target)

    return mmd_loss

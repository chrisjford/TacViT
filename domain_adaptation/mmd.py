import torch
import torch.nn.functional as F

def mmd_loss(source_features, target_features):
    """
    Computes Maximum Mean Discrepancy (MMD) loss to align feature distributions
    between source and target domain samples.

    Args:
        source_features (Tensor): Feature representations from source domain (e.g., one TacTip sensor).
        target_features (Tensor): Feature representations from target domain (e.g., another TacTip sensor).

    Returns:
        Tensor: MMD loss value.
    """
    mean_source = torch.mean(source_features, dim=0)
    mean_target = torch.mean(target_features, dim=0)
    return F.mse_loss(mean_source, mean_target)  # Minimize feature distance

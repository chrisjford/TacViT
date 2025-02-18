import torch
import config

class TrAdaBoost:
    """
    Implements Transfer AdaBoost (TrAdaBoost) to focus learning on hard-to-adapt target samples.
    """
    def __init__(self, target_sensor, n_iters=config.TRADABOOST_ITERS):
        """
        Args:
            target_sensor (int): The sensor ID representing the target domain.
            n_iters (int): Number of boosting iterations.
        """
        self.n_iters = n_iters
        self.target_sensor = target_sensor

    def update_weights(self, images, labels, predictions):
        """
        Updates sample weights based on prediction errors.

        Args:
            images (Tensor): Input images.
            labels (Tensor): True labels.
            predictions (Tensor): Model predictions.
        
        Returns:
            Tensor: Weighted loss value.
        """
        # Ensure everything is on the same device
        labels = labels.to(config.DEVICE)
        predictions = predictions.to(config.DEVICE)

        # Compute per-sample error
        errors = torch.abs(predictions - labels).mean(dim=1)  

        # Create a per-batch weight vector initialized to 1
        batch_size = labels.shape[0]
        updated_weights = torch.ones(batch_size, device=config.DEVICE)  

        # Create mask for target domain samples and move it to the correct device
        mask = (labels[:, -1] == self.target_sensor).to(config.DEVICE)

        # Boost weights for hard-to-adapt samples (only for the current batch)
        updated_weights[mask] *= torch.exp(errors[mask])

        # Normalize weights within the batch
        updated_weights /= updated_weights.sum() + 1e-8  # Prevent division by zero

        # Compute weighted loss
        weighted_loss = (errors * updated_weights).mean()

        return weighted_loss

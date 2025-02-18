import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainAdversarialLoss(nn.Module):
    """
    Implements the domain adversarial loss for DANN.
    """
    def __init__(self, num_domains):
        super(DomainAdversarialLoss, self).__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(768, 512),  # Assuming 768-dim features from ViT
            nn.ReLU(),
            nn.Linear(512, num_domains),
            nn.LogSoftmax(dim=1),
        )
        self.loss_fn = nn.NLLLoss()  # Negative Log-Likelihood Loss

    def forward(self, features, domain_labels, alpha):
        """
        Applies gradient reversal and computes domain classification loss.

        Args:
            features (Tensor): Extracted features from the model.
            domain_labels (Tensor): Ground-truth domain labels.
            alpha (float): Scaling factor for the gradient reversal.

        Returns:
            Tensor: Domain classification loss.
        """
        # Apply domain classifier
        domain_pred = self.domain_classifier(features)

        # Compute domain classification loss
        loss = self.loss_fn(domain_pred, domain_labels)

        return loss, domain_pred

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
import config  # Import config to toggle DANN

class GradientReversalLayer(torch.autograd.Function):
    """
    Implements Gradient Reversal Layer (GRL) for domain adversarial training.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class ViT6DoF(nn.Module):
    def __init__(self, num_domains=config.NUM_DOMAINS, use_lora=True):
        """
        Args:
            num_domains (int): Number of different sensor domains (for DANN).
            use_lora (bool): Whether to apply LoRA fine-tuning.
        """
        super(ViT6DoF, self).__init__()

        # Load Pretrained ViT
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # Pose regression head
        self.pose_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 6)  # Output 6 values: (z, Rx, Ry, Fx, Fy, Fz)
        )

        # Domain Classifier (Only if DANN is enabled)
        self.use_dann = config.USE_DANN
        if self.use_dann:
            self.domain_classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, num_domains),
                nn.LogSoftmax(dim=1)
            )

        self.use_lora = use_lora  # LoRA integration flag

    def forward(self, x, alpha=1.0, return_features=False):
        """
        Forward pass that returns pose predictions and domain classification if DANN is enabled.
        """
        features = self.vit(x).last_hidden_state[:, 0, :]  # Extract CLS token features

        # Pose prediction
        pose = self.pose_head(features)

        # Domain classification (Only if DANN is enabled)
        domain_pred = None
        if self.use_dann:
            reversed_features = GradientReversalLayer.apply(features, alpha)
            domain_pred = self.domain_classifier(reversed_features)
        
        if return_features:
            return pose, domain_pred, features  # Return features for MMD/DANN
        else:
            return pose, domain_pred

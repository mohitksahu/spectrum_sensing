"""Focal Loss implementation for handling class imbalance in PU detection.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss for binary/multi-class classification with class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Where:
        p_t = probability of correct class
        alpha_t = class balancing weight
        gamma = focusing parameter (reduces loss for well-classified examples)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """Initialize Focal Loss.
        
        Args:
            gamma: Focusing parameter. Higher values down-weight easy examples more.
                   gamma=0 reduces to standard Cross-Entropy.
            alpha: Class balancing weights of shape (num_classes,).
                   If None, no class balancing is applied.
            reduction: 'none', 'mean', or 'sum'.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            logits: Predicted logits of shape (B, num_classes).
            targets: Ground truth class indices of shape (B,).
            
        Returns:
            Focal loss value (scalar if reduction='mean' or 'sum').
        """
        num_classes = logits.shape[1]
        
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get probability of target class
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        p_t = (probs * targets_one_hot).sum(dim=1)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute cross-entropy component
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha class weighting
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

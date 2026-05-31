"""Kendall Uncertainty Weighting for multi-task loss balancing.

Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses
for Scene Geometry and Semantics", CVPR 2018.

Loss = (1/(2*sigma1^2))*L1 + (1/(2*sigma2^2))*L2 + (1/(2*sigma3^2))*L3
       + log(sigma1) + log(sigma2) + log(sigma3)
"""

import torch
import torch.nn as nn
from typing import Dict


class KendallUncertaintyLoss(nn.Module):
    """Multi-task loss with learnable homoscedastic uncertainty weighting.
    
    Automatically balances three task losses (PU detection, modulation
    classification, SNR estimation) using learnable log-variance parameters.
    """
    
    def __init__(self, num_tasks: int = 3):
        """Initialize uncertainty loss with learnable parameters.
        
        Args:
            num_tasks: Number of tasks to balance (3: PU, Mod, SNR).
        """
        super().__init__()
        # Learnable log(sigma^2) parameters - initialized to 0
        # This means initial sigma=1 and equal weighting
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute weighted multi-task loss.
        
        Args:
            losses: Dictionary with individual task losses:
                'pu': PU detection focal loss
                'mod': Modulation cross-entropy loss
                'snr': SNR Huber loss
                
        Returns:
            Dictionary with:
                'total': Combined weighted loss
                'pu_weighted': Weighted PU loss component
                'mod_weighted': Weighted modulation loss component
                'snr_weighted': Weighted SNR loss component
                'weights': Current task weights (1/(2*sigma^2))
        """
        loss_pu = losses["pu"]
        loss_mod = losses["mod"]
        loss_snr = losses["snr"]
        
        # Compute precision (1/sigma^2) = exp(-log_var)
        precision_pu = torch.exp(-self.log_vars[0])
        precision_mod = torch.exp(-self.log_vars[1])
        precision_snr = torch.exp(-self.log_vars[2])
        
        # Weighted losses: (1/(2*sigma^2)) * L + log(sigma)
        # = 0.5 * exp(-log_var) * L + 0.5 * log_var
        weighted_pu = 0.5 * precision_pu * loss_pu + 0.5 * self.log_vars[0]
        weighted_mod = 0.5 * precision_mod * loss_mod + 0.5 * self.log_vars[1]
        weighted_snr = 0.5 * precision_snr * loss_snr + 0.5 * self.log_vars[2]
        
        total_loss = weighted_pu + weighted_mod + weighted_snr
        
        return {
            "total": total_loss,
            "pu_weighted": weighted_pu,
            "mod_weighted": weighted_mod,
            "snr_weighted": weighted_snr,
            "weights": torch.stack([
                0.5 * precision_pu,
                0.5 * precision_mod,
                0.5 * precision_snr,
            ]),
            "log_vars": self.log_vars.detach(),
        }

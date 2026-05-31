"""Stage 4: Multi-Task Prediction Heads.

Implements three task-specific heads for PU detection, modulation classification,
and SNR estimation. All heads operate on the [CLS] token representation.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class PUDetectionHead(nn.Module):
    """Binary PU detection head.
    
    Predicts Primary User presence: 0 (idle) or 1 (active).
    Uses Focal Loss for training to handle class imbalance.
    """
    
    def __init__(self, in_features: int = 64, num_classes: int = 2):
        """Initialize PU detection head.
        
        Args:
            in_features: Input feature dimension from CLS token.
            num_classes: Number of output classes (2 for binary).
        """
        super().__init__()
        self.head = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: CLS token of shape (B, 64).
            
        Returns:
            Logits of shape (B, 2).
        """
        return self.head(x)


class ModulationHead(nn.Module):
    """5-class modulation classification head.
    
    Classifies modulation scheme: BPSK(0), QPSK(1), 8PSK(2), 16QAM(3), DQPSK(4).
    Uses CrossEntropy with class weights for training.
    """
    
    def __init__(self, in_features: int = 64, num_classes: int = 5):
        """Initialize modulation classification head.
        
        Args:
            in_features: Input feature dimension from CLS token.
            num_classes: Number of modulation classes.
        """
        super().__init__()
        self.head = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: CLS token of shape (B, 64).
            
        Returns:
            Logits of shape (B, 5).
        """
        return self.head(x)


class SNRHead(nn.Module):
    """SNR estimation regression head.
    
    Predicts SNR value in dB. Uses Huber Loss for training
    to be robust to outliers.
    """
    
    def __init__(self, in_features: int = 64):
        """Initialize SNR estimation head.
        
        Args:
            in_features: Input feature dimension from CLS token.
        """
        super().__init__()
        self.head = nn.Linear(in_features, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: CLS token of shape (B, 64).
            
        Returns:
            SNR prediction of shape (B, 1).
        """
        return self.head(x)


class MultiTaskHead(nn.Module):
    """Combined multi-task prediction module.
    
    Wraps all three task heads and returns predictions as a dictionary.
    """
    
    def __init__(self, in_features: int = 64):
        """Initialize multi-task heads.
        
        Args:
            in_features: Input feature dimension from CLS token.
        """
        super().__init__()
        self.pu_head = PUDetectionHead(in_features, num_classes=2)
        self.mod_head = ModulationHead(in_features, num_classes=5)
        self.snr_head = SNRHead(in_features)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through all task heads.
        
        Args:
            x: CLS token of shape (B, 64).
            
        Returns:
            Dictionary with keys 'pu', 'mod', 'snr' containing predictions.
        """
        return {
            "pu": self.pu_head(x),       # (B, 2)
            "mod": self.mod_head(x),     # (B, 5)
            "snr": self.snr_head(x),     # (B, 1)
        }
    
    def get_num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

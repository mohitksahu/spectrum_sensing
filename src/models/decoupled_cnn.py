"""Stage 1: Decoupled CNN Front-End for multi-scale spectral feature extraction.

This module implements the hierarchical decoupled CNN architecture adapted from
Zhang et al. (IEEE TCCN 2024). Instead of band-specific decoupling, we use
scale-specific decoupling with different kernel sizes to capture features at
different spectral granularities.
"""

import torch
import torch.nn as nn
from typing import Tuple


class SharedCNNBlock(nn.Module):
    """Shared convolutional layers for common feature extraction.
    
    Two-layer CNN block that extracts features common to all spectral scales
    before branching into parallel pathways.
    """
    
    def __init__(self):
        """Initialize shared CNN block.
        
        Architecture:
            Conv1D(1→32, k=7, p=3) → BN → GELU → Conv1D(32→32, k=5, p=2) → BN → GELU
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through shared layers.
        
        Args:
            x: Input tensor of shape (B, 1, 192).
            
        Returns:
            Feature tensor of shape (B, 32, 192).
        """
        return self.layers(x)


class DecoupledBranch(nn.Module):
    """Single decoupled branch with specific kernel size for scale-specific features.
    
    Each branch specializes in capturing features at a particular spectral scale:
    - Small kernel (k=3): Sharp peaks, narrow features
    - Medium kernel (k=7): Modulation bandwidth, main lobe
    - Large kernel (k=11): Spectral envelope, band transitions
    """
    
    def __init__(self, kernel_size: int, padding: int):
        """Initialize decoupled branch.
        
        Args:
            kernel_size: Convolutional kernel size.
            padding: Padding for maintaining sequence length.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through branch.
        
        Args:
            x: Input tensor of shape (B, 32, 192).
            
        Returns:
            Feature tensor of shape (B, 16, 192).
        """
        return self.conv(x)


class DecoupledCNNFrontEnd(nn.Module):
    """Complete Decoupled CNN Front-End (Stage 1).
    
    Implements shared feature extraction followed by parallel scale-specific
    branches and channel fusion. Output provides multi-scale local spectral
    features for downstream Transformer tokenization.
    
    Architecture:
        Shared Block → [Branch-A(k=3) || Branch-B(k=7) || Branch-C(k=11)]
        → Concatenate (48ch) → Conv1D(48→16, k=1) fusion
    
    Output shape: (B, 192, 16) — ready for patch tokenization.
    """
    
    def __init__(self):
        """Initialize the Decoupled CNN Front-End."""
        super().__init__()
        
        # Shared layers
        self.shared = SharedCNNBlock()
        
        # Parallel decoupled branches
        self.branch_a = DecoupledBranch(kernel_size=3, padding=1)   # Fine features
        self.branch_b = DecoupledBranch(kernel_size=7, padding=3)   # Medium features
        self.branch_c = DecoupledBranch(kernel_size=11, padding=5)  # Coarse features
        
        # Channel fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(48, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Decoupled CNN Front-End.
        
        Args:
            x: Input PSD tensor of shape (B, 1, 192).
            
        Returns:
            Multi-scale feature tensor of shape (B, 192, 16).
            Permuted from (B, 16, 192) for downstream patch tokenization.
        """
        # Shared feature extraction
        shared_features = self.shared(x)  # (B, 32, 192)
        
        # Parallel scale-specific branches
        feat_a = self.branch_a(shared_features)  # (B, 16, 192) - fine
        feat_b = self.branch_b(shared_features)  # (B, 16, 192) - medium
        feat_c = self.branch_c(shared_features)  # (B, 16, 192) - coarse
        
        # Concatenate along channel dimension
        concat = torch.cat([feat_a, feat_b, feat_c], dim=1)  # (B, 48, 192)
        
        # Channel fusion
        fused = self.fusion(concat)  # (B, 16, 192)
        
        # Permute for patch tokenization: (B, channels, seq) → (B, seq, channels)
        output = fused.permute(0, 2, 1)  # (B, 192, 16)
        
        return output
    
    def get_num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

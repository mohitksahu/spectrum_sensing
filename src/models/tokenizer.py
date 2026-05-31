"""Stage 2: Patch Tokenization and Positional Encoding.

Groups CNN output into non-overlapping patches, projects to model dimension,
prepends a learnable [CLS] token, and adds learnable positional encoding.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PatchTokenizer(nn.Module):
    """Converts CNN features into patch tokens for Transformer input.
    
    Takes the (B, 192, 16) CNN output, groups into 24 non-overlapping patches
    of size 8 (covering ~1.67 MHz each), flattens each patch (8*16=128 dim),
    and projects to d_model=64.
    
    Architecture:
        Input (B, 192, 16) → Reshape (B, 24, 128) → Linear(128, 64)
        → Prepend [CLS] → Add positional encoding → Output (B, 25, 64)
    """
    
    def __init__(
        self,
        seq_length: int = 192,
        patch_size: int = 8,
        in_channels: int = 16,
        d_model: int = 64,
    ):
        """Initialize patch tokenizer.
        
        Args:
            seq_length: Input sequence length (192 frequency bins).
            patch_size: Number of frequency bins per patch.
            in_channels: Number of CNN output channels.
            d_model: Transformer model dimension.
        """
        super().__init__()
        
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_patches = seq_length // patch_size  # 24
        self.patch_dim = patch_size * in_channels     # 128
        
        # Linear projection from patch_dim to d_model
        self.projection = nn.Linear(self.patch_dim, d_model)
        
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Learnable positional encoding (num_patches + 1 for CLS)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, d_model) * 0.02
        )
        
        # Layer norm after tokenization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: create patch tokens with positional encoding.
        
        Args:
            x: CNN output tensor of shape (B, 192, 16).
            
        Returns:
            Token sequence of shape (B, 25, 64) — 24 patches + 1 CLS token.
        """
        B = x.shape[0]
        
        # Reshape into patches: (B, 192, 16) → (B, 24, 8, 16) → (B, 24, 128)
        x = x.reshape(B, self.num_patches, self.patch_size, self.in_channels)
        x = x.reshape(B, self.num_patches, self.patch_dim)
        
        # Project to d_model: (B, 24, 128) → (B, 24, 64)
        tokens = self.projection(x)
        
        # Prepend [CLS] token: (B, 24, 64) → (B, 25, 64)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        # Add positional encoding
        tokens = tokens + self.pos_encoding
        
        # Layer normalization
        tokens = self.norm(tokens)
        
        return tokens
    
    def get_num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

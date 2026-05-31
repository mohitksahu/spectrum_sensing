"""Stage 3: Transformer Encoder for global spectral context modeling.

Implements a lightweight Transformer encoder with Pre-LayerNorm architecture,
multi-head self-attention, and GELU-activated FFN layers.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention mechanism.
    
    Implements scaled dot-product attention with multiple heads.
    Each head has d_k = d_model // num_heads = 16 dimensions.
    """
    
    def __init__(self, d_model: int = 64, num_heads: int = 4, dropout: float = 0.1):
        """Initialize MHSA.
        
        Args:
            d_model: Model dimension.
            num_heads: Number of attention heads.
            dropout: Attention dropout rate.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 16
        self.scale = math.sqrt(self.d_k)
        
        # QKV projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-head self-attention.
        
        Args:
            x: Input tensor of shape (B, seq_len, d_model).
            mask: Optional attention mask.
            
        Returns:
            Attention output of shape (B, seq_len, d_model).
        """
        B, N, _ = x.shape
        
        # Compute Q, K, V and reshape for multi-head
        Q = self.W_q(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.d_model)
        output = self.W_o(attn_output)
        
        return output


class FeedForwardNetwork(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation."""
    
    def __init__(self, d_model: int = 64, ffn_dim: int = 128, dropout: float = 0.1):
        """Initialize FFN.
        
        Args:
            d_model: Input and output dimension.
            ffn_dim: Hidden dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FFN.
        
        Args:
            x: Input tensor of shape (B, seq_len, d_model).
            
        Returns:
            Output tensor of shape (B, seq_len, d_model).
        """
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with Pre-LayerNorm architecture.
    
    Architecture: LN → MHSA → Residual → LN → FFN → Residual
    """
    
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ):
        """Initialize Transformer encoder layer.
        
        Args:
            d_model: Model dimension.
            num_heads: Number of attention heads.
            ffn_dim: FFN hidden dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, ffn_dim, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through encoder layer.
        
        Args:
            x: Input tensor of shape (B, seq_len, d_model).
            mask: Optional attention mask.
            
        Returns:
            Output tensor of shape (B, seq_len, d_model).
        """
        # Pre-norm MHSA with residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, mask)
        x = self.dropout1(x)
        x = residual + x
        
        # Pre-norm FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class TransformerEncoder(nn.Module):
    """Complete Transformer Encoder (Stage 3).
    
    Stack of L=2 Pre-LayerNorm Transformer layers with a final LayerNorm.
    Extracts the [CLS] token representation for downstream task heads.
    """
    
    def __init__(
        self,
        num_layers: int = 2,
        d_model: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ):
        """Initialize Transformer encoder.
        
        Args:
            num_layers: Number of encoder layers.
            d_model: Model dimension.
            num_heads: Number of attention heads.
            ffn_dim: FFN hidden dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through Transformer encoder.
        
        Args:
            x: Input token sequence of shape (B, 25, 64).
            mask: Optional attention mask.
            
        Returns:
            CLS token representation of shape (B, 64).
        """
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.final_norm(x)
        
        # Extract [CLS] token (first position)
        cls_output = x[:, 0, :]  # (B, 64)
        
        return cls_output
    
    def forward_full(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass returning all token representations.
        
        Used during MSM pre-training where all tokens are needed for reconstruction.
        
        Args:
            x: Input token sequence of shape (B, 25, 64).
            mask: Optional attention mask.
            
        Returns:
            Full token sequence of shape (B, 25, 64).
        """
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.final_norm(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

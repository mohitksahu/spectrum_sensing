"""SpectraSense: Complete Hybrid Decoupled-CNN + Transformer Model.

Integrates all four stages into a unified architecture for single-band
cognitive radio spectrum sensing. Supports both supervised multi-task
inference and self-supervised masked spectrum modeling pre-training.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .decoupled_cnn import DecoupledCNNFrontEnd
from .tokenizer import PatchTokenizer
from .transformer import TransformerEncoder
from .multitask_head import MultiTaskHead


class SpectraSense(nn.Module):
    """SpectraSense: Hybrid Decoupled-CNN + Transformer for Spectrum Sensing.
    
    Four-stage architecture:
        Stage 1: Decoupled CNN Front-End (multi-scale local features)
        Stage 2: Patch Tokenization + Positional Encoding
        Stage 3: Transformer Encoder (global spectral context)
        Stage 4: Multi-Task Prediction Heads
    
    Input: Single-band PSD vector of shape (B, 192)
    Output: Dictionary with 'pu' (B, 2), 'mod' (B, 5), 'snr' (B, 1)
    
    Total parameters: ~93K
    """
    
    def __init__(
        self,
        input_dim: int = 192,
        patch_size: int = 8,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        cnn_channels: int = 16,
    ):
        """Initialize SpectraSense model.
        
        Args:
            input_dim: Input PSD vector dimension (192 frequency bins).
            patch_size: Number of frequency bins per patch.
            d_model: Transformer model dimension.
            num_layers: Number of Transformer encoder layers.
            num_heads: Number of attention heads.
            ffn_dim: FFN hidden dimension.
            dropout: Dropout rate.
            cnn_channels: CNN output channels after fusion.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Stage 1: Decoupled CNN Front-End
        self.cnn = DecoupledCNNFrontEnd()
        
        # Stage 2: Patch Tokenization + Positional Encoding
        self.tokenizer = PatchTokenizer(
            seq_length=input_dim,
            patch_size=patch_size,
            in_channels=cnn_channels,
            d_model=d_model,
        )
        
        # Stage 3: Transformer Encoder
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        
        # Stage 4: Multi-Task Heads
        self.heads = MultiTaskHead(in_features=d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through complete SpectraSense model.
        
        Args:
            x: Input PSD tensor of shape (B, 192) or (B, 1, 192).
            
        Returns:
            Dictionary with predictions:
                'pu': PU detection logits (B, 2)
                'mod': Modulation classification logits (B, 5)
                'snr': SNR estimation (B, 1)
        """
        # Ensure correct input shape: (B, 192) → (B, 1, 192)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Stage 1: CNN feature extraction
        cnn_features = self.cnn(x)  # (B, 192, 16)
        
        # Stage 2: Patch tokenization
        tokens = self.tokenizer(cnn_features)  # (B, 25, 64)
        
        # Stage 3: Transformer encoding
        cls_token = self.transformer(tokens)  # (B, 64)
        
        # Stage 4: Multi-task prediction
        predictions = self.heads(cls_token)
        
        return predictions
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token features without task heads.
        
        Useful for feature visualization and analysis.
        
        Args:
            x: Input PSD tensor of shape (B, 192) or (B, 1, 192).
            
        Returns:
            CLS token features of shape (B, 64).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        cnn_features = self.cnn(x)
        tokens = self.tokenizer(cnn_features)
        cls_token = self.transformer(tokens)
        
        return cls_token
    
    def forward_msm(
        self,
        x: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for Masked Spectrum Modeling pre-training.
        
        Masks specified token positions after CNN processing and
        reconstructs through the Transformer.
        
        Args:
            x: Input PSD tensor of shape (B, 192) or (B, 1, 192).
            mask_indices: Boolean mask of shape (B, 24) indicating masked patches.
            
        Returns:
            Reconstructed token features at masked positions.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Stage 1: CNN (no masking here - CNN sees full input)
        cnn_features = self.cnn(x)  # (B, 192, 16)
        
        # Stage 2: Tokenize
        tokens = self.tokenizer(cnn_features)  # (B, 25, 64)
        
        # Apply masking to patch tokens (not CLS)
        # mask_indices: (B, 24) → expand to (B, 24, 64)
        mask_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        
        # Create mask token (learnable)
        mask_token = self.mask_token.expand(x.shape[0], -1, -1)  # (B, 24, 64)
        
        # Replace masked positions with mask token
        tokens[:, 1:, :] = torch.where(mask_expanded, mask_token, tokens[:, 1:, :])
        
        # Stage 3: Transformer (full sequence)
        all_tokens = self.transformer.forward_full(tokens)  # (B, 25, 64)
        
        # Return patch tokens only (exclude CLS)
        return all_tokens[:, 1:, :]  # (B, 24, 64)
    
    def enable_msm(self) -> None:
        """Enable masked spectrum modeling by adding mask token and reconstruction head."""
        self.mask_token = nn.Parameter(torch.randn(1, 24, self.d_model) * 0.02)
        self.reconstruction_head = nn.Linear(self.d_model, self.patch_size * 16)
    
    def get_num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_parameter_breakdown(self) -> Dict[str, int]:
        """Get per-stage parameter counts.
        
        Returns:
            Dictionary with parameter counts for each stage.
        """
        return {
            "cnn_frontend": sum(p.numel() for p in self.cnn.parameters() if p.requires_grad),
            "tokenizer": sum(p.numel() for p in self.tokenizer.parameters() if p.requires_grad),
            "transformer": sum(p.numel() for p in self.transformer.parameters() if p.requires_grad),
            "task_heads": sum(p.numel() for p in self.heads.parameters() if p.requires_grad),
            "total": self.get_num_parameters(),
        }


def build_spectrasense(config: Optional[Dict] = None) -> SpectraSense:
    """Factory function to build SpectraSense model from config.
    
    Args:
        config: Optional model configuration dictionary.
        
    Returns:
        Initialized SpectraSense model.
    """
    if config is None:
        return SpectraSense()
    
    model_cfg = config.get("model", config)
    
    return SpectraSense(
        input_dim=model_cfg.get("input_dim", 192),
        patch_size=model_cfg.get("tokenizer", {}).get("patch_size", 8),
        d_model=model_cfg.get("transformer", {}).get("d_model", 64),
        num_layers=model_cfg.get("transformer", {}).get("num_layers", 2),
        num_heads=model_cfg.get("transformer", {}).get("num_heads", 4),
        ffn_dim=model_cfg.get("transformer", {}).get("ffn_dim", 128),
        dropout=model_cfg.get("transformer", {}).get("dropout", 0.1),
    )

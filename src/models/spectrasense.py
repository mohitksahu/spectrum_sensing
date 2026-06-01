"""SpectraSense: Complete Hybrid Decoupled-CNN + Transformer Model.

Integrates all four stages into a unified architecture for single-band
cognitive radio spectrum sensing. Supports both supervised multi-task
inference and self-supervised masked spectrum modeling pre-training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .decoupled_cnn import NoiseAwareCNNFrontEnd
from .tokenizer import OverlappingPatchTokenizer
from .transformer import SpectrumTransformerEncoder
from .multitask_head import MultiTaskHead


# Model constants
D_MODEL = 96
NUM_TRANSFORMER_LAYERS = 3
TRANSFORMER_FFN_DIM = 192
PATCH_SIZE = 8
PATCH_STRIDE = 4
NUM_PATCHES = 47
SEQ_LEN = 48
CNN_OUT_CHANNELS = 16
PATCH_FLAT_DIM = 128

SE_REDUCTION_RATIO = 4
CNN_FUSED_CHANNELS = 48
NOISE_POOL_KERNEL = 16
ENERGY_BIAS_INIT = 0.0


class MSMHead(nn.Module):
    """Masked Spectrum Modeling reconstruction head."""

    def __init__(self, d_model: int = D_MODEL, patch_flat_dim: int = PATCH_FLAT_DIM):
        super().__init__()
        self.reconstruct = nn.Linear(d_model, patch_flat_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        patch_tokens = z[:, 1:, :]
        return self.reconstruct(patch_tokens)


class PSDDenoisingHead(nn.Module):
    """Raw PSD denoising head for Phase 1A pre-training."""

    def __init__(self, d_model: int = D_MODEL, output_bins: int = 192):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, output_bins),
        )

    def forward(self, e_cls: torch.Tensor) -> torch.Tensor:
        return self.decoder(e_cls)


def compute_denoising_target(raw_psd: torch.Tensor, pool_kernel: int = 11) -> torch.Tensor:
    """Compute self-supervised denoising target from raw PSD."""
    x = raw_psd.unsqueeze(1)
    padding = pool_kernel // 2
    smoothed = F.avg_pool1d(x, kernel_size=pool_kernel, stride=1, padding=padding)
    return smoothed.squeeze(1)


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
        patch_size: int = PATCH_SIZE,
        d_model: int = D_MODEL,
        num_layers: int = NUM_TRANSFORMER_LAYERS,
        num_heads: int = 4,
        ffn_dim: int = TRANSFORMER_FFN_DIM,
        dropout: float = 0.1,
        cnn_channels: int = CNN_OUT_CHANNELS,
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
        self.patch_stride = PATCH_STRIDE
        self.num_patches = NUM_PATCHES
        self.seq_len = SEQ_LEN
        self.d_model = d_model
        self.patch_flat_dim = PATCH_FLAT_DIM
        
        # Stage 1: Noise-aware CNN front-end
        self.cnn = NoiseAwareCNNFrontEnd()
        
        # Stage 2: Overlapping Patch Tokenizer
        self.tokenizer = OverlappingPatchTokenizer(
            cnn_channels=cnn_channels,
            patch_size=patch_size,
            patch_stride=self.patch_stride,
            num_patches=self.num_patches,
            d_model=d_model,
        )
        
        # Stage 3: Energy-guided Transformer Encoder
        self.transformer = SpectrumTransformerEncoder(
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
    
    def forward(self, x: torch.Tensor, return_features: bool = False):
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
        f_local = self.cnn(x)  # (B, 16, 192)
        
        # Stage 2: Energy bias from local patch energies
        energy_bias = self.tokenizer.get_patch_energies(f_local)  # (B, 48)
        
        # Stage 2: Patch tokenization
        tokens = self.tokenizer(f_local)  # (B, 48, 96)
        
        # Stage 3: Transformer encoding
        _, cls_token = self.transformer(tokens, energy_bias=energy_bias)  # (B, 96)
        
        # Stage 4: Multi-task prediction
        predictions = self.heads(cls_token)
        if return_features:
            return predictions["pu"], predictions["mod"], predictions["snr"], cls_token
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
        
        f_local = self.cnn(x)
        energy_bias = self.tokenizer.get_patch_energies(f_local)
        tokens = self.tokenizer(f_local)
        _, cls_token = self.transformer(tokens, energy_bias=energy_bias)
        
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
            mask_indices: Boolean mask of shape (B, 47) indicating masked patches.
            
        Returns:
            Reconstructed token features at masked positions.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Stage 1: CNN (no masking here - CNN sees full input)
        f_local = self.cnn(x)  # (B, 16, 192)
        energy_bias = self.tokenizer.get_patch_energies(f_local)
        
        # Stage 2: Tokenize
        tokens = self.tokenizer(f_local)  # (B, 48, 96)
        
        # Apply masking to patch tokens (not CLS)
        # mask_indices: (B, 47) → expand to (B, 47, 96)
        mask_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        
        # Create mask token (learnable)
        mask_token = self.mask_token.expand(x.shape[0], -1, -1)  # (B, 47, 96)
        
        # Replace masked positions with mask token
        tokens[:, 1:, :] = torch.where(mask_expanded, mask_token, tokens[:, 1:, :])
        
        # Stage 3: Transformer (full sequence)
        all_tokens, _ = self.transformer(tokens, energy_bias=energy_bias)  # (B, 48, 96)
        
        # Return patch tokens only (exclude CLS)
        return all_tokens[:, 1:, :]  # (B, 47, 96)
    
    def enable_msm(self) -> None:
        """Enable masked spectrum modeling by adding mask token and reconstruction head."""
        self.mask_token = nn.Parameter(torch.randn(1, self.num_patches, self.d_model) * 0.02)
        self.msm_head = MSMHead(d_model=self.d_model, patch_flat_dim=self.patch_flat_dim)
        self.reconstruction_head = self.msm_head.reconstruct
    
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
        patch_size=model_cfg.get("tokenizer", {}).get("patch_size", PATCH_SIZE),
        d_model=model_cfg.get("transformer", {}).get("d_model", D_MODEL),
        num_layers=model_cfg.get("transformer", {}).get("num_layers", NUM_TRANSFORMER_LAYERS),
        num_heads=model_cfg.get("transformer", {}).get("num_heads", 4),
        ffn_dim=model_cfg.get("transformer", {}).get("ffn_dim", TRANSFORMER_FFN_DIM),
        dropout=model_cfg.get("transformer", {}).get("dropout", 0.1),
    )

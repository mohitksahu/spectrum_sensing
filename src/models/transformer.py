"""Stage 3: Energy-guided transformer encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


D_MODEL = 96
NUM_TRANSFORMER_LAYERS = 3
TRANSFORMER_FFN_DIM = 192
ENERGY_BIAS_INIT = 0.0


class EnergyGuidedTransformerLayer(nn.Module):
    """Pre-LN transformer layer with energy-guided attention bias."""

    def __init__(
        self,
        d_model: int = D_MODEL,
        num_heads: int = 4,
        ffn_dim: int = TRANSFORMER_FFN_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.energy_lambda = nn.Parameter(torch.tensor(ENERGY_BIAS_INIT))

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, energy_bias: torch.Tensor | None = None) -> torch.Tensor:
        b, l, d = x.shape
        x_norm = self.norm1(x)

        q = self.w_q(x_norm).view(b, l, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x_norm).view(b, l, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x_norm).view(b, l, self.num_heads, self.d_k).transpose(1, 2)

        scale = self.d_k ** -0.5
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale

        if energy_bias is not None:
            e_bias = energy_bias.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits + self.energy_lambda * e_bias

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, l, d)
        attn_out = self.w_o(attn_out)

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class SpectrumTransformerEncoder(nn.Module):
    """Stack of energy-guided transformer layers."""

    def __init__(
        self,
        num_layers: int = NUM_TRANSFORMER_LAYERS,
        d_model: int = D_MODEL,
        num_heads: int = 4,
        ffn_dim: int = TRANSFORMER_FFN_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [EnergyGuidedTransformerLayer(d_model, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, t: torch.Tensor, energy_bias: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = t
        for layer in self.layers:
            x = layer(x, energy_bias=energy_bias)
        x = self.final_norm(x)
        e_cls = x[:, 0, :]
        return x, e_cls


class TransformerEncoder(nn.Module):
    """Compatibility wrapper exposing legacy forward/forward_full API."""

    def __init__(
        self,
        num_layers: int = NUM_TRANSFORMER_LAYERS,
        d_model: int = D_MODEL,
        num_heads: int = 4,
        ffn_dim: int = TRANSFORMER_FFN_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = SpectrumTransformerEncoder(num_layers, d_model, num_heads, ffn_dim, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, energy_bias: torch.Tensor | None = None) -> torch.Tensor:
        _, e_cls = self.encoder(x, energy_bias=energy_bias)
        return e_cls

    def forward_full(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        energy_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z, _ = self.encoder(x, energy_bias=energy_bias)
        return z

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

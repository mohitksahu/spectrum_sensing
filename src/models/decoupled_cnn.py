"""Stage 1: Noise-aware CNN front-end for spectral feature extraction."""

import torch
import torch.nn as nn


SE_REDUCTION_RATIO = 4
CNN_OUT_CHANNELS = 16
CNN_FUSED_CHANNELS = 48
NOISE_POOL_KERNEL = 16


class SqueezeExcitation1D(nn.Module):
    """Channel attention module for 1D feature maps."""

    def __init__(self, num_channels: int, reduction_ratio: int = 4):
        super().__init__()
        bottleneck = max(num_channels // reduction_ratio, 4)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, num_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.squeeze(x)
        w = self.excitation(w)
        w = w.unsqueeze(-1)
        return x * w


class NoiseAwareCNNFrontEnd(nn.Module):
    """Noise-aware decoupled CNN front-end.

    Input: (B, 1, 192)
    Output: (B, 16, 192)
    """

    def __init__(self):
        super().__init__()

        self.shared_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )

        left_pad = (NOISE_POOL_KERNEL - 1) // 2
        right_pad = NOISE_POOL_KERNEL // 2
        self.noise_pool = nn.Sequential(
            nn.ReplicationPad1d((left_pad, right_pad)),
            nn.AvgPool1d(kernel_size=NOISE_POOL_KERNEL, stride=1, padding=0),
        )

        self.path_a = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )
        self.path_b = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=7, padding=3, dilation=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )
        self.path_c = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )

        self.se_block = SqueezeExcitation1D(
            num_channels=CNN_FUSED_CHANNELS,
            reduction_ratio=SE_REDUCTION_RATIO,
        )

        self.channel_reduce = nn.Sequential(
            nn.Conv1d(CNN_FUSED_CHANNELS, CNN_OUT_CHANNELS, kernel_size=1),
            nn.BatchNorm1d(CNN_OUT_CHANNELS),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_shared = self.shared_block(x)
        f_noise = self.noise_pool(f_shared)
        f_diff = f_shared - f_noise

        f_a = self.path_a(f_diff)
        f_b = self.path_b(f_diff)
        f_c = self.path_c(f_diff)

        f_cat = torch.cat([f_a, f_b, f_c], dim=1)
        f_attended = self.se_block(f_cat)
        f_local = self.channel_reduce(f_attended)
        return f_local

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Backward-compatible alias for existing imports.
DecoupledCNNFrontEnd = NoiseAwareCNNFrontEnd

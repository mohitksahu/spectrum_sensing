"""Stage 2: Overlapping patch tokenization and positional encoding."""

import torch
import torch.nn as nn


PATCH_SIZE = 8
PATCH_STRIDE = 4
NUM_PATCHES = 47
D_MODEL = 96
CNN_OUT_CHANNELS = 16


class OverlappingPatchTokenizer(nn.Module):
    """Convert CNN feature map to overlapping patch tokens.

    Input: (B, 16, 192)
    Output: (B, 48, 96)
    """

    def __init__(
        self,
        cnn_channels: int = CNN_OUT_CHANNELS,
        patch_size: int = PATCH_SIZE,
        patch_stride: int = PATCH_STRIDE,
        num_patches: int = NUM_PATCHES,
        d_model: int = D_MODEL,
        **_: dict,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.num_patches = num_patches
        self.d_model = d_model

        patch_flat_dim = patch_size * cnn_channels
        self.patch_projection = nn.Linear(patch_flat_dim, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_encoding = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    def forward(self, f_local: torch.Tensor) -> torch.Tensor:
        b = f_local.shape[0]
        patches = f_local.unfold(dimension=2, size=self.patch_size, step=self.patch_stride)
        patches = patches.permute(0, 2, 1, 3)
        patches = patches.reshape(b, self.num_patches, -1)

        patch_tokens = self.patch_projection(patches)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        t = torch.cat([cls_tokens, patch_tokens], dim=1)
        t = t + self.pos_encoding
        return t

    def get_patch_energies(self, f_local: torch.Tensor) -> torch.Tensor:
        b = f_local.shape[0]
        patches = f_local.unfold(dimension=2, size=self.patch_size, step=self.patch_stride)
        patches = patches.permute(0, 2, 1, 3).reshape(b, self.num_patches, -1)

        patch_energies = patches.pow(2).mean(dim=-1)
        e_min = patch_energies.min(dim=1, keepdim=True).values
        e_max = patch_energies.max(dim=1, keepdim=True).values
        patch_energies_norm = (patch_energies - e_min) / (e_max - e_min + 1e-8)

        cls_energy = torch.ones(b, 1, device=f_local.device)
        energies = torch.cat([cls_energy, patch_energies_norm], dim=1)
        return energies

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Backward-compatible alias.
PatchTokenizer = OverlappingPatchTokenizer

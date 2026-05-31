"""Quick unit test to validate model shapes and a forward pass."""

import torch
from src.models.spectrasense import build_spectrasense


def run_shape_checks():
    model = build_spectrasense()
    print("Model built. Parameter breakdown:")
    print(model.get_parameter_breakdown())

    model.eval()
    B = 4
    x = torch.randn(B, 192)
    outputs = model(x)

    assert isinstance(outputs, dict)
    assert "pu" in outputs and "mod" in outputs and "snr" in outputs
    assert outputs["pu"].shape == (B, 2)
    assert outputs["mod"].shape[0] == B
    assert outputs["snr"].shape == (B, 1)

    print("Supervised forward shapes OK")

    # MSM check
    model.enable_msm()
    mask_ratio = 0.2
    num_patches = 24
    mask = torch.rand(B, num_patches) < mask_ratio
    msm_tokens = model.forward_msm(x, mask)
    assert msm_tokens.shape == (B, num_patches, model.d_model)

    # Reconstruction head
    recon = model.reconstruction_head(msm_tokens)
    assert recon.shape == (B, num_patches, model.patch_size * 16)
    print("MSM forward and reconstruction shapes OK")


if __name__ == "__main__":
    run_shape_checks()
    print("All tests passed")

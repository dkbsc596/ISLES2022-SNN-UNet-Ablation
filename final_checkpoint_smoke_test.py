import os
import sys
import torch

ROOT = "/home/krishna_19622/dataset/stroke_3d_snn1_xai"
sys.path.insert(0, ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("BLOCK50 FINAL CHECKPOINT SMOKE TEST")
print("=" * 70)
print("Device:", DEVICE)
print()

# ============================================================
# A0 — EXACT ARCHITECTURE FROM A0_train.py
# ============================================================

import torch.nn as nn


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                bias=False,
                padding=1
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                bias=False,
                padding=1
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv3D(3, 16)
        self.enc2 = DoubleConv3D(16, 32)
        self.enc3 = DoubleConv3D(32, 64)

        self.pool = nn.MaxPool3d(
            kernel_size=2,
            stride=2
        )

        self.bottleneck = DoubleConv3D(64, 128)

        self.up3 = nn.ConvTranspose3d(
            128, 64,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv3D(128, 64)

        self.up2 = nn.ConvTranspose3d(
            64, 32,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv3D(64, 32)

        self.up1 = nn.ConvTranspose3d(
            32, 16,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv3D(32, 16)

        self.out_conv = nn.Conv3d(
            16, 1,
            kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        b = self.bottleneck(
            self.pool(e3)
        )

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


# ============================================================
# A1/A2/A3 — CONTROLLED SNN U-NET
# ============================================================

from models.controlled_snn_unet import (
    A1SNNUNet3D,
    A2SNNUNet3D,
    A3SNNUNet3D
)


# ============================================================
# CHECKPOINT TEST FUNCTION
# ============================================================

def test_checkpoint(name, model, checkpoint_path, is_snn=False):

    print("-" * 70)
    print(name)
    print("-" * 70)

    if not os.path.exists(checkpoint_path):
        print("CHECKPOINT: MISSING")
        print(checkpoint_path)
        return False

    print("Checkpoint:", checkpoint_path)

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    state = ckpt["model_state_dict"]

    print("Checkpoint epoch:", ckpt.get("epoch"))
    print("Best epoch:", ckpt.get("best_epoch"))
    print("Best validation Dice:", ckpt.get("best_val_dice"))

    model = model.to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    # 1 × 3 × 128 × 128 × 64
    x = torch.randn(
        1, 3, 128, 128, 64,
        device=DEVICE
    )

    if is_snn:
        from spikingjelly.activation_based import functional
        functional.reset_net(model)

    with torch.no_grad():

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(DEVICE.type == "cuda")
        ):
            y = model(x)

    if is_snn:
        from spikingjelly.activation_based import functional
        functional.reset_net(model)

    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Output finite:", bool(torch.isfinite(y).all().item()))

    expected = (1, 1, 128, 128, 64)

    if tuple(y.shape) != expected:
        print("RESULT: FAIL — wrong output shape")
        return False

    if not torch.isfinite(y).all():
        print("RESULT: FAIL — non-finite output")
        return False

    print("RESULT: PASS")
    return True


# ============================================================
# RUN ALL FOUR
# ============================================================

results = {}

results["A0"] = test_checkpoint(
    "A0 — 3D CNN U-Net",
    UNet3D(),
    "checkpoints/block50_epoch/A0/A0_best.pth",
    False
)

results["A1"] = test_checkpoint(
    "A1 — 3D SNN U-Net",
    A1SNNUNet3D(),
    "checkpoints/block50_epoch/A1/A1_best.pth",
    True
)

results["A2"] = test_checkpoint(
    "A2 — 3D SNN U-Net + Attention",
    A2SNNUNet3D(),
    "checkpoints/block50_epoch/A2/A2_best.pth",
    True
)

results["A3"] = test_checkpoint(
    "A3 — 3D SNN U-Net + Attention + Residual",
    A3SNNUNet3D(),
    "checkpoints/block50_epoch/A3/A3_best.pth",
    True
)


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)
print("FINAL SMOKE TEST RESULT")
print("=" * 70)

for name, result in results.items():
    print(f"{name}: {'PASS' if result else 'FAIL'}")

if all(results.values()):
    print()
    print("ALL FOUR CHECKPOINTS PASSED.")
    print("SAFE TO PROCEED TO FINAL HELD-OUT TEST EVALUATION.")
else:
    print()
    print("DO NOT RUN FINAL TEST.")
    print("At least one checkpoint failed.")

print("=" * 70)

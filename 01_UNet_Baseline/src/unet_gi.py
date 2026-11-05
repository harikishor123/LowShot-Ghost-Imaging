"""Lightweight PyTorch U-Net for 64x64 grayscale ghost-image reconstruction (Phase 2B).

    input  1x64x64  (calibrated, clipped DGI-N estimate)
    enc1   [Conv3x3-ReLU]x2, base      ch, 64x64  -> MaxPool
    enc2   [Conv3x3-ReLU]x2, 2*base    ch, 32x32  -> MaxPool
    enc3   [Conv3x3-ReLU]x2, 4*base    ch, 16x16  -> MaxPool
    bottle [Conv3x3-ReLU]x2, 8*base    ch,  8x8
    dec3..dec1: ConvTranspose2x2 (stride 2) -> concat skip -> [Conv3x3-ReLU]x2
    head   Conv1x1 -> 1x64x64, LINEAR output (no sigmoid)

The output is on the same [0, 1] physical scale as the target; it is clipped to
[0, 1] only for images/metrics, never inside the loss. No normalisation layers,
matching the earlier photon U-Net; base=32 gives ~1.9 M parameters.
"""

from __future__ import annotations

import torch
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
    )


class UNetGI(nn.Module):
    def __init__(self, base: int = 32, depth: int = 3, in_ch: int = 1, out_ch: int = 1):
        super().__init__()
        chs = [base * 2 ** d for d in range(depth)]
        self.enc = nn.ModuleList()
        c = in_ch
        for ch in chs:
            self.enc.append(_block(c, ch))
            c = ch
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _block(c, c * 2)
        c = c * 2
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for ch in reversed(chs):
            self.up.append(nn.ConvTranspose2d(c, ch, 2, stride=2))
            self.dec.append(_block(2 * ch, ch))
            c = ch
        self.head = nn.Conv2d(c, out_ch, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, block, skip in zip(self.up, self.dec, reversed(skips)):
            x = block(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

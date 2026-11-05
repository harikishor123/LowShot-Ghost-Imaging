"""
Conditional Diffusion U-Net for residual ghost-image reconstruction (E3).

Architecture adapted from DiffTransFuse Stage1DUNet with one modification:
  - in_ch = 2 (concat of noisy residual r_t + DGI conditioning)
  - out_ch = 1 (ε-prediction, same spatial dims as input)

Channel progression: 2 → 32 → 64 → 128 → 256 → 128 → 64 → 32 → 1
Timestep conditioning: sinusoidal embedding (dim=128) → MLP → additive injection
Downsampling: strided Conv2d(4, stride=2, padding=1)
Upsampling: ConvTranspose2d(4, stride=2, padding=1)
Skip connections: encoder-to-decoder concatenation (channel doubling)
Output: linear (no sigmoid) — predicting noise ε ∈ ℝ
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = timestep.device
        emb_scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=device) * -emb_scale)
        embedding = timestep.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([embedding.sin(), embedding.cos()], dim=1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResBlock(nn.Module):
    """Residual block with additive timestep conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        groups = min(8, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(x)
        hidden = self.norm1(hidden)
        hidden = F.silu(hidden)
        hidden = hidden + self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)
        hidden = F.silu(hidden)
        return hidden + self.skip(x)


class DiffusionUNet(nn.Module):
    """
    Conditional diffusion U-Net for ε-prediction.

    Input:  concat(r_t, DGI) → [B, 2, 64, 64]
    Output: ε̂ → [B, 1, 64, 64]

    Architecture matches DiffTransFuse Stage1DUNet exactly, except:
      - in_ch=2 (concatenated conditioning)
      - final Conv2d outputs 1 channel
    """

    def __init__(
        self,
        in_ch: int = 2,
        out_ch: int = 1,
        base_ch: int = 32,
        time_dim: int = 128,
    ) -> None:
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.enc1 = ResBlock(in_ch, base_ch, time_dim)           # 2 → 32
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)
        self.enc2 = ResBlock(base_ch, base_ch * 2, time_dim)     # 32 → 64
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)
        self.enc3 = ResBlock(base_ch * 2, base_ch * 4, time_dim) # 64 → 128
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1)

        # Bottleneck
        self.mid = ResBlock(base_ch * 4, base_ch * 8, time_dim)  # 128 → 256

        # Decoder (skip connections double input channels)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, stride=2, padding=1)
        self.dec3 = ResBlock(base_ch * 8, base_ch * 4, time_dim)  # cat(128+128)=256 → 128
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec2 = ResBlock(base_ch * 4, base_ch * 2, time_dim)  # cat(64+64)=128 → 64
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec1 = ResBlock(base_ch * 2, base_ch, time_dim)      # cat(32+32)=64 → 32

        # Output head — linear, no activation (predicting noise ε ∈ ℝ)
        self.out = nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:        [B, 2, H, W] — concat(r_t, DGI)
            timestep: [B] — integer timesteps in [0, T-1]
        Returns:
            eps_pred: [B, 1, H, W] — predicted noise
        """
        time_emb = self.time_mlp(timestep)

        # Encoder
        e1 = self.enc1(x, time_emb)          # [B, 32, 64, 64]
        e2 = self.enc2(self.down1(e1), time_emb)  # [B, 64, 32, 32]
        e3 = self.enc3(self.down2(e2), time_emb)  # [B, 128, 16, 16]

        # Bottleneck
        mid = self.mid(self.down3(e3), time_emb)   # [B, 256, 8, 8]

        # Decoder with skip connections
        h = self.up3(mid)                            # [B, 128, 16, 16]
        h = self.dec3(torch.cat([h, e3], dim=1), time_emb)  # [B, 128, 16, 16]
        h = self.up2(h)                              # [B, 64, 32, 32]
        h = self.dec2(torch.cat([h, e2], dim=1), time_emb)  # [B, 64, 32, 32]
        h = self.up1(h)                              # [B, 32, 64, 64]
        h = self.dec1(torch.cat([h, e1], dim=1), time_emb)  # [B, 32, 64, 64]

        return self.out(h)                           # [B, 1, 64, 64]

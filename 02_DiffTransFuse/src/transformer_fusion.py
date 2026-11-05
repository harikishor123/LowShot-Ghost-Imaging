"""
TransFusionNet — Stage 2 of DiffTransFuse for ghost-image reconstruction (E4).

Takes frozen E3 diffusion encoder features + DGI → reconstructed object O
in a single forward pass (no iterative DDIM sampling).

Architecture:
  1. Frozen E3 encoder extracts multi-scale features from DGI+DGI at t=0
     → e1 [B,32,64,64], e2 [B,64,32,32], e3 [B,128,16,16], mid [B,256,8,8]
  2. Transformer processes bottleneck tokens (8×8 = 64 tokens, dim=256)
  3. CNN decoder fuses encoder skips via channel attention at each scale
  4. Output: O_pred [B, 1, 64, 64] — linear head, clamp at inference

Input strategy: DGI+DGI at t=0 was selected by feature-variance analysis
(e1_var=0.86, highest among 6 candidates).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w


class FusionBlock(nn.Module):
    """Fuse upsampled features with encoder skip connection + DGI at one scale."""

    def __init__(self, up_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        in_ch = up_ch + skip_ch + 1  # +1 for DGI channel
        groups = min(8, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.ca = ChannelAttention(out_ch)

    def forward(
        self, up: torch.Tensor, skip: torch.Tensor, dgi: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([up, skip, dgi], dim=1)
        x = F.silu(self.norm1(self.conv1(x)))
        x = F.silu(self.norm2(self.conv2(x)))
        return self.ca(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: self-attention + FFN."""

    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed)[0]
        x = x + self.ff(self.norm2(x))
        return x


class TransFusionNet(nn.Module):
    """
    Stage 2 of DiffTransFuse: single-pass reconstruction.

    Frozen E3 encoder → multi-scale features → Transformer at bottleneck
    → CNN decoder with channel-attention fusion → O_pred.

    Only the transformer and decoder are trainable (~4.5M params).
    The frozen E3 encoder (~3M params) is used as a feature extractor.
    """

    def __init__(
        self,
        frozen_encoder: nn.Module,
        embed_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        base_ch: int = 32,
    ) -> None:
        super().__init__()

        # Frozen E3 encoder (not registered as submodule to avoid saving its params)
        self.encoder = frozen_encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # Learnable positional embedding for 8×8 = 64 spatial tokens
        self.pos_embed = nn.Parameter(torch.randn(1, 64, embed_dim) * 0.02)

        # Bottleneck Transformer (operates on 64 tokens of dim=256)
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(embed_dim, n_heads) for _ in range(n_layers)]
        )
        self.transformer_norm = nn.LayerNorm(embed_dim)

        # Decoder: progressive upsampling with channel-attention fusion
        # 8→16: transformer out (256) + e3 skip (128) + DGI_16 (1) → 128
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, stride=2, padding=1)
        self.fuse3 = FusionBlock(base_ch * 4, base_ch * 4, base_ch * 4)

        # 16→32: (128) + e2 skip (64) + DGI_32 (1) → 64
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.fuse2 = FusionBlock(base_ch * 2, base_ch * 2, base_ch * 2)

        # 32→64: (64) + e1 skip (32) + DGI (1) → 32
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.fuse1 = FusionBlock(base_ch, base_ch, base_ch)

        # Output head — linear, no activation (MSE drives predictions toward [0,1])
        self.out = nn.Conv2d(base_ch, 1, 3, padding=1)

    def _extract_frozen_features(
        self, dgi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract multi-scale features from frozen E3 encoder with DGI+DGI at t=0."""
        self.encoder.eval()
        with torch.no_grad():
            B = dgi.shape[0]
            x_input = torch.cat([dgi, dgi], dim=1)  # [B, 2, 64, 64]
            t_zero = torch.zeros(B, device=dgi.device, dtype=torch.long)
            time_emb = self.encoder.time_mlp(t_zero)

            e1 = self.encoder.enc1(x_input, time_emb)                # [B, 32, 64, 64]
            e2 = self.encoder.enc2(self.encoder.down1(e1), time_emb)  # [B, 64, 32, 32]
            e3 = self.encoder.enc3(self.encoder.down2(e2), time_emb)  # [B, 128, 16, 16]
            mid = self.encoder.mid(self.encoder.down3(e3), time_emb)  # [B, 256, 8, 8]

        return e1, e2, e3, mid

    def forward(self, dgi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dgi: [B, 1, 64, 64] — DGI input image
        Returns:
            o_pred: [B, 1, 64, 64] — reconstructed object (linear, no sigmoid)
        """
        e1, e2, e3, mid = self._extract_frozen_features(dgi)

        # Transformer on bottleneck tokens
        B, C, H, W = mid.shape  # [B, 256, 8, 8]
        tokens = mid.flatten(2).transpose(1, 2)  # [B, 64, 256]
        tokens = tokens + self.pos_embed
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        tokens = self.transformer_norm(tokens)
        h = tokens.transpose(1, 2).reshape(B, C, H, W)  # [B, 256, 8, 8]

        # DGI at each decoder scale
        dgi_16 = F.interpolate(dgi, size=(16, 16), mode="bilinear", align_corners=False)
        dgi_32 = F.interpolate(dgi, size=(32, 32), mode="bilinear", align_corners=False)

        # Progressive decoder with skip-attention fusion
        h = self.up3(h)                     # [B, 128, 16, 16]
        h = self.fuse3(h, e3, dgi_16)       # [B, 128, 16, 16]

        h = self.up2(h)                     # [B, 64, 32, 32]
        h = self.fuse2(h, e2, dgi_32)       # [B, 64, 32, 32]

        h = self.up1(h)                     # [B, 32, 64, 64]
        h = self.fuse1(h, e1, dgi)          # [B, 32, 64, 64]

        return self.out(h)                  # [B, 1, 64, 64]

    def train(self, mode: bool = True):
        """Override to keep frozen encoder in eval mode."""
        super().train(mode)
        self.encoder.eval()
        return self

    def trainable_parameters(self):
        """Iterator over only the trainable (non-frozen) parameters."""
        for name, p in self.named_parameters():
            if p.requires_grad:
                yield p

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_frozen_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

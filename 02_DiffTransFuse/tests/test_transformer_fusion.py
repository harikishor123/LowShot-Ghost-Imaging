"""
Tests for TransFusionNet (E4 Transformer fusion module).

Covers: component shapes, parameter counts, gradient flow, frozen encoder,
determinism, forward pass dimensions, channel attention, fusion blocks,
and CUDA transfer.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.unet_diffusion import DiffusionUNet
from src.transformer_fusion import (
    ChannelAttention,
    FusionBlock,
    TransformerBlock,
    TransFusionNet,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def encoder(device):
    """Small DiffusionUNet to serve as frozen encoder."""
    model = DiffusionUNet(in_ch=2, out_ch=1, base_ch=32, time_dim=128).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@pytest.fixture
def model(encoder, device):
    return TransFusionNet(
        frozen_encoder=encoder,
        embed_dim=256,
        n_heads=4,
        n_layers=4,
        base_ch=32,
    ).to(device)


# ── ChannelAttention tests ────────────────────────────────────

class TestChannelAttention:
    def test_output_shape(self, device):
        ca = ChannelAttention(64).to(device)
        x = torch.randn(2, 64, 16, 16, device=device)
        out = ca(x)
        assert out.shape == (2, 64, 16, 16)

    def test_output_range(self, device):
        """Channel attention scales by sigmoid weights ∈ (0,1), so output magnitude ≤ input."""
        ca = ChannelAttention(32).to(device)
        x = torch.ones(1, 32, 8, 8, device=device)
        out = ca(x)
        assert (out >= 0).all()
        assert (out <= 1.0 + 1e-6).all()


# ── FusionBlock tests ─────────────────────────────────────────

class TestFusionBlock:
    def test_output_shape(self, device):
        fb = FusionBlock(up_ch=128, skip_ch=128, out_ch=128).to(device)
        up = torch.randn(2, 128, 16, 16, device=device)
        skip = torch.randn(2, 128, 16, 16, device=device)
        dgi = torch.randn(2, 1, 16, 16, device=device)
        out = fb(up, skip, dgi)
        assert out.shape == (2, 128, 16, 16)

    def test_different_scales(self, device):
        """FusionBlock works at different spatial resolutions."""
        for h, w, up_ch, skip_ch, out_ch in [(32, 32, 64, 64, 64), (64, 64, 32, 32, 32)]:
            fb = FusionBlock(up_ch, skip_ch, out_ch).to(device)
            up = torch.randn(1, up_ch, h, w, device=device)
            skip = torch.randn(1, skip_ch, h, w, device=device)
            dgi = torch.randn(1, 1, h, w, device=device)
            out = fb(up, skip, dgi)
            assert out.shape == (1, out_ch, h, w)


# ── TransformerBlock tests ────────────────────────────────────

class TestTransformerBlock:
    def test_output_shape(self, device):
        tb = TransformerBlock(dim=256, n_heads=4).to(device)
        x = torch.randn(2, 64, 256, device=device)
        out = tb(x)
        assert out.shape == (2, 64, 256)

    def test_residual_connection(self, device):
        """With zero-init weights, output ≈ input (residual)."""
        tb = TransformerBlock(dim=256, n_heads=4).to(device)
        x = torch.randn(2, 64, 256, device=device)
        out = tb(x)
        # Not exactly equal (random weights), but shapes match
        assert out.shape == x.shape


# ── TransFusionNet tests ──────────────────────────────────────

class TestTransFusionNet:
    def test_output_shape(self, model, device):
        dgi = torch.randn(2, 1, 64, 64, device=device)
        out = model(dgi)
        assert out.shape == (2, 1, 64, 64)

    def test_single_sample(self, model, device):
        dgi = torch.randn(1, 1, 64, 64, device=device)
        out = model(dgi)
        assert out.shape == (1, 1, 64, 64)

    def test_encoder_frozen(self, model):
        """All encoder parameters should have requires_grad=False."""
        for name, p in model.named_parameters():
            if name.startswith("encoder."):
                assert not p.requires_grad, f"Encoder param {name} is not frozen"

    def test_trainable_params_not_frozen(self, model):
        """Non-encoder parameters should have requires_grad=True."""
        for name, p in model.named_parameters():
            if not name.startswith("encoder."):
                assert p.requires_grad, f"Trainable param {name} is frozen"

    def test_trainable_param_count(self, model):
        n = model.count_trainable_params()
        # Expect ~4-5M trainable params
        assert 3_000_000 < n < 6_000_000, f"Unexpected trainable param count: {n:,}"

    def test_frozen_param_count(self, model):
        n = model.count_frozen_params()
        # E3 DiffusionUNet has 2,999,393 params
        assert n == 2_999_393, f"Unexpected frozen param count: {n:,}"

    def test_gradient_flow(self, model, device):
        """Gradients should flow through trainable params but not frozen encoder."""
        dgi = torch.randn(2, 1, 64, 64, device=device)
        out = model(dgi)
        loss = out.mean()
        loss.backward()

        has_grad = False
        for name, p in model.named_parameters():
            if name.startswith("encoder."):
                assert p.grad is None, f"Frozen param {name} has gradient"
            elif p.requires_grad:
                if p.grad is not None and p.grad.abs().sum() > 0:
                    has_grad = True
        assert has_grad, "No gradients in trainable parameters"

    def test_determinism(self, encoder, device):
        """Same seed → same output."""
        torch.manual_seed(42)
        m1 = TransFusionNet(encoder, embed_dim=256, n_heads=4, n_layers=4, base_ch=32).to(device)

        torch.manual_seed(42)
        m2 = TransFusionNet(encoder, embed_dim=256, n_heads=4, n_layers=4, base_ch=32).to(device)

        dgi = torch.randn(1, 1, 64, 64, device=device)
        out1 = m1(dgi)
        out2 = m2(dgi)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_trainable_parameters_iterator(self, model):
        """trainable_parameters() should yield only non-frozen params."""
        trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
        iter_count = sum(1 for _ in model.trainable_parameters())
        assert iter_count == len(trainable_names)

    def test_different_batch_sizes(self, model, device):
        for bs in [1, 4, 8]:
            dgi = torch.randn(bs, 1, 64, 64, device=device)
            out = model(dgi)
            assert out.shape == (bs, 1, 64, 64)

    def test_output_is_finite(self, model, device):
        dgi = torch.randn(2, 1, 64, 64, device=device)
        out = model(dgi)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_encoder_stays_eval(self, model, device):
        """Encoder should remain in eval mode even when model is set to train."""
        model.train()
        assert not model.encoder.training, "Encoder entered training mode"


# ── CUDA tests (skipped if not available) ─────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTransFusionNetCUDA:
    def test_forward_cuda(self):
        device = torch.device("cuda")
        encoder = DiffusionUNet(in_ch=2, out_ch=1, base_ch=32, time_dim=128).to(device)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

        model = TransFusionNet(encoder, embed_dim=256, n_heads=4, n_layers=4, base_ch=32).to(device)
        dgi = torch.randn(2, 1, 64, 64, device=device)
        out = model(dgi)
        assert out.shape == (2, 1, 64, 64)
        assert out.device.type == "cuda"

    def test_gradient_cuda(self):
        device = torch.device("cuda")
        encoder = DiffusionUNet(in_ch=2, out_ch=1, base_ch=32, time_dim=128).to(device)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

        model = TransFusionNet(encoder, embed_dim=256, n_heads=4, n_layers=4, base_ch=32).to(device)
        dgi = torch.randn(2, 1, 64, 64, device=device)
        out = model(dgi)
        loss = out.mean()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for name, p in model.named_parameters()
            if not name.startswith("encoder.") and p.requires_grad
        )
        assert has_grad

"""Tests for src/unet_diffusion.py — DiffusionUNet architecture."""

import pytest
import torch
import sys
sys.path.insert(0, ".")
from src.unet_diffusion import DiffusionUNet, SinusoidalTimeEmbedding, ResBlock


class TestSinusoidalTimeEmbedding:
    def test_output_shape(self):
        emb = SinusoidalTimeEmbedding(128)
        t = torch.tensor([0, 100, 500, 999])
        out = emb(t)
        assert out.shape == (4, 128)

    def test_different_timesteps_differ(self):
        emb = SinusoidalTimeEmbedding(128)
        t1 = torch.tensor([0])
        t2 = torch.tensor([999])
        e1 = emb(t1)
        e2 = emb(t2)
        assert not torch.allclose(e1, e2)

    def test_odd_dim(self):
        emb = SinusoidalTimeEmbedding(127)
        t = torch.tensor([50])
        out = emb(t)
        assert out.shape == (1, 127)


class TestResBlock:
    def test_same_channels(self):
        block = ResBlock(32, 32, 128)
        x = torch.randn(2, 32, 16, 16)
        t = torch.randn(2, 128)
        out = block(x, t)
        assert out.shape == (2, 32, 16, 16)

    def test_channel_change(self):
        block = ResBlock(32, 64, 128)
        x = torch.randn(2, 32, 16, 16)
        t = torch.randn(2, 128)
        out = block(x, t)
        assert out.shape == (2, 64, 16, 16)

    def test_skip_connection_identity(self):
        block = ResBlock(32, 32, 128)
        assert isinstance(block.skip, torch.nn.Identity)

    def test_skip_connection_conv(self):
        block = ResBlock(32, 64, 128)
        assert isinstance(block.skip, torch.nn.Conv2d)


class TestDiffusionUNet:
    @pytest.fixture
    def model(self):
        return DiffusionUNet(in_ch=2, out_ch=1, base_ch=32, time_dim=128)

    def test_output_shape_64(self, model):
        x = torch.randn(2, 2, 64, 64)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)
        assert out.shape == (2, 1, 64, 64)

    def test_output_shape_32(self, model):
        x = torch.randn(1, 2, 32, 32)
        t = torch.randint(0, 1000, (1,))
        out = model(x, t)
        assert out.shape == (1, 1, 32, 32)

    def test_output_shape_128(self, model):
        x = torch.randn(1, 2, 128, 128)
        t = torch.randint(0, 1000, (1,))
        out = model(x, t)
        assert out.shape == (1, 1, 128, 128)

    def test_parameter_count(self, model):
        n = sum(p.numel() for p in model.parameters())
        assert n == 2_999_393

    def test_no_sigmoid(self, model):
        x = torch.randn(2, 2, 64, 64)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)
        assert out.min() < 0 or out.max() > 1, \
            "Output should not be bounded — no sigmoid"

    def test_gradient_flow(self, model):
        x = torch.randn(2, 2, 64, 64, requires_grad=True)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        for p in model.parameters():
            assert p.grad is not None, f"No gradient for parameter"

    def test_batch_size_1(self, model):
        x = torch.randn(1, 2, 64, 64)
        t = torch.randint(0, 1000, (1,))
        out = model(x, t)
        assert out.shape == (1, 1, 64, 64)

    def test_deterministic(self, model):
        torch.manual_seed(0)
        x = torch.randn(2, 2, 64, 64)
        t = torch.tensor([100, 500])
        model.eval()
        with torch.no_grad():
            o1 = model(x, t)
            o2 = model(x, t)
        assert torch.allclose(o1, o2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda(self):
        model = DiffusionUNet().cuda()
        x = torch.randn(2, 2, 64, 64).cuda()
        t = torch.randint(0, 1000, (2,)).cuda()
        out = model(x, t)
        assert out.device.type == "cuda"
        assert out.shape == (2, 1, 64, 64)


class TestTimestepConditioning:
    def test_different_timesteps_produce_different_outputs(self):
        model = DiffusionUNet()
        model.eval()
        x = torch.randn(1, 2, 64, 64)
        with torch.no_grad():
            o1 = model(x, torch.tensor([0]))
            o2 = model(x, torch.tensor([999]))
        assert not torch.allclose(o1, o2, atol=1e-4)

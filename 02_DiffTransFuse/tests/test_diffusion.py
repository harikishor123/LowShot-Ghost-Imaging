"""Tests for src/diffusion.py — GaussianDiffusion schedule, forward, DDIM."""

import pytest
import torch
import math
import sys
sys.path.insert(0, ".")
from src.diffusion import GaussianDiffusion


@pytest.fixture
def diff():
    return GaussianDiffusion(timesteps=1000, beta_start=1e-4, beta_end=2e-2)


class TestSchedule:
    def test_alpha_cumprod_range(self, diff):
        assert diff.alpha_cumprod[0].item() == pytest.approx(0.9999, abs=1e-4)
        assert diff.alpha_cumprod[999].item() < 0.001

    def test_alpha_cumprod_monotonic(self, diff):
        deltas = diff.alpha_cumprod[1:] - diff.alpha_cumprod[:-1]
        assert (deltas < 0).all(), "alpha_cumprod must be strictly decreasing"

    def test_sqrt_values(self, diff):
        for i in [0, 100, 500, 999]:
            expected_sqrt = math.sqrt(diff.alpha_cumprod[i].item())
            assert diff.sqrt_alpha_cumprod[i].item() == pytest.approx(expected_sqrt, abs=1e-5)

    def test_schedule_length(self, diff):
        assert len(diff.alpha_cumprod) == 1000
        assert diff.T == 1000


class TestForwardProcess:
    def test_q_sample_shape(self, diff):
        x0 = torch.randn(4, 1, 64, 64)
        noise = torch.randn_like(x0)
        t = torch.full((4,), 500, dtype=torch.long)
        x_t = diff.q_sample(x0, t, noise)
        assert x_t.shape == (4, 1, 64, 64)

    def test_q_sample_at_t0(self, diff):
        x0 = torch.randn(2, 1, 32, 32)
        noise = torch.randn_like(x0)
        t = torch.zeros(2, dtype=torch.long)
        x_t = diff.q_sample(x0, t, noise)
        assert torch.allclose(x_t, x0, atol=0.05)

    def test_q_sample_at_t999(self, diff):
        x0 = torch.randn(2, 1, 32, 32)
        noise = torch.randn_like(x0)
        t = torch.full((2,), 999, dtype=torch.long)
        x_t = diff.q_sample(x0, t, noise)
        assert torch.allclose(x_t, noise, atol=0.05)

    def test_q_sample_different_timesteps(self, diff):
        x0 = torch.randn(4, 1, 32, 32)
        noise = torch.randn_like(x0)
        t = torch.tensor([0, 100, 500, 999])
        x_t = diff.q_sample(x0, t, noise)
        assert x_t.shape == x0.shape


class TestPredictX0:
    def test_recover_x0_with_true_noise(self, diff):
        torch.manual_seed(0)
        x0 = torch.randn(4, 1, 64, 64)
        noise = torch.randn_like(x0)
        for t_val in [0, 50, 500, 999]:
            t = torch.full((4,), t_val, dtype=torch.long)
            x_t = diff.q_sample(x0, t, noise)
            x0_pred = diff.predict_x0_from_eps(x_t, t, noise)
            assert torch.allclose(x0_pred, x0, atol=1e-3), f"Failed at t={t_val}"

    def test_x0_requires_no_clamp_during_training(self, diff):
        x0 = torch.randn(2, 1, 32, 32)
        noise = torch.randn_like(x0)
        t = torch.full((2,), 999, dtype=torch.long)
        x_t = diff.q_sample(x0, t, noise)
        x0_pred = diff.predict_x0_from_eps(x_t, t, noise)
        assert not torch.isnan(x0_pred).any()
        assert not torch.isinf(x0_pred).any()


class TestDDIMSchedule:
    def test_schedule_length(self, diff):
        for n in [5, 10, 25, 50, 100]:
            sched = diff.ddim_timestep_schedule(n)
            assert len(sched) <= n
            assert len(sched) > 0

    def test_schedule_descending(self, diff):
        sched = diff.ddim_timestep_schedule(50)
        for i in range(len(sched) - 1):
            assert sched[i] > sched[i + 1], f"Not descending at {i}: {sched[i]} <= {sched[i+1]}"

    def test_schedule_starts_at_T_minus_1(self, diff):
        sched = diff.ddim_timestep_schedule(100)
        assert sched[0] == 999

    def test_schedule_no_duplicates(self, diff):
        for n in [5, 25, 100, 250]:
            sched = diff.ddim_timestep_schedule(n)
            assert len(sched) == len(set(sched))


class TestDDIMStep:
    def test_step_with_true_noise(self, diff):
        torch.manual_seed(42)
        x0 = torch.randn(2, 1, 32, 32)
        noise = torch.randn_like(x0)
        t = torch.full((2,), 500, dtype=torch.long)
        x_t = diff.q_sample(x0, t, noise)
        x_next = diff.ddim_step(x_t, noise, 500, 499)
        t_prev = torch.full((2,), 499, dtype=torch.long)
        x_prev_true = diff.q_sample(x0, t_prev, noise)
        assert torch.allclose(x_next, x_prev_true, atol=1e-5)

    def test_final_step_returns_x0(self, diff):
        torch.manual_seed(42)
        x0 = torch.randn(2, 1, 32, 32)
        noise = torch.randn_like(x0)
        t = torch.full((2,), 0, dtype=torch.long)
        x_t = diff.q_sample(x0, t, noise)
        x_clean = diff.ddim_step(x_t, noise, 0, -1)
        assert torch.allclose(x_clean, x0, atol=1e-4)


class TestDDIMSample:
    def test_output_shape(self, diff):
        class DummyModel(torch.nn.Module):
            def forward(self, x, t):
                return x[:, :1]  # return first channel only

        model = DummyModel()
        cond = torch.randn(2, 1, 64, 64)
        result = diff.ddim_sample(model, cond, num_steps=5)
        assert result.shape == (2, 1, 64, 64)

    def test_deterministic(self, diff):
        class DummyModel(torch.nn.Module):
            def forward(self, x, t):
                return x[:, :1]

        model = DummyModel()
        cond = torch.randn(1, 1, 32, 32)
        torch.manual_seed(42)
        r1 = diff.ddim_sample(model, cond, num_steps=10)
        torch.manual_seed(42)
        r2 = diff.ddim_sample(model, cond, num_steps=10)
        assert torch.allclose(r1, r2)


class TestDeviceTransfer:
    def test_to_device(self, diff):
        diff2 = GaussianDiffusion(timesteps=100, device="cpu")
        diff2.to(torch.device("cpu"))
        assert diff2.alpha_cumprod.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_to_cuda(self):
        diff = GaussianDiffusion(timesteps=100)
        diff.to(torch.device("cuda"))
        assert diff.alpha_cumprod.device.type == "cuda"

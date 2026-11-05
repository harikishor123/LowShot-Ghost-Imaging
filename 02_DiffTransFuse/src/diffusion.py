"""
Gaussian diffusion process for residual ghost-image reconstruction (E3).

Adapted from DiffTransFuse Stage 1 GaussianDiffusion.  Key design choices:
  - LINEAR beta schedule: linspace(1e-4, 2e-2, T=1000)
  - ε-prediction: model predicts noise, not x0
  - DDIM sampler: deterministic (η=0), supports 25/50/100/250 steps
  - Residual formulation: r₀ = O − DGI

Residual statistics (measured from pool_train.npy, k=0):
  N=500:  mean=−0.014, std=0.365, range=[−1.0, 0.95]
  N=1000: mean=−0.009, std=0.308, range=[−1.0, 0.94]
  N=2000: mean=−0.005, std=0.247, range=[−0.99, 0.94]
  N=5000: mean=−0.003, std=0.172, range=[−0.97, 0.93]

The residual is approximately zero-mean (bias < 0.014) with moderate spread
(std 0.17–0.37 depending on shot count).  No normalisation is applied:
  - The mean is negligible relative to the std.
  - The range [−1, 1] is compatible with the linear schedule.
  - The std at high shot counts (0.17) is small enough that the schedule's
    β_end=0.02 doesn't destroy signal too early.

For N=500 (std=0.365), the signal-to-noise ratio at t=500 is:
  √(ᾱ_500)/√(1−ᾱ_500) ≈ 0.14, so SNR × std ≈ 0.05 — the signal is nearly
  gone by the midpoint.  This is acceptable for the linear schedule.
"""

from __future__ import annotations

import torch
import numpy as np


class GaussianDiffusion:
    """Linear-schedule Gaussian diffusion for ε-prediction training + DDIM sampling."""

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str | torch.device = "cpu",
    ) -> None:
        self.T = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        # Store as float32 for GPU efficiency, but compute in float64
        self.betas = betas.float().to(device)
        self.alphas = alphas.float().to(device)
        self.alpha_cumprod = alpha_cumprod.float().to(device)
        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod).float().to(device)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod).float().to(device)

    def to(self, device: torch.device) -> "GaussianDiffusion":
        """Move all schedule tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        self.sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(device)
        self.sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod.to(device)
        return self

    # ──────────────────────────────────────────────────────────
    # Forward process
    # ──────────────────────────────────────────────────────────

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion: x_t = √(ᾱ_t) · x0 + √(1−ᾱ_t) · ε

        Args:
            x0:    [B, C, H, W] clean signal (the residual r₀)
            t:     [B] integer timesteps in [0, T-1]
            noise: [B, C, H, W] ~ N(0, I)
        Returns:
            x_t:   [B, C, H, W] noisy signal
        """
        a = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return a * x0 + b * noise

    # ──────────────────────────────────────────────────────────
    # x0 derivation from ε-prediction
    # ──────────────────────────────────────────────────────────

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Derive x̂₀ from ε-prediction:
            x̂₀ = (x_t − √(1−ᾱ_t) · ε̂) / √(ᾱ_t)

        Note: at t close to T-1, √(ᾱ_t) is very small (~0.01), so x̂₀ can
        have large magnitude.  This is expected — the estimate is noisy at
        large t.  Callers should NOT clamp x̂₀ during training (it would
        bias the gradient).  Clamp only the final reconstruction O_pred.
        """
        a = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return (x_t - b * eps_pred) / a

    # ──────────────────────────────────────────────────────────
    # DDIM sampling (deterministic, η=0)
    # ──────────────────────────────────────────────────────────

    def ddim_timestep_schedule(self, num_steps: int) -> list[int]:
        """
        Build a DDIM sub-sequence of `num_steps` timesteps from [T-1, ..., 0].

        Returns a list of length `num_steps` in DESCENDING order.
        Example for T=1000, num_steps=5: [999, 799, 599, 399, 199]
        The final step transitions to t_prev = −1 (meaning t=0 / clean).
        """
        step_size = self.T / num_steps
        # Evenly spaced from T-1 down to near 0
        steps = [int(round((self.T - 1) - i * step_size)) for i in range(num_steps)]
        # Ensure the last step is 0
        steps[-1] = max(steps[-1], 0)
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for s in steps:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    def ddim_step(
        self,
        x_t: torch.Tensor,
        eps_pred: torch.Tensor,
        t_now: int,
        t_prev: int,
    ) -> torch.Tensor:
        """
        Single deterministic DDIM step from t_now to t_prev (η=0).

        DDIM equation:
            x̂₀ = (x_t − √(1−ᾱ_t) · ε̂) / √(ᾱ_t)
            x_{t_prev} = √(ᾱ_{t_prev}) · x̂₀ + √(1−ᾱ_{t_prev}) · ε̂

        When t_prev < 0 (i.e., transitioning to clean signal), returns x̂₀
        directly (since ᾱ₀ → 1 and the noise term vanishes).
        """
        abar_t = self.alpha_cumprod[t_now]
        sqrt_abar_t = torch.sqrt(abar_t)
        sqrt_1m_abar_t = torch.sqrt(1.0 - abar_t)

        # Predict clean signal
        x0_pred = (x_t - sqrt_1m_abar_t * eps_pred) / sqrt_abar_t

        if t_prev < 0:
            return x0_pred

        abar_s = self.alpha_cumprod[t_prev]
        sqrt_abar_s = torch.sqrt(abar_s)
        sqrt_1m_abar_s = torch.sqrt(1.0 - abar_s)

        return sqrt_abar_s * x0_pred + sqrt_1m_abar_s * eps_pred

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        condition: torch.Tensor,
        num_steps: int = 100,
        shape: tuple | None = None,
    ) -> torch.Tensor:
        """
        Full DDIM sampling loop.

        Args:
            model:     DiffusionUNet, takes (concat(x_t, cond), timestep) → ε̂
            condition: [B, 1, H, W] DGI conditioning image
            num_steps: number of DDIM sampling steps (25/50/100/250)
            shape:     optional (B, C, H, W) for x_T; defaults to condition's shape

        Returns:
            x0_pred:   [B, 1, H, W] denoised residual estimate
        """
        device = condition.device
        B = condition.shape[0]
        if shape is None:
            shape = condition.shape

        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        schedule = self.ddim_timestep_schedule(num_steps)

        for i in range(len(schedule)):
            t_now = schedule[i]
            t_prev = schedule[i + 1] if i + 1 < len(schedule) else -1

            # Build timestep tensor
            t_tensor = torch.full((B,), t_now, device=device, dtype=torch.long)

            # Model input: concat(noisy_residual, DGI)
            model_input = torch.cat([x_t, condition], dim=1)
            eps_pred = model(model_input, t_tensor)

            x_t = self.ddim_step(x_t, eps_pred, t_now, t_prev)

        return x_t  # this is x̂₀ after the final step

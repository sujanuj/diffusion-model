"""DDPM noise schedule.

The noise schedule defines how much noise is added at each of the T
timesteps in the forward diffusion process.

Forward process:
  q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t)*x_{t-1}, beta_t*I)

Which means at timestep t, the noisy image is:
  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
  where epsilon ~ N(0, I)

This closed-form lets us jump directly from x_0 to any x_t without
simulating all intermediate steps -- critical for efficient training.

Two schedules:
  - Linear: beta increases linearly from beta_start to beta_end
  - Cosine: smoother schedule that preserves more signal at early steps
    (from Nichol & Dhariwal 2021, "Improved DDPM")
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class NoiseSchedule:
    """DDPM noise schedule with precomputed coefficients.

    Args:
        T: number of diffusion timesteps (default 1000)
        schedule: 'linear' or 'cosine'
        beta_start: starting noise level (linear schedule)
        beta_end: ending noise level (linear schedule)
        device: torch device
    """

    def __init__(
        self,
        T: int = 1000,
        schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        self.T = T
        self.device = device

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, T)
        elif schedule == "cosine":
            # Nichol & Dhariwal 2021 cosine schedule
            steps = T + 1
            t = torch.linspace(0, T, steps) / T
            alpha_bar = torch.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # Precompute all coefficients used in training and sampling
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.tensor([1.0]), alpha_bar[:-1]])

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bar = alpha_bar.to(device)
        self.alpha_bar_prev = alpha_bar_prev.to(device)

        # sqrt(alpha_bar): scale factor for x_0 in forward process
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar).to(device)

        # sqrt(1 - alpha_bar): scale factor for noise in forward process
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar).to(device)

        # 1 / sqrt(alpha): used in reverse process
        self.sqrt_recip_alpha = torch.sqrt(1.0 / alphas).to(device)

        # Posterior variance for reverse process
        self.posterior_variance = (
            betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        ).to(device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Forward diffusion: sample x_t given x_0 and timestep t.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        Args:
            x0: clean image tensor (B, C, H, W), values in [-1, 1]
            t: timestep indices (B,), integers in [0, T-1]
            noise: optional pre-sampled noise (B, C, H, W)

        Returns:
            x_t: noisy image at timestep t
            noise: the noise that was added (needed for training loss)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Extract coefficients for each sample's timestep
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]

        x_t = sqrt_ab * x0 + sqrt_one_minus_ab * noise
        return x_t, noise

    def predict_x0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor,
                               predicted_noise: torch.Tensor) -> torch.Tensor:
        """Recover x_0 estimate from predicted noise.

        Inverse of q_sample:
          x_0 = (x_t - sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        """
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return (x_t - sqrt_one_minus_ab * predicted_noise) / sqrt_ab

    def visualize_forward_process(self, x0: torch.Tensor, save_path: str = None):
        """Visualize how noise is added across timesteps."""
        timesteps = [0, 100, 200, 400, 600, 800, 999]
        fig, axes = plt.subplots(1, len(timesteps), figsize=(14, 2))

        noise = torch.randn_like(x0.unsqueeze(0))
        for i, t_val in enumerate(timesteps):
            t = torch.tensor([t_val])
            x_t, _ = self.q_sample(x0.unsqueeze(0), t, noise)
            img = x_t[0].permute(1, 2, 0).clamp(-1, 1).numpy()
            img = (img + 1) / 2  # [-1,1] -> [0,1]
            if img.shape[2] == 1:
                img = img[:, :, 0]
                axes[i].imshow(img, cmap='gray')
            else:
                axes[i].imshow(img)
            axes[i].set_title(f"t={t_val}", fontsize=8)
            axes[i].axis('off')

        plt.suptitle("Forward diffusion process: clean → noise", fontsize=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved to {save_path}")
        plt.show()
        return fig

    def visualize_schedule(self, save_path: str = None):
        """Plot the noise schedule coefficients."""
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        t = np.arange(self.T)

        axes[0].plot(t, self.betas.cpu().numpy())
        axes[0].set_title("Beta (noise level)")
        axes[0].set_xlabel("Timestep")

        axes[1].plot(t, self.alpha_bar.cpu().numpy())
        axes[1].set_title("Alpha bar (signal)")
        axes[1].set_xlabel("Timestep")

        axes[2].plot(t, self.sqrt_one_minus_alpha_bar.cpu().numpy())
        axes[2].set_title("Sqrt(1 - alpha_bar) (noise scale)")
        axes[2].set_xlabel("Timestep")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        return fig

"""DDPM training loop and reverse diffusion sampler.

Training:
  For each batch:
    1. Sample random timesteps t ~ Uniform(0, T)
    2. Sample noise epsilon ~ N(0, I)
    3. Compute x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
    4. Predict epsilon_hat = U-Net(x_t, t)
    5. Loss = MSE(epsilon_hat, epsilon)

The model never sees clean images directly during training -- it only
sees noisy images and learns to predict the noise that was added.

Sampling (reverse diffusion):
  Start from x_T ~ N(0, I), then for t = T-1, T-2, ..., 0:
    1. Predict noise: epsilon_hat = U-Net(x_t, t)
    2. Compute x_{t-1} using the DDPM reverse formula:
       x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_hat)
                 + sqrt(posterior_variance_t) * z   where z ~ N(0,I) if t > 0
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from model.noise_schedule import NoiseSchedule
from model.unet import UNet


class DDPMTrainer:
    """Training loop for DDPM.

    Args:
        model: U-Net denoising network
        schedule: NoiseSchedule instance
        lr: learning rate (default 2e-4, standard for DDPM)
        device: 'cpu' or 'cuda' or 'mps'
    """

    def __init__(
        self,
        model: UNet,
        schedule: NoiseSchedule,
        lr: float = 2e-4,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.schedule = schedule
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.losses: List[float] = []
        self.step = 0

    def train_step(self, x0: torch.Tensor) -> float:
        """One training step on a batch of clean images.

        Args:
            x0: clean images (B, C, H, W), values in [-1, 1]

        Returns:
            loss value (float)
        """
        x0 = x0.to(self.device)
        B = x0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.schedule.T, (B,), device=self.device)

        # Sample noise and create noisy images
        noise = torch.randn_like(x0)
        x_t, _ = self.schedule.q_sample(x0, t, noise)

        # Predict noise
        self.optimizer.zero_grad()
        predicted_noise = self.model(x_t, t)

        # MSE loss between predicted and actual noise
        loss = F.mse_loss(predicted_noise, noise)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.step += 1
        loss_val = float(loss.item())
        self.losses.append(loss_val)
        return loss_val

    def train(
        self,
        dataloader: DataLoader,
        epochs: int,
        log_interval: int = 100,
        save_dir: Optional[str] = None,
        sample_interval: int = 500,
    ) -> dict:
        """Full training loop.

        Args:
            dataloader: DataLoader yielding (images, labels) batches
            epochs: number of epochs to train
            log_interval: print loss every N steps
            save_dir: directory to save checkpoints and samples
            sample_interval: generate sample images every N steps

        Returns:
            dict with losses and training stats
        """
        self.model.train()
        t0 = time.time()

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            for batch_idx, (x0, _) in enumerate(dataloader):
                # Normalize to [-1, 1]
                x0 = x0 * 2 - 1  # assumes input in [0, 1]

                loss = self.train_step(x0)

                if self.step % log_interval == 0:
                    elapsed = time.time() - t0
                    avg_loss = np.mean(self.losses[-log_interval:])
                    print(f"epoch {epoch+1}/{epochs} | step {self.step} | "
                          f"loss={avg_loss:.4f} | elapsed={elapsed:.1f}s")

                if save_dir and self.step % sample_interval == 0:
                    self.model.eval()
                    samples = DDPMSampler(self.model, self.schedule,
                                         self.device).sample(n=16)
                    save_image_grid(samples,
                                    f"{save_dir}/samples_step{self.step}.png")
                    self.model.train()

        total_time = time.time() - t0
        print(f"\nTraining complete: {self.step} steps in {total_time:.1f}s")
        return {
            "losses": self.losses,
            "total_steps": self.step,
            "total_time_s": total_time,
        }

    def save(self, path: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": self.step,
            "losses": self.losses,
        }, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.step = ckpt["step"]
        self.losses = ckpt["losses"]
        print(f"Loaded checkpoint from {path} (step {self.step})")


class DDPMSampler:
    """DDPM reverse diffusion sampler.

    Generates images by starting from pure noise and iteratively
    denoising using the trained U-Net.

    Args:
        model: trained U-Net
        schedule: NoiseSchedule (must match training schedule)
        device: torch device
    """

    def __init__(self, model: UNet, schedule: NoiseSchedule, device: str = "cpu"):
        self.model = model.to(device)
        self.schedule = schedule
        self.device = device

    @torch.no_grad()
    def sample(
        self,
        n: int = 16,
        image_size: tuple = (28, 28),
        in_channels: int = 1,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Generate n images from pure noise.

        Args:
            n: number of images to generate
            image_size: (H, W) of the output images
            in_channels: number of image channels
            show_progress: print denoising progress

        Returns:
            Tensor of shape (n, in_channels, H, W), values in [0, 1]
        """
        self.model.eval()
        H, W = image_size

        # Start from pure noise
        x = torch.randn(n, in_channels, H, W, device=self.device)

        # Reverse diffusion: T-1 -> T-2 -> ... -> 0
        for t_val in reversed(range(self.schedule.T)):
            if show_progress and t_val % 100 == 0:
                print(f"  Sampling t={t_val}...")

            t = torch.full((n,), t_val, device=self.device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.model(x, t)

            # DDPM reverse step
            beta_t = self.schedule.betas[t_val]
            sqrt_recip_alpha_t = self.schedule.sqrt_recip_alpha[t_val]
            sqrt_one_minus_ab_t = self.schedule.sqrt_one_minus_alpha_bar[t_val]

            # Mean of reverse distribution
            mean = sqrt_recip_alpha_t * (
                x - beta_t / sqrt_one_minus_ab_t * predicted_noise
            )

            if t_val > 0:
                # Add noise (not at last step)
                posterior_var = self.schedule.posterior_variance[t_val]
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(posterior_var) * noise
            else:
                x = mean

        # Clamp and rescale from [-1, 1] to [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        return x

    @torch.no_grad()
    def sample_progressive(
        self,
        n: int = 4,
        image_size: tuple = (28, 28),
        in_channels: int = 1,
        save_every: int = 100,
    ) -> List[torch.Tensor]:
        """Generate images and save intermediate denoising steps.

        Returns list of tensors at each save_every timestep -- used to
        create the animated denoising visualization.
        """
        self.model.eval()
        H, W = image_size
        x = torch.randn(n, in_channels, H, W, device=self.device)
        frames = []

        for t_val in reversed(range(self.schedule.T)):
            t = torch.full((n,), t_val, device=self.device, dtype=torch.long)
            predicted_noise = self.model(x, t)

            beta_t = self.schedule.betas[t_val]
            sqrt_recip_alpha_t = self.schedule.sqrt_recip_alpha[t_val]
            sqrt_one_minus_ab_t = self.schedule.sqrt_one_minus_alpha_bar[t_val]

            mean = sqrt_recip_alpha_t * (
                x - beta_t / sqrt_one_minus_ab_t * predicted_noise
            )

            if t_val > 0:
                posterior_var = self.schedule.posterior_variance[t_val]
                x = mean + torch.sqrt(posterior_var) * torch.randn_like(x)
            else:
                x = mean

            if t_val % save_every == 0:
                frames.append((x.clamp(-1, 1) + 1) / 2)

        return frames


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 4):
    """Save a grid of images to disk.

    Args:
        images: (N, C, H, W) tensor, values in [0, 1]
        path: output file path
        nrow: images per row
    """
    try:
        from torchvision.utils import save_image
        save_image(images, path, nrow=nrow, normalize=False)
    except ImportError:
        # Fallback: save with PIL
        import numpy as np
        from PIL import Image

        imgs = images.cpu().numpy()
        N, C, H, W = imgs.shape
        ncols = nrow
        nrows = (N + ncols - 1) // ncols
        grid = np.ones((nrows * H, ncols * W, max(C, 3)), dtype=np.uint8) * 255

        for i, img in enumerate(imgs):
            r, c = i // ncols, i % ncols
            img_np = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            if C == 1:
                img_np = np.repeat(img_np, 3, axis=2)
            grid[r*H:(r+1)*H, c*W:(c+1)*W] = img_np

        Image.fromarray(grid).save(path)
    print(f"Saved image grid to {path}")

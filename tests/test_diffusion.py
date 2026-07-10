"""Tests for the DDPM trainer and sampler."""

import sys
from pathlib import Path
import torch
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.noise_schedule import NoiseSchedule
from model.unet import UNet
from model.diffusion import DDPMTrainer, DDPMSampler, save_image_grid


def tiny_model():
    return UNet(in_channels=1, base_channels=8, channel_mults=(1, 2),
                time_embed_dim=32, num_groups=4)


def tiny_schedule():
    return NoiseSchedule(T=100, schedule="linear")


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------

def test_trainer_train_step_returns_float():
    model = tiny_model()
    schedule = tiny_schedule()
    trainer = DDPMTrainer(model, schedule, lr=1e-3)
    x0 = torch.rand(2, 1, 28, 28)  # [0,1] range, trainer normalizes
    x0 = x0 * 2 - 1  # normalize to [-1, 1]
    loss = trainer.train_step(x0)
    assert isinstance(loss, float)
    assert loss > 0
    assert not np.isnan(loss)


def test_trainer_loss_decreases_on_memorization():
    """Train on a single batch for many steps -- loss should decrease."""
    torch.manual_seed(42)
    model = tiny_model()
    schedule = tiny_schedule()
    trainer = DDPMTrainer(model, schedule, lr=1e-3)

    x0 = torch.rand(4, 1, 28, 28) * 2 - 1
    losses = []
    for _ in range(50):
        loss = trainer.train_step(x0)
        losses.append(loss)

    first_avg = np.mean(losses[:5])
    last_avg = np.mean(losses[-5:])
    assert last_avg < first_avg, \
        f"Loss did not decrease: first={first_avg:.4f}, last={last_avg:.4f}"


def test_trainer_step_counter_increments():
    model = tiny_model()
    schedule = tiny_schedule()
    trainer = DDPMTrainer(model, schedule)
    x0 = torch.rand(2, 1, 28, 28) * 2 - 1
    assert trainer.step == 0
    trainer.train_step(x0)
    assert trainer.step == 1
    trainer.train_step(x0)
    assert trainer.step == 2


def test_trainer_losses_recorded():
    model = tiny_model()
    schedule = tiny_schedule()
    trainer = DDPMTrainer(model, schedule)
    x0 = torch.rand(2, 1, 28, 28) * 2 - 1
    for _ in range(5):
        trainer.train_step(x0)
    assert len(trainer.losses) == 5


def test_trainer_save_load(tmp_path):
    model = tiny_model()
    schedule = tiny_schedule()
    trainer = DDPMTrainer(model, schedule)
    x0 = torch.rand(2, 1, 28, 28) * 2 - 1
    trainer.train_step(x0)
    initial_loss = trainer.losses[0]

    path = str(tmp_path / "ckpt.pt")
    trainer.save(path)

    model2 = tiny_model()
    trainer2 = DDPMTrainer(model2, schedule)
    trainer2.load(path)

    assert trainer2.step == 1
    assert abs(trainer2.losses[0] - initial_loss) < 1e-6


# ---------------------------------------------------------------------------
# Sampler tests
# ---------------------------------------------------------------------------

def test_sampler_output_shape():
    model = tiny_model()
    schedule = tiny_schedule()
    sampler = DDPMSampler(model, schedule)
    samples = sampler.sample(n=4, image_size=(28, 28), in_channels=1)
    assert samples.shape == (4, 1, 28, 28)


def test_sampler_output_in_zero_one_range():
    model = tiny_model()
    schedule = tiny_schedule()
    sampler = DDPMSampler(model, schedule)
    samples = sampler.sample(n=4, image_size=(28, 28), in_channels=1)
    assert samples.min() >= 0.0, f"Min value {samples.min()} < 0"
    assert samples.max() <= 1.0, f"Max value {samples.max()} > 1"


def test_sampler_output_no_nan():
    model = tiny_model()
    schedule = tiny_schedule()
    sampler = DDPMSampler(model, schedule)
    samples = sampler.sample(n=2, image_size=(28, 28), in_channels=1)
    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()


def test_sampler_different_samples_each_call():
    """Each sampling call should produce different images (stochastic)."""
    torch.manual_seed(0)
    model = tiny_model()
    schedule = tiny_schedule()
    sampler = DDPMSampler(model, schedule)
    s1 = sampler.sample(n=2, image_size=(28, 28))
    s2 = sampler.sample(n=2, image_size=(28, 28))
    assert not torch.allclose(s1, s2), "Sampler should produce different images each call"


def test_sampler_progressive_returns_frames():
    model = tiny_model()
    schedule = tiny_schedule()
    sampler = DDPMSampler(model, schedule)
    frames = sampler.sample_progressive(n=2, image_size=(28, 28), save_every=25)
    assert len(frames) > 0
    for frame in frames:
        assert frame.shape == (2, 1, 28, 28)


def test_save_image_grid(tmp_path):
    images = torch.rand(4, 1, 28, 28)
    path = str(tmp_path / "grid.png")
    save_image_grid(images, path)
    assert Path(path).exists()

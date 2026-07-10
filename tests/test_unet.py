"""Tests for the U-Net denoising network."""

import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.unet import UNet, SinusoidalTimeEmbedding, TimeEmbedding, ResidualBlock


# ---------------------------------------------------------------------------
# Time embedding tests
# ---------------------------------------------------------------------------

def test_sinusoidal_embedding_shape():
    emb = SinusoidalTimeEmbedding(128)
    t = torch.randint(0, 1000, (4,))
    out = emb(t)
    assert out.shape == (4, 128)


def test_sinusoidal_embedding_different_timesteps():
    emb = SinusoidalTimeEmbedding(64)
    t1 = torch.tensor([0])
    t2 = torch.tensor([500])
    out1 = emb(t1)
    out2 = emb(t2)
    assert not torch.allclose(out1, out2), "Different timesteps should have different embeddings"


def test_time_embedding_shape():
    te = TimeEmbedding(128)
    t = torch.randint(0, 1000, (4,))
    out = te(t)
    assert out.shape == (4, 128)


# ---------------------------------------------------------------------------
# Residual block tests
# ---------------------------------------------------------------------------

def test_residual_block_same_channels():
    block = ResidualBlock(32, 32, time_dim=128)
    x = torch.randn(2, 32, 16, 16)
    t = torch.randn(2, 128)
    out = block(x, t)
    assert out.shape == (2, 32, 16, 16)


def test_residual_block_different_channels():
    block = ResidualBlock(32, 64, time_dim=128)
    x = torch.randn(2, 32, 16, 16)
    t = torch.randn(2, 128)
    out = block(x, t)
    assert out.shape == (2, 64, 16, 16)


# ---------------------------------------------------------------------------
# Full U-Net tests
# ---------------------------------------------------------------------------

def test_unet_output_shape_mnist():
    """U-Net output should match input shape for MNIST (1, 28, 28)."""
    model = UNet(in_channels=1, base_channels=16, channel_mults=(1, 2),
                 time_embed_dim=64, num_groups=4)
    x = torch.randn(2, 1, 28, 28)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    assert out.shape == (2, 1, 28, 28), f"Expected (2,1,28,28), got {out.shape}"


def test_unet_output_shape_cifar():
    """U-Net output should match input shape for CIFAR-10 (3, 32, 32)."""
    model = UNet(in_channels=3, base_channels=16, channel_mults=(1, 2, 4),
                 time_embed_dim=64, num_groups=4)
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    assert out.shape == (2, 3, 32, 32)


def test_unet_output_no_nan():
    model = UNet(in_channels=1, base_channels=16, channel_mults=(1, 2),
                 time_embed_dim=64, num_groups=4)
    x = torch.randn(2, 1, 28, 28)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    assert not torch.isnan(out).any(), "U-Net output contains NaN"
    assert not torch.isinf(out).any(), "U-Net output contains Inf"


def test_unet_different_timesteps_give_different_outputs():
    """The time conditioning must actually affect the output."""
    model = UNet(in_channels=1, base_channels=16, channel_mults=(1, 2),
                 time_embed_dim=64, num_groups=4)
    x = torch.randn(1, 1, 28, 28)
    t1 = torch.tensor([0])
    t2 = torch.tensor([999])
    out1 = model(x, t1)
    out2 = model(x, t2)
    assert not torch.allclose(out1, out2), \
        "Same input with different timesteps should give different outputs"


def test_unet_gradient_flows():
    model = UNet(in_channels=1, base_channels=16, channel_mults=(1, 2),
                 time_embed_dim=64, num_groups=4)
    x = torch.randn(2, 1, 28, 28)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    loss = out.mean()
    loss.backward()
    # Check that at least some parameters have gradients
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_unet_parameter_count():
    model = UNet(in_channels=1, base_channels=32, channel_mults=(1, 2, 4),
                 time_embed_dim=128, num_groups=8)
    n = model.num_parameters()
    assert n > 0
    print(f"\nU-Net parameters: {n:,}")


def test_unet_batch_size_one():
    model = UNet(in_channels=1, base_channels=16, channel_mults=(1, 2),
                 time_embed_dim=64, num_groups=4)
    x = torch.randn(1, 1, 28, 28)
    t = torch.tensor([500])
    out = model(x, t)
    assert out.shape == (1, 1, 28, 28)

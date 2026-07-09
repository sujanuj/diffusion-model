"""Tests for the DDPM noise schedule."""

import sys
from pathlib import Path
import torch
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.noise_schedule import NoiseSchedule


def test_linear_schedule_shape():
    ns = NoiseSchedule(T=1000, schedule="linear")
    assert ns.betas.shape == (1000,)
    assert ns.alpha_bar.shape == (1000,)


def test_cosine_schedule_shape():
    ns = NoiseSchedule(T=1000, schedule="cosine")
    assert ns.betas.shape == (1000,)


def test_betas_in_valid_range():
    ns = NoiseSchedule(T=1000, schedule="linear")
    assert (ns.betas >= 0).all()
    assert (ns.betas <= 1).all()


def test_alpha_bar_decreasing():
    """Alpha bar should decrease monotonically -- signal decreases over time."""
    ns = NoiseSchedule(T=1000)
    diffs = ns.alpha_bar[1:] - ns.alpha_bar[:-1]
    assert (diffs < 0).all(), "alpha_bar should be strictly decreasing"


def test_alpha_bar_starts_near_one():
    """At t=0, very little noise -- alpha_bar should be close to 1."""
    ns = NoiseSchedule(T=1000)
    assert float(ns.alpha_bar[0]) > 0.99


def test_alpha_bar_ends_near_zero():
    """At t=T-1, pure noise -- alpha_bar should be close to 0."""
    ns = NoiseSchedule(T=1000)
    assert float(ns.alpha_bar[-1]) < 0.02


def test_q_sample_output_shape():
    ns = NoiseSchedule(T=1000)
    x0 = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 1000, (4,))
    x_t, noise = ns.q_sample(x0, t)
    assert x_t.shape == x0.shape
    assert noise.shape == x0.shape


def test_q_sample_at_t0_close_to_x0():
    """At t=0, almost no noise added -- x_t should be very close to x0."""
    ns = NoiseSchedule(T=1000)
    x0 = torch.randn(2, 1, 28, 28)
    t = torch.zeros(2, dtype=torch.long)
    noise = torch.zeros_like(x0)  # zero noise to isolate scale factor
    x_t, _ = ns.q_sample(x0, t, noise)
    # x_t = sqrt(alpha_bar[0]) * x0 ≈ 0.9999 * x0
    assert torch.allclose(x_t, ns.sqrt_alpha_bar[0] * x0, atol=1e-5)


def test_q_sample_at_T_close_to_noise():
    """At t=T-1, almost pure noise."""
    ns = NoiseSchedule(T=1000)
    x0 = torch.zeros(2, 1, 28, 28)  # black image
    t = torch.full((2,), 999, dtype=torch.long)
    noise = torch.ones_like(x0)
    x_t, _ = ns.q_sample(x0, t, noise)
    # x_t ≈ sqrt(1 - alpha_bar[999]) * noise ≈ noise since alpha_bar[999] ≈ 0
    expected = ns.sqrt_one_minus_alpha_bar[999] * noise
    assert torch.allclose(x_t, expected, atol=1e-5)


def test_predict_x0_from_noise_inverts_q_sample():
    """predict_x0_from_noise should recover x0 from x_t and the noise."""
    ns = NoiseSchedule(T=1000)
    x0 = torch.randn(2, 1, 28, 28)
    t = torch.randint(1, 999, (2,))
    noise = torch.randn_like(x0)
    x_t, _ = ns.q_sample(x0, t, noise)
    x0_recovered = ns.predict_x0_from_noise(x_t, t, noise)
    assert torch.allclose(x0, x0_recovered, atol=1e-5), \
        f"Max diff: {(x0 - x0_recovered).abs().max()}"


def test_cosine_schedule_alpha_bar_higher_than_linear_early():
    """Cosine schedule preserves more signal at early timesteps than linear."""
    linear = NoiseSchedule(T=1000, schedule="linear")
    cosine = NoiseSchedule(T=1000, schedule="cosine")
    # At t=200 (20% through), cosine should have higher alpha_bar (more signal)
    assert float(cosine.alpha_bar[200]) > float(linear.alpha_bar[200]),         "Cosine schedule should preserve more signal at early timesteps"


def test_unknown_schedule_raises():
    with pytest.raises(ValueError):
        NoiseSchedule(T=1000, schedule="unknown")

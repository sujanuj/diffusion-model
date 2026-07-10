# diffusion-model

A DDPM (Denoising Diffusion Probabilistic Model) built from scratch in PyTorch. Implements the full diffusion pipeline — noise schedule, U-Net denoising network, training loop, and reverse diffusion sampler — and trains on MNIST to generate handwritten digits.

---

## What it does

1. **Forward process**: adds Gaussian noise to a clean image over T=1000 steps until it becomes pure noise
2. **Trains a U-Net** to predict the noise added at each step
3. **Reverse process**: starts from pure noise and iteratively denoises over 1000 steps to generate new images
4. **Gradio demo**: interactive web UI to generate images in real time

---

## Phases

**Phase 1: Noise schedule — done**

- [x] `model/noise_schedule.py` — linear and cosine beta schedules, precomputed coefficients (`alpha_bar`, `sqrt_alpha_bar`, `posterior_variance`)
- [x] `q_sample()` — closed-form forward process: jump directly from x_0 to any x_t without simulating intermediate steps
- [x] `predict_x0_from_noise()` — inverse of q_sample, recovers clean image from noisy image and predicted noise
- [x] Cosine schedule verified to preserve more signal at early timesteps than linear (higher `alpha_bar` at t=200)
- [x] 12 tests: schedule shapes, monotonicity, boundary values, q_sample correctness, invertibility

**Phase 2: U-Net denoising network — done**

- [x] `model/unet.py` — encoder-decoder U-Net with skip connections and sinusoidal time embedding
- [x] `SinusoidalTimeEmbedding` — maps scalar timestep t to dense vector, same formula as transformer positional encodings
- [x] `ResidualBlock` — GroupNorm → SiLU → Conv → inject time embedding → GroupNorm → SiLU → Conv + skip
- [x] Time conditioning verified: same image at t=0 vs t=999 produces different predicted noise
- [x] Gradient flows through the full network
- [x] 12 tests: embedding shapes, residual blocks, output shapes for MNIST and CIFAR, NaN checks, gradient flow

**Phase 3: DDPM trainer and sampler — done**

- [x] `model/diffusion.py` — `DDPMTrainer` runs the training loop: sample t → add noise → predict noise → MSE loss → Adam step
- [x] `DDPMSampler` — reverse diffusion: start from x_T ~ N(0,I), denoise T steps using the DDPM formula
- [x] `sample_progressive()` — saves intermediate frames for denoising animation
- [x] Checkpoint save/load
- [x] Loss verified to decrease on memorization: 50 steps on a fixed batch, last 5 avg < first 5 avg
- [x] Sampler output verified in [0,1] range, no NaN, stochastic (different each call)
- [x] 11 tests

**Planned:**

- [ ] Phase 4: Train on MNIST, show generated digit images
- [ ] Phase 5: Gradio demo with live generation

---

## Running tests

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision numpy matplotlib pillow tqdm pytest
python -m pytest tests/ -v   # 35 tests
```

---

## Project layout

```
diffusion-model/
├── model/
│   ├── noise_schedule.py   <- linear/cosine beta schedule, q_sample (Phase 1)
│   ├── unet.py             <- U-Net with residual blocks, time embedding (Phase 2)
│   └── diffusion.py        <- DDPMTrainer, DDPMSampler, save/load (Phase 3)
├── data/                   <- MNIST dataset (downloaded automatically)
├── tests/                  <- 35 tests, all passing
└── train.py                <- training script (Phase 4, planned)
```

---

## Author

**Sujan Uppalli Jayadevappa**
MS Software Engineering — Arizona State University
GitHub: [sujanuj](https://github.com/sujanuj)

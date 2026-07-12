"""Gradio demo for the DDPM diffusion model."""

import numpy as np
import torch
from pathlib import Path
import gradio as gr
from PIL import Image

from model.noise_schedule import NoiseSchedule
from model.unet import UNet
from model.diffusion import DDPMSampler


def load_model(checkpoint_path: str = "data/checkpoint.pt"):
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    schedule = NoiseSchedule(T=1000, schedule="linear", device=device)
    model = UNet(in_channels=1, base_channels=64, channel_mults=(1, 2, 4),
                 time_embed_dim=256, num_groups=8)
    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint (step {ckpt.get('step','?')}) on {device}")
    else:
        print("No checkpoint found, using random weights")
    model = model.to(device).eval()
    return DDPMSampler(model, schedule, device=device), device


print("Loading model...")
sampler, device = load_model()
print(f"Ready on {device}")


def tensor_to_pil(t):
    arr = (t.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").resize((112, 112), Image.NEAREST)


def generate(n_images, n_steps, seed):
    if seed >= 0:
        torch.manual_seed(int(seed))
    T = sampler.schedule.T
    x = torch.randn(int(n_images), 1, 28, 28, device=device)
    step_size = max(1, T // int(n_steps))
    timesteps = list(range(0, T, step_size))[::-1]

    with torch.no_grad():
        for t_val in timesteps:
            t = torch.full((int(n_images),), t_val, device=device, dtype=torch.long)
            eps = sampler.model(x, t)
            beta = sampler.schedule.betas[t_val]
            recip_sqrt_a = sampler.schedule.sqrt_recip_alpha[t_val]
            sqrt_1mab = sampler.schedule.sqrt_one_minus_alpha_bar[t_val]
            mean = recip_sqrt_a * (x - beta / sqrt_1mab * eps)
            if t_val > 0:
                x = mean + torch.sqrt(sampler.schedule.posterior_variance[t_val]) * torch.randn_like(x)
            else:
                x = mean

    samples = (x.clamp(-1, 1) + 1) / 2
    return [tensor_to_pil(samples[i]) for i in range(int(n_images))]


def denoising_process(seed):
    if seed >= 0:
        torch.manual_seed(int(seed))
    save_at = {999, 800, 600, 400, 200, 100, 50, 10, 0}
    frames = []
    x = torch.randn(1, 1, 28, 28, device=device)

    with torch.no_grad():
        for t_val in reversed(range(sampler.schedule.T)):
            t = torch.full((1,), t_val, device=device, dtype=torch.long)
            eps = sampler.model(x, t)
            beta = sampler.schedule.betas[t_val]
            recip_sqrt_a = sampler.schedule.sqrt_recip_alpha[t_val]
            sqrt_1mab = sampler.schedule.sqrt_one_minus_alpha_bar[t_val]
            mean = recip_sqrt_a * (x - beta / sqrt_1mab * eps)
            if t_val > 0:
                x = mean + torch.sqrt(sampler.schedule.posterior_variance[t_val]) * torch.randn_like(x)
            else:
                x = mean
            if t_val in save_at:
                img = (x.clamp(-1, 1) + 1) / 2
                frames.append((tensor_to_pil(img[0]), f"t={t_val}"))
    return frames


with gr.Blocks(title="DDPM Diffusion Model", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # DDPM Diffusion Model — Handwritten Digit Generation
    Built from scratch in PyTorch. Generates digits by reversing a 1000-step noising process.
    **Model:** 8.9M parameter U-Net | **Trained on:** 60K MNIST images | **Loss:** 1.12 → 0.02
    """)

    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column(scale=1):
                n_imgs = gr.Slider(1, 16, value=8, step=1, label="Number of images")
                n_steps = gr.Slider(10, 1000, value=200, step=10,
                                    label="Denoising steps (fewer=faster)")
                seed = gr.Slider(-1, 1000, value=-1, step=1, label="Seed (-1=random)")
                btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Generated digits", columns=4, height=400)
        btn.click(fn=generate, inputs=[n_imgs, n_steps, seed], outputs=gallery)

    with gr.Tab("Denoising Process"):
        gr.Markdown("Watch a digit emerge from pure noise over 1000 steps.")
        dseed = gr.Slider(-1, 1000, value=42, step=1, label="Seed")
        dbtn = gr.Button("Show denoising", variant="primary")
        dgallery = gr.Gallery(label="t=999 (noise) → t=0 (digit)", columns=9, height=200)
        dbtn.click(fn=denoising_process, inputs=[dseed], outputs=dgallery)

    with gr.Tab("About"):
        gr.Markdown("""
        ## How DDPM works
        **Forward process:** Add noise to image over T=1000 steps until it becomes pure noise.
        Closed form: x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε

        **Training:** U-Net learns to predict the noise ε given noisy image x_t and timestep t.
        Loss = MSE(ε_predicted, ε_actual)

        **Sampling:** Start from x_T ~ N(0,I), run U-Net 1000 times to denoise step by step.

        **GitHub:** [sujanuj/diffusion-model](https://github.com/sujanuj/diffusion-model)
        """)

    demo.load(fn=lambda: generate(8, 200, 42), outputs=gallery)

if __name__ == "__main__":
    demo.launch(share=True)

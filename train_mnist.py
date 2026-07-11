"""Train DDPM on MNIST and generate handwritten digit images.

This is the payoff of Phases 1-3: train the full diffusion pipeline on
real data and generate new images from pure noise.

Model config (trains in ~20 min on CPU / ~5 min on MPS):
  dataset: MNIST (28x28 grayscale digits, 60K training images)
  U-Net: base_channels=64, channel_mults=(1,2,4), ~4M parameters
  T: 1000 diffusion steps
  batch_size: 128
  epochs: 10
  lr: 2e-4

Run:
  python train_mnist.py                    # full training
  python train_mnist.py --epochs 2         # quick test
  python train_mnist.py --sample-only      # generate from saved checkpoint
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.noise_schedule import NoiseSchedule
from model.unet import UNet
from model.diffusion import DDPMTrainer, DDPMSampler, save_image_grid


def get_mnist_loader(batch_size: int, data_dir: str = "data"):
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
    ])
    dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=0, drop_last=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=1000)
    parser.add_argument("--save-dir", type=str, default="data/outputs")
    parser.add_argument("--checkpoint", type=str, default="data/checkpoint.pt")
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--n-samples", type=int, default=64)
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Build model and schedule
    schedule = NoiseSchedule(T=args.T, schedule="linear", device=device)
    model = UNet(
        in_channels=1,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4),
        time_embed_dim=args.base_channels * 4,
        num_groups=8,
    )
    n_params = model.num_parameters()
    print(f"Model: {n_params:,} parameters")

    trainer = DDPMTrainer(model, schedule, lr=args.lr, device=device)

    # Load checkpoint if it exists
    if Path(args.checkpoint).exists():
        trainer.load(args.checkpoint)

    if args.sample_only:
        print(f"Generating {args.n_samples} samples...")
        sampler = DDPMSampler(model, schedule, device=device)
        samples = sampler.sample(
            n=args.n_samples,
            image_size=(28, 28),
            in_channels=1,
            show_progress=True,
        )
        out_path = f"{args.save_dir}/generated_{args.n_samples}.png"
        save_image_grid(samples, out_path, nrow=8)
        print(f"Saved to {out_path}")
        return

    # Training
    loader = get_mnist_loader(args.batch_size)
    print(f"\nTraining on MNIST: {len(loader.dataset):,} images")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"  T={args.T}, base_channels={args.base_channels}")
    print()

    t0 = time.time()
    all_losses = []

    for epoch in range(args.epochs):
        epoch_losses = []
        for batch_idx, (x0, _) in enumerate(loader):
            x0 = x0 * 2 - 1  # [0,1] -> [-1,1]
            loss = trainer.train_step(x0)
            epoch_losses.append(loss)
            all_losses.append(loss)

            if trainer.step % args.log_interval == 0:
                avg = sum(epoch_losses[-args.log_interval:]) / min(
                    len(epoch_losses), args.log_interval)
                elapsed = time.time() - t0
                print(f"epoch {epoch+1}/{args.epochs} | "
                      f"step {trainer.step} | "
                      f"loss={avg:.4f} | "
                      f"elapsed={elapsed:.0f}s")

            if trainer.step % args.sample_interval == 0:
                print(f"  Generating samples at step {trainer.step}...")
                model.eval()
                sampler = DDPMSampler(model, schedule, device=device)
                samples = sampler.sample(n=16, image_size=(28, 28), in_channels=1)
                save_image_grid(
                    samples,
                    f"{args.save_dir}/samples_step{trainer.step:06d}.png",
                    nrow=4
                )
                model.train()
                trainer.save(args.checkpoint)

        epoch_avg = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1}/{args.epochs} complete | avg_loss={epoch_avg:.4f}")

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Final loss: {all_losses[-1]:.4f}")
    print(f"Loss reduction: {all_losses[0]:.4f} -> {min(all_losses[-100:]):.4f}")

    # Save final checkpoint and results
    trainer.save(args.checkpoint)
    results = {
        "epochs": args.epochs,
        "total_steps": trainer.step,
        "total_time_s": total_time,
        "final_loss": all_losses[-1],
        "initial_loss": all_losses[0],
        "n_params": n_params,
        "device": device,
    }
    Path(f"{args.save_dir}/results.json").write_text(json.dumps(results, indent=2))

    # Generate final sample grid
    print("\nGenerating final samples...")
    model.eval()
    sampler = DDPMSampler(model, schedule, device=device)
    samples = sampler.sample(
        n=64, image_size=(28, 28), in_channels=1, show_progress=True
    )
    final_path = f"{args.save_dir}/final_samples.png"
    save_image_grid(samples, final_path, nrow=8)
    print(f"Final samples saved to {final_path}")

    # Also save progressive denoising frames
    print("Generating denoising animation frames...")
    frames = sampler.sample_progressive(n=4, image_size=(28, 28), save_every=100)
    for i, frame in enumerate(frames):
        save_image_grid(
            frame,
            f"{args.save_dir}/denoise_frame_{i:03d}.png",
            nrow=4
        )
    print(f"Saved {len(frames)} denoising frames to {args.save_dir}/")


if __name__ == "__main__":
    main()

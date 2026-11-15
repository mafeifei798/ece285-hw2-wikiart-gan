import argparse
import copy
import csv
import json
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from augment import DiffAugment
from data import WikiArt
from model import Discriminator, Generator, get_device, initialize


def discriminator_loss(real_logits, fake_logits, loss="bce"):
    if loss == "hinge":
        return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
    # Sum of the real and fake terms, so the equilibrium value is 2 ln 2.
    return (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))


def generator_loss(fake_logits, loss="bce"):
    if loss == "hinge":
        return -fake_logits.mean()
    # Non-saturating loss; the equilibrium value is ln 2.
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def train_step(real, generator, discriminator, optimizer_g, optimizer_d, loss="bce", augment=None):
    """Update D and G; return their losses and mean real/fake logits."""
    augment = augment or (lambda images: images)
    z = torch.randn(len(real), generator.latent_dim, device=real.device)
    fake = generator(z)

    optimizer_d.zero_grad(set_to_none=True)
    real_logits = discriminator(augment(real))
    fake_logits = discriminator(augment(fake.detach()))
    loss_d = discriminator_loss(real_logits, fake_logits, loss)
    loss_d.backward()
    optimizer_d.step()

    discriminator.requires_grad_(False)
    optimizer_g.zero_grad(set_to_none=True)
    loss_g = generator_loss(discriminator(augment(fake)), loss)
    loss_g.backward()
    optimizer_g.step()
    discriminator.requires_grad_(True)
    return torch.stack([loss_d, loss_g, real_logits.mean(), fake_logits.mean()]).detach()


@torch.no_grad()
def update_ema(ema, model, decay):
    for averaged, current in zip(ema.parameters(), model.parameters()):
        averaged.lerp_(current, 1 - decay)


@torch.no_grad()
def recalibrate_batchnorm(generator, batches=32, batch_size=64, seed=0):
    """Refresh EMA BatchNorm statistics using a separate noise generator."""
    device = next(generator.parameters()).device
    norms = [m for m in generator.modules() if isinstance(m, nn.BatchNorm2d)]
    momenta = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over the calibration batches
    rng = torch.Generator().manual_seed(seed)
    generator.train()
    for _ in range(batches):
        generator(torch.randn(batch_size, generator.latent_dim, generator=rng).to(device))
    for m, momentum in zip(norms, momenta):
        m.momentum = momentum
    generator.eval()


def save_checkpoint(generator, config, epoch, path, raw=None):
    """Save the generator used for evaluation, plus the raw weights when it is an EMA copy."""
    checkpoint = {"generator": generator.state_dict(), "config": config, "epoch": epoch}
    if raw is not None:
        checkpoint["generator_raw"] = raw.state_dict()
    torch.save(checkpoint, path)


@torch.inference_mode()
def save_samples(generator, noise, path):
    was_training = generator.training
    generator.eval()
    save_image(generator(noise).add(1).div(2).clamp(0, 1), path, nrow=8)
    generator.train(was_training)


def plot_losses(history, path, loss="bce"):
    """history rows: [epoch, loss_d, loss_g, mean real logit, mean fake logit, ...]."""
    epochs = [r[0] for r in history]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.2))
    left.plot(epochs, [r[1] for r in history], label="Discriminator")
    left.plot(epochs, [r[2] for r in history], label="Generator")
    left.set(xlabel="Epoch", ylabel="Hinge loss" if loss == "hinge" else "BCE loss")
    left.legend()
    right.plot(epochs, [r[3] for r in history], label="Real images")
    right.plot(epochs, [r[4] for r in history], label="Generated images")
    right.axhline(0, color="0.6", linewidth=0.8)
    right.set(xlabel="Epoch", ylabel="Mean discriminator logit")
    right.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train a 128x128 DCGAN on WikiArt")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="output/dcgan128")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    # Options added after the baseline run; the defaults reproduce the baseline.
    parser.add_argument("--flip", action="store_true", help="Random horizontal flips of training images")
    parser.add_argument("--ema", type=float, default=0.0,
                        help="Decay of an exponential moving average of the generator weights, e.g. 0.999 (0 = off)")
    parser.add_argument("--diffaug", default="",
                        help="DiffAugment policy for the discriminator input, e.g. color,translation,cutout")
    parser.add_argument("--upsample", choices=["deconv", "nearest"], default="deconv",
                        help="Generator upsampling: strided transposed conv, or nearest neighbour + 3x3 conv")
    parser.add_argument("--d-norm", choices=["bn", "sn"], default="bn",
                        help="Discriminator normalization: BatchNorm or spectral normalization")
    parser.add_argument("--loss", choices=["bce", "hinge"], default="bce")
    args = parser.parse_args()
    output = Path(args.output)
    if (output / "losses.csv").exists():
        raise FileExistsError("Choose a new output directory for this run")
    if args.epochs < 1 or args.save_every < 1 or args.batch_size < 2:
        parser.error("epochs/save-every must be positive and batch-size must be at least 2")
    if not 0 <= args.ema < 1:
        parser.error("--ema must be in [0, 1)")
    try:
        augment = DiffAugment(args.diffaug) if args.diffaug else None
    except ValueError as error:
        parser.error(str(error))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    if device.type == "mps" and args.d_norm == "sn" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
        parser.error("PyTorch's spectral norm uses torch.vdot, which MPS lacks; "
                     "set PYTORCH_ENABLE_MPS_FALLBACK=1 or train on cuda/cpu")
    dataset = WikiArt(args.data, "train", flip=args.flip)
    if len(dataset) < args.batch_size:
        parser.error("batch-size exceeds the training set")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    generator = Generator(args.latent_dim, args.width, args.upsample).to(device)
    discriminator = Discriminator(args.width, args.d_norm).to(device)
    generator.apply(initialize)
    discriminator.apply(initialize)
    ema = copy.deepcopy(generator).requires_grad_(False) if args.ema else None
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    fixed_noise = torch.randn(64, args.latent_dim,
                             generator=torch.Generator().manual_seed(args.seed)).to(device)
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"image_size": 128}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Device: {device}; training images: {len(dataset)}; "
          f"parameters: G {sum(p.numel() for p in generator.parameters()):,}, "
          f"D {sum(p.numel() for p in discriminator.parameters()):,}", flush=True)
    history = []
    with (output / "losses.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss_d", "loss_g", "d_real", "d_fake", "seconds"])
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            totals = torch.zeros(4, device=device)
            for real in loader:
                totals += train_step(real.to(device, non_blocking=True), generator, discriminator,
                                     optimizer_g, optimizer_d, args.loss, augment)
                if ema is not None:
                    update_ema(ema, generator, args.ema)
            means = (totals / len(loader)).tolist()
            row = [epoch, *means, time.time() - start]
            history.append(row)
            writer.writerow(row)
            f.flush()
            sampler = generator
            if ema is not None:
                recalibrate_batchnorm(ema)
                sampler = ema
            raw = generator if ema is not None else None
            save_checkpoint(sampler, config, epoch, output / "last.pt", raw)
            if epoch == 1 or epoch % args.save_every == 0 or epoch == args.epochs:
                save_checkpoint(sampler, config, epoch, output / f"epoch_{epoch:03d}.pt", raw)
                save_samples(sampler, fixed_noise, output / f"samples_{epoch:03d}.png")
                plot_losses(history, output / "losses.png", args.loss)
            print(f"Epoch {epoch:3d}: D={row[1]:.4f}, G={row[2]:.4f}, "
                  f"D(real)={row[3]:+.2f}, D(fake)={row[4]:+.2f}, {row[5]:.1f}s", flush=True)


if __name__ == "__main__":
    main()

import argparse
import copy
import os
import tempfile
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader

from augment import DiffAugment, cutout, translation
from data import WikiArt, preprocess, save_dataset, select_images
from evaluate import inception_input, score_probabilities
from model import Discriminator, Generator, get_device, initialize, load_generator
from train import (plot_losses, recalibrate_batchnorm, save_checkpoint, save_samples, train_step,
                   update_ema)

# (name, generator upsampling, discriminator normalization, loss, DiffAugment policy)
CONFIGS = [("baseline", "deconv", "bn", "bce", ""),
           ("diffaug", "deconv", "bn", "bce", "color,translation,cutout"),
           ("nearest+sn+hinge", "nearest", "sn", "hinge", "color,translation,cutout")]


def check_training(real, upsample, d_norm, loss, policy):
    generator = Generator(upsample=upsample).to(real.device)
    discriminator = Discriminator(norm=d_norm).to(real.device)
    generator.apply(initialize)
    discriminator.apply(initialize)
    ema = copy.deepcopy(generator).requires_grad_(False)
    optim_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    before_g = [p.detach().clone() for p in generator.parameters()]
    before_d = [p.detach().clone() for p in discriminator.parameters()]
    augment = DiffAugment(policy) if policy else None
    stats = []
    for _ in range(2):
        stats.append(train_step(real, generator, discriminator, optim_g, optim_d, loss, augment))
        update_ema(ema, generator, 0.5)
    stats = torch.stack(stats)
    assert stats.shape == (2, 4) and torch.isfinite(stats).all()
    assert all(not torch.equal(a, b) for a, b in zip(before_g, generator.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(before_d, discriminator.parameters()))
    assert all(torch.isfinite(p).all() for net in (generator, discriminator) for p in net.parameters())
    # The EMA copy lies strictly between the initial and the current weights.
    for start, averaged, current in zip(before_g, ema.parameters(), generator.parameters()):
        assert not torch.equal(averaged, current) and not torch.equal(averaged, start)
    recalibrate_batchnorm(ema, batches=2, batch_size=4)
    assert not ema.training
    assert all(torch.isfinite(m.running_var).all() and m.momentum == 0.1
               for m in ema.modules() if isinstance(m, torch.nn.BatchNorm2d))
    return generator, ema, stats


def check_augment():
    x = torch.rand(6, 3, 128, 128).mul(2).sub(1).requires_grad_(True)
    y = DiffAugment()(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad.abs().sum() > 0
    ones = torch.ones(4, 3, 128, 128)
    zero_fraction = (cutout(ones) == 0).float().mean().item()
    assert 0 < zero_fraction <= 0.25 + 1e-6
    # Translation only moves pixels: every output pixel is an input pixel or zero padding.
    shifted = translation(ones * 3)
    assert set(shifted.unique().tolist()) <= {0.0, 3.0}
    try:
        DiffAugment("color,rotate")
        raise AssertionError("unknown policy accepted")
    except ValueError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Optimizer steps for every option, plus file/metric checks")
    parser.add_argument("--data", default="data")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--inception", action="store_true", help="Also check pretrained Inception on one batch")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(7)
    device = get_device(args.device)
    datasets = [WikiArt(args.data, split) for split in ("train", "val", "test")]
    files = [{r["source"] for r in d.rows} for d in datasets]
    assert not (files[0] & files[1] or files[0] & files[2] or files[1] & files[2])
    real = next(iter(DataLoader(datasets[0], batch_size=4))).to(device)
    assert real.shape == (4, 3, 128, 128) and real.min() >= -1 and real.max() <= 1
    flipped = WikiArt(args.data, "train", flip=True)
    views = {tuple(flipped[0].flatten()[:64].tolist()) for _ in range(40)}
    assert {tuple(datasets[0][0].flatten()[:64].tolist()), tuple(datasets[0][0].flip(-1).flatten()[:64].tolist())} >= views
    assert len(views) == 2
    check_augment()

    noise = torch.randn(2, 128, device=device)
    with tempfile.TemporaryDirectory(prefix="ece285_hw2_smoke_") as folder:
        folder = Path(folder)
        for name, upsample, d_norm, loss, policy in CONFIGS:
            if device.type == "mps" and d_norm == "sn" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
                print(f"Skipped {name}: spectral norm on MPS needs PYTORCH_ENABLE_MPS_FALLBACK=1")
                continue
            generator, ema, stats = check_training(real, upsample, d_norm, loss, policy)
            config = {"latent_dim": 128, "width": 32, "image_size": 128, "upsample": upsample}
            save_checkpoint(ema, config, 0, folder / "model.pt", raw=generator)
            loaded_ema, checkpoint = load_generator(folder / "model.pt", device)
            loaded_raw, _ = load_generator(folder / "model.pt", device, raw=True)
            assert checkpoint["weights"] == "ema"
            for expected, loaded in [(ema, loaded_ema), (generator, loaded_raw)]:
                assert all(torch.equal(value, loaded.state_dict()[key])
                           for key, value in expected.state_dict().items())
            generator.eval()
            with torch.no_grad():
                # Identical CUDA weights can use convolution kernels with slightly different rounding.
                tolerance = {"rtol": 1e-4, "atol": 1e-5} if device.type == "cuda" else {"rtol": 1e-5, "atol": 1e-6}
                torch.testing.assert_close(loaded_ema(noise), ema(noise), **tolerance)
                torch.testing.assert_close(loaded_raw(noise), generator(noise), **tolerance)
                out = loaded_ema(noise)
            assert out.shape == (2, 3, 128, 128) and out.min() >= -1 and out.max() <= 1
            save_samples(ema, noise, folder / f"samples_{name}.png")
            plot_losses([[i + 1, *row.tolist(), 0] for i, row in enumerate(stats)],
                        folder / f"losses_{name}.png", loss)
            print(f"Training step and checkpoint: {name} passed")

        # Checkpoints written before the new options existed (no "upsample", no EMA) still load.
        old = Generator().to(device)
        save_checkpoint(old, {"latent_dim": 128, "width": 32, "image_size": 128}, 0, folder / "old.pt")
        loaded, checkpoint = load_generator(folder / "old.pt", device)
        assert checkpoint["weights"] == "raw"
        assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in old.state_dict().items())

        for name in sorted(p.name for p in folder.glob("*.png")):
            with Image.open(folder / name) as image:
                image.verify()
        toy = [(f"style/{i}.jpg", preprocess(Image.new("RGB", (160, 100), (i * 20, 30, 40)))) for i in range(9)]
        info = save_dataset(toy + [toy[0]], folder / "data", holdout=2)
        assert info["counts"] == {"train": 5, "val": 2, "test": 2} and info["duplicates_removed"] == 1
        names = [name for name, _ in toy]
        assert select_images(names, 5) == select_images(reversed(names), 5)
    resized = inception_input(real)
    assert resized.shape == (4, 3, 299, 299) and 0 <= resized.min() <= resized.max() <= 1
    uniform = torch.full((8, 4), 0.25)
    metrics = score_probabilities(uniform, uniform)
    assert abs(metrics["fid_ignite"]) < 1e-6 and abs(metrics["inception_score"] - 1) < 1e-6
    diverse = torch.eye(4).repeat(2, 1)
    assert abs(score_probabilities(diverse, diverse)["inception_score"] - 4) < 1e-6
    if args.inception:
        from ignite.metrics.gan.utils import InceptionModel
        extractor = InceptionModel(return_features=False, device=device).eval()
        with torch.inference_mode():
            probabilities = extractor(resized.to(device)).cpu()
        assert probabilities.shape == (4, 1000) and torch.isfinite(probabilities).all()
        assert torch.allclose(probabilities.sum(1), torch.ones(4), atol=1e-5)
        print("Pretrained Inception: passed")
    print(f"Smoke test passed on {device}: data, flips, DiffAugment, GAN steps for every option, "
          "EMA, checkpoints, plots, FID/IS")


if __name__ == "__main__":
    main()

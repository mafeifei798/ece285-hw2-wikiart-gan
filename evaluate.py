import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel

from data import WikiArt
from model import get_device, load_generator

FID_FEATURES = "Ignite Inception probabilities (1000 dimensions)"


def inception_input(images):
    pixels = images.detach().cpu().add(1).mul(127.5).round().clamp(0, 255).byte()
    result = []
    for pixels_rgb in pixels:
        image = Image.fromarray(pixels_rgb.permute(1, 2, 0).numpy())
        image = image.resize((299, 299), Image.Resampling.BILINEAR)
        result.append(to_tensor(image))
    return torch.stack(result)


def fid_score(a, b):
    # The assignment's Ignite tutorial uses 1000-dimensional class probabilities for FID.
    a, b = a.double().cpu(), b.double().cpu()
    fid = FID(num_features=a.shape[1], feature_extractor=nn.Identity(), device="cpu")
    fid.update((a, b))
    return float(fid.compute())


def inception_score(probabilities):
    probabilities = probabilities.double().cpu()
    score = InceptionScore(num_features=probabilities.shape[1], feature_extractor=nn.Identity(), device="cpu")
    score.update(probabilities)
    return float(score.compute())


def score_probabilities(real, fake):
    return {"fid_ignite": fid_score(fake, real), "inception_score": inception_score(fake)}


def real_subset(data, split, count, seed):
    """A seeded random subset of a split (the whole split when count is None)."""
    dataset = WikiArt(data, split)
    count = len(dataset) if count is None else count
    if not 2 <= count <= len(dataset):
        raise ValueError(f"num-samples must be between 2 and {len(dataset)} for split {split}")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:count]
    return Subset(dataset, indices.tolist())


@torch.inference_mode()
def real_probabilities(images, inception, batch_size, device):
    probabilities = []
    for real in DataLoader(images, batch_size=batch_size):
        probabilities.append(inception(inception_input(real).to(device)).cpu())
    return torch.cat(probabilities)


@torch.inference_mode()
def generated_probabilities(generator, count, seed, inception, batch_size, device):
    noise = torch.randn(count, generator.latent_dim, generator=torch.Generator().manual_seed(seed))
    probabilities, samples = [], []
    for z in noise.split(batch_size):
        fake = generator(z.to(device))
        probabilities.append(inception(inception_input(fake).to(device)).cpu())
        samples.append(fake.cpu())
    return torch.cat(probabilities), torch.cat(samples)


def save_results(folder, samples, metrics):
    folder.mkdir(parents=True, exist_ok=True)
    generated = samples.add(1).div(2).clamp(0, 1)
    save_image(generated[:64], folder / "samples.png", nrow=8)
    np.savez_compressed(folder / "samples.npz", images=generated.mul(255).round().byte().numpy())
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def evaluate(args):
    device = get_device(args.device)
    real = real_subset(args.data, args.split, args.num_samples, args.seed)
    count = len(real)
    inception = InceptionModel(return_features=False, device=device).eval()
    real_prob = real_probabilities(real, inception, args.batch_size, device)
    print(f"Scored {count} real {args.split} images", flush=True)
    output = Path(args.output)
    sweep = len(args.checkpoint) > 1
    rows = []
    for path in args.checkpoint:
        generator, checkpoint = load_generator(path, device, raw=args.raw)
        fake_prob, samples = generated_probabilities(generator, count, args.seed, inception,
                                                     args.batch_size, device)
        metrics = score_probabilities(real_prob, fake_prob)
        metrics.update({"checkpoint": str(path), "epoch": checkpoint["epoch"],
                        "weights": checkpoint["weights"], "split": args.split, "seed": args.seed,
                        "real_count": count, "generated_count": count, "fid_features": FID_FEATURES})
        save_results(output / Path(path).stem if sweep else output, samples, metrics)
        rows.append(metrics)
        print(f"{path}: FID {metrics['fid_ignite']:.4f}, IS {metrics['inception_score']:.3f}", flush=True)
    if sweep:
        fields = ["checkpoint", "epoch", "weights", "fid_ignite", "inception_score"]
        with (output / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        best = min(rows, key=lambda r: r["fid_ignite"])
        print(f"Lowest {args.split} FID: {best['checkpoint']} "
              f"(FID {best['fid_ignite']:.4f}, IS {best['inception_score']:.3f})")
    else:
        print(json.dumps(rows[0], indent=2))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a GAN with the course's Ignite metrics")
    parser.add_argument("--checkpoint", required=True, nargs="+",
                        help="One checkpoint, or several to score each on the same real images")
    parser.add_argument("--raw", action="store_true", help="Use the raw instead of the EMA weights")
    parser.add_argument("--data", default="data")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output", default="output/evaluation",
                        help="Output folder; a sweep writes one subfolder per checkpoint and summary.csv")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()

"""Compute real-image IS and train/val/test FID with the evaluation preprocessing."""
import argparse
import json
from pathlib import Path

from ignite.metrics.gan.utils import InceptionModel

from evaluate import FID_FEATURES, fid_score, inception_score, real_probabilities, real_subset
from model import get_device


def main():
    parser = argparse.ArgumentParser(description="Real-vs-real FID and real-image IS")
    parser.add_argument("--data", default="data")
    parser.add_argument("--num-samples", type=int, default=1000, help="Images drawn from each split")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output", default="output/reference")
    args = parser.parse_args()
    device = get_device(args.device)
    inception = InceptionModel(return_features=False, device=device).eval()
    probabilities = {}
    for split in ("train", "val", "test"):
        images = real_subset(args.data, split, args.num_samples, args.seed)
        probabilities[split] = real_probabilities(images, inception, args.batch_size, device)
        print(f"Scored {len(images)} real {split} images", flush=True)
    metrics = {
        "inception_score": {split: inception_score(p) for split, p in probabilities.items()},
        "fid_ignite": {
            "train_vs_val": fid_score(probabilities["train"], probabilities["val"]),
            "train_vs_test": fid_score(probabilities["train"], probabilities["test"]),
            "val_vs_test": fid_score(probabilities["val"], probabilities["test"]),
        },
        "count_per_split": args.num_samples, "seed": args.seed, "fid_features": FID_FEATURES,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

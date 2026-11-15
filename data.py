import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


SOURCE = "https://www.kaggle.com/datasets/steubk/wikiart"
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def select_images(names, limit=12000, seed=2026):
    names = sorted(name for name in names if Path(name).suffix.lower() in EXTENSIONS)
    return sorted(random.Random(seed).sample(names, min(limit, len(names))))


def preprocess(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.fit(image, (128, 128), method=Image.Resampling.LANCZOS)


def save_dataset(images, output, seed=2026, holdout=1000):
    output = Path(output)
    if (output / "splits.csv").exists():
        raise FileExistsError(f"A prepared dataset already exists at {output}")
    (output / "images").mkdir(parents=True, exist_ok=True)
    records, seen = [], set()
    duplicates = 0
    for index, (source, image) in enumerate(images):
        # Remove identical resized images before assigning the splits.
        digest = hashlib.sha1(image.tobytes()).digest()
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        filename = f"images/{index:05d}.png"
        image.save(output / filename)
        records.append({"file": filename, "source": source,
                        "style": source.split("/")[0]})
    if len(records) <= 2 * holdout or holdout < 1:
        raise ValueError("Need more images than the two holdout splits combined")
    random.Random(seed).shuffle(records)
    for i, row in enumerate(records):
        row["split"] = "test" if i < holdout else "val" if i < 2 * holdout else "train"
    with (output / "splits.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "source", "style", "split"])
        writer.writeheader()
        writer.writerows(records)
    info = {"source": SOURCE, "seed": seed, "image_size": 128,
            "preprocessing": "EXIF orientation, RGB, aspect-preserving resize and center crop",
            "duplicates_removed": duplicates,
            "counts": {s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")}}
    (output / "dataset.json").write_text(json.dumps(info, indent=2) + "\n")
    preview = Image.new("RGB", (8 * 128, 8 * 128))
    for i, row in enumerate(r for r in records if r["split"] == "train"):
        if i == 64:
            break
        with Image.open(output / row["file"]) as image:
            preview.paste(image, (i % 8 * 128, i // 8 * 128))
    preview.save(output / "preview.png")
    return info


class WikiArt(Dataset):
    def __init__(self, root="data", split="train", flip=False):
        self.root = Path(root)
        self.flip = flip  # random horizontal flips; used for training only
        with (self.root / "splits.csv").open() as f:
            self.rows = [row for row in csv.DictReader(f) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No images in split {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.root / self.rows[index]["file"]) as image:
            pixels = to_tensor(image.convert("RGB")).mul(2).sub(1)
        if self.flip and torch.rand(()) < 0.5:
            pixels = pixels.flip(-1)
        return pixels


def main():
    parser = argparse.ArgumentParser(description="Prepare a WikiArt subset from extracted images")
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=12000)
    parser.add_argument("--holdout", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    names = [p.relative_to(args.source_dir).as_posix() for p in args.source_dir.rglob("*") if p.is_file()]
    selected = select_images(names, args.limit, args.seed)

    def images():
        for i, name in enumerate(selected):
            with Image.open(args.source_dir / name) as image:
                yield name, preprocess(image)
            if (i + 1) % 500 == 0:
                print(f"Prepared {i + 1}/{len(selected)}", flush=True)

    print(json.dumps(save_dataset(images(), args.output, args.seed, args.holdout), indent=2))


if __name__ == "__main__":
    main()

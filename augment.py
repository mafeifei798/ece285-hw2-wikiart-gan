"""DiffAugment (Zhao et al., 2020): color, translation and cutout.

Applied to real and generated images, with gradients flowing through the
transforms. Input shape: (N, 3, H, W), with pixels in [-1, 1].
"""
import torch
import torch.nn.functional as F


def color(x):
    """Random brightness, saturation and contrast, one draw per image."""
    n = len(x)
    x = x + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5)
    mean = x.mean(1, keepdim=True)
    x = (x - mean) * (torch.rand(n, 1, 1, 1, device=x.device) * 2) + mean
    mean = x.mean((1, 2, 3), keepdim=True)
    return (x - mean) * (torch.rand(n, 1, 1, 1, device=x.device) + 0.5) + mean


def translation(x, ratio=0.125):
    """Shift each image by up to ratio * size pixels, filling the border with zeros."""
    n, _, h, w = x.shape
    dy, dx = round(h * ratio), round(w * ratio)
    shift_y = torch.randint(-dy, dy + 1, (n, 1, 1), device=x.device)
    shift_x = torch.randint(-dx, dx + 1, (n, 1, 1), device=x.device)
    # Index into a 1-pixel zero-padded copy; out-of-range positions land on the padding.
    rows = (torch.arange(h, device=x.device).view(1, h, 1) + shift_y + 1).clamp(0, h + 1)
    cols = (torch.arange(w, device=x.device).view(1, 1, w) + shift_x + 1).clamp(0, w + 1)
    batch = torch.arange(n, device=x.device).view(n, 1, 1)
    padded = F.pad(x, (1, 1, 1, 1)).permute(0, 2, 3, 1)
    # contiguous(): the MPS convolution backward rejects the permuted strides.
    return padded[batch, rows, cols].permute(0, 3, 1, 2).contiguous()


def cutout(x, ratio=0.5):
    """Zero out one square of side ratio * size at a random position in each image."""
    n, _, h, w = x.shape
    size_y, size_x = round(h * ratio), round(w * ratio)
    top = torch.randint(0, h, (n, 1, 1), device=x.device) - size_y // 2
    left = torch.randint(0, w, (n, 1, 1), device=x.device) - size_x // 2
    rows = torch.arange(h, device=x.device).view(1, h, 1)
    cols = torch.arange(w, device=x.device).view(1, 1, w)
    inside = (rows >= top) & (rows < top + size_y) & (cols >= left) & (cols < left + size_x)
    return x * (~inside).unsqueeze(1).to(x.dtype)


TRANSFORMS = {"color": color, "translation": translation, "cutout": cutout}


class DiffAugment:
    def __init__(self, policy="color,translation,cutout"):
        self.names = [name.strip() for name in policy.split(",") if name.strip()]
        unknown = [name for name in self.names if name not in TRANSFORMS]
        if unknown or not self.names:
            raise ValueError(f"DiffAugment policy must list some of {sorted(TRANSFORMS)}, got {policy!r}")

    def __call__(self, x):
        for name in self.names:
            x = TRANSFORMS[name](x)
        return x

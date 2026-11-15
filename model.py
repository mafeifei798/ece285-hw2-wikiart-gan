import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import spectral_norm


def upsample_block(channels_in, channels_out, mode):
    """Double the resolution with a strided transposed conv or nearest-neighbour upsampling + 3x3 conv."""
    if mode == "deconv":
        return [nn.ConvTranspose2d(channels_in, channels_out, 4, 2, 1, bias=False)]
    if mode == "nearest":
        # Avoids the checkerboard artifacts of strided transposed convolutions (Odena et al., 2016).
        return [nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(channels_in, channels_out, 3, 1, 1, bias=False)]
    raise ValueError(f"Unknown upsampling mode: {mode}")


class Generator(nn.Module):
    def __init__(self, latent_dim=128, width=32, upsample="deconv"):
        super().__init__()
        self.latent_dim = latent_dim
        layers = [nn.ConvTranspose2d(latent_dim, width * 16, 4, 1, 0, bias=False),
                  nn.BatchNorm2d(width * 16), nn.ReLU(True)]
        for a, b in [(16, 8), (8, 4), (4, 2), (2, 1)]:
            layers += [*upsample_block(width * a, width * b, upsample),
                       nn.BatchNorm2d(width * b), nn.ReLU(True)]
        layers += [*upsample_block(width, 3, upsample), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z.reshape(-1, self.latent_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, width=32, norm="bn"):
        super().__init__()
        if norm not in ("bn", "sn"):
            raise ValueError(f"Unknown discriminator normalization: {norm}")
        # "bn" is the DCGAN discriminator. "sn" replaces BatchNorm with spectral
        # normalization on every convolution (Miyato et al., 2018).
        sn = norm == "sn"

        def conv(channels_in, channels_out, *args):
            layer = nn.Conv2d(channels_in, channels_out, *args, bias=sn)
            return spectral_norm(layer) if sn else layer

        layers = [conv(3, width, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        for a, b in [(1, 2), (2, 4), (4, 8), (8, 16)]:
            layers.append(conv(width * a, width * b, 4, 2, 1))
            if not sn:
                layers.append(nn.BatchNorm2d(width * b))
            layers.append(nn.LeakyReLU(0.2, True))
        layers += [conv(width * 16, 1, 4, 1, 0)]
        self.net = nn.Sequential(*layers)

    def forward(self, images):
        return self.net(images).flatten()


def initialize(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        if parametrize.is_parametrized(module, "weight"):
            return  # Spectral-normalized layers keep PyTorch's default initialization.
        nn.init.normal_(module.weight, 0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1, 0.02)
        nn.init.zeros_(module.bias)


def load_generator(path, device="cpu", raw=False):
    """Load the generator from a checkpoint.

    Checkpoints from EMA runs hold the averaged weights under "generator" and the
    raw training weights under "generator_raw"; raw=True selects the latter.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    generator = Generator(config["latent_dim"], config["width"], config.get("upsample", "deconv"))
    key = "generator_raw" if raw and "generator_raw" in checkpoint else "generator"
    generator.load_state_dict(checkpoint[key])
    checkpoint["weights"] = "ema" if key == "generator" and "generator_raw" in checkpoint else "raw"
    return generator.to(device).eval(), checkpoint


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)

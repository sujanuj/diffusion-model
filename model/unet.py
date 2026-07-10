"""U-Net denoising network for DDPM.

The U-Net predicts the noise added to an image at a given timestep.
Architecture:
  - Encoder: series of downsampling blocks (Conv -> GroupNorm -> SiLU)
  - Bottleneck: two residual blocks
  - Decoder: series of upsampling blocks with skip connections from encoder
  - Time embedding: sinusoidal embedding of timestep t, projected into
    each residual block via a learned linear layer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, num_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class UNet(nn.Module):
    """U-Net denoising network for DDPM.

    Args:
        in_channels: image channels (1=MNIST, 3=RGB)
        base_channels: base channel width
        channel_mults: multipliers at each resolution
        time_embed_dim: time embedding dimension
        num_groups: GroupNorm groups
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_mults: tuple = (1, 2, 4),
        time_embed_dim: int = 128,
        num_groups: int = 8,
    ):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_embed_dim)
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Build channel sizes at each level
        channels = [base_channels] + [base_channels * m for m in channel_mults]
        # channels = [base, base*m0, base*m1, ...]

        # Encoder
        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(channel_mults)):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            self.downs.append(nn.ModuleList([
                ResidualBlock(in_ch, out_ch, time_embed_dim, num_groups),
                ResidualBlock(out_ch, out_ch, time_embed_dim, num_groups),
            ]))
            # Downsample all but last level
            if i < len(channel_mults) - 1:
                self.downsamples.append(nn.Conv2d(out_ch, out_ch, 4, 2, 1))
            else:
                self.downsamples.append(nn.Identity())

        # Bottleneck
        bot_ch = channels[-1]
        self.mid1 = ResidualBlock(bot_ch, bot_ch, time_embed_dim, num_groups)
        self.mid2 = ResidualBlock(bot_ch, bot_ch, time_embed_dim, num_groups)

        # Decoder
        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in reversed(range(len(channel_mults))):
            in_ch = channels[i + 1]
            skip_ch = channels[i + 1]
            out_ch = channels[i]
            # Upsample all but the last (highest resolution) level
            if i > 0:
                self.upsamples.append(nn.ConvTranspose2d(in_ch, in_ch, 4, 2, 1))
            else:
                self.upsamples.append(nn.Identity())
            self.ups.append(nn.ModuleList([
                ResidualBlock(in_ch + skip_ch, out_ch, time_embed_dim, num_groups),
                ResidualBlock(out_ch, out_ch, time_embed_dim, num_groups),
            ]))

        self.final_norm = nn.GroupNorm(num_groups, channels[1])
        self.final_conv = nn.Conv2d(channels[1], in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        x = self.init_conv(x)

        # Encoder: save skip connections
        skips = []
        for (res1, res2), ds in zip(self.downs, self.downsamples):
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            skips.append(x)
            x = ds(x)

        # Bottleneck
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        # Decoder: use skip connections in reverse
        for (res1, res2), us, skip in zip(self.ups, self.upsamples, reversed(skips)):
            x = us(x)
            x = torch.cat([x, skip], dim=1)
            x = res1(x, t_emb)
            x = res2(x, t_emb)

        x = F.silu(self.final_norm(x))
        x = self.final_conv(x)
        return x

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

"""U-Net denoising network for DDPM - fixed architecture."""

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
    """U-Net for DDPM.

    Simple fixed architecture:
      init_conv -> [res, res, downsample] x N -> mid -> [upsample, res, res] x N -> out
    Skip connections connect each encoder level to the corresponding decoder level.
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
        td = time_embed_dim
        ng = num_groups

        ch = [base_channels * m for m in channel_mults]
        # ch[0], ch[1], ..., ch[-1] from coarse to fine (reversed from mults order)
        # Actually ch[0]=base*mults[0] is finest, ch[-1] is coarsest

        self.init_conv = nn.Conv2d(in_channels, ch[0], 3, padding=1)

        # Encoder: for each level, two res blocks then downsample
        self.enc_res = nn.ModuleList()
        self.enc_ds  = nn.ModuleList()
        in_c = ch[0]
        self.enc_channels = []  # track output channels for skip connections
        for i, out_c in enumerate(ch):
            self.enc_res.append(nn.ModuleList([
                ResidualBlock(in_c, out_c, td, ng),
                ResidualBlock(out_c, out_c, td, ng),
            ]))
            self.enc_channels.append(out_c)
            if i < len(ch) - 1:
                self.enc_ds.append(nn.Conv2d(out_c, out_c, 4, 2, 1))
            else:
                self.enc_ds.append(nn.Identity())
            in_c = out_c

        # Bottleneck
        self.mid = nn.ModuleList([
            ResidualBlock(ch[-1], ch[-1], td, ng),
            ResidualBlock(ch[-1], ch[-1], td, ng),
        ])

        # Decoder: for each level (reversed), upsample then two res blocks
        self.dec_us  = nn.ModuleList()
        self.dec_res = nn.ModuleList()
        rev_ch = list(reversed(ch))  # from coarsest to finest
        rev_enc = list(reversed(self.enc_channels))

        in_c = rev_ch[0]
        for i in range(len(rev_ch) - 1):
            skip_c = rev_enc[i + 1]
            out_c  = rev_ch[i + 1]
            self.dec_us.append(nn.ConvTranspose2d(in_c, in_c, 4, 2, 1))
            self.dec_res.append(nn.ModuleList([
                ResidualBlock(in_c + skip_c, out_c, td, ng),
                ResidualBlock(out_c, out_c, td, ng),
            ]))
            in_c = out_c

        self.final_norm = nn.GroupNorm(ng, ch[0])
        self.final_conv = nn.Conv2d(ch[0], in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        x = self.init_conv(x)

        # Encoder
        skips = []
        for (r1, r2), ds in zip(self.enc_res, self.enc_ds):
            x = r1(x, t_emb)
            x = r2(x, t_emb)
            skips.append(x)
            x = ds(x)

        # Bottleneck
        x = self.mid[0](x, t_emb)
        x = self.mid[1](x, t_emb)

        # Decoder: use all skips in reverse order except the deepest
        skip_list = list(reversed(skips))
        for i, (us, (r1, r2)) in enumerate(zip(self.dec_us, self.dec_res)):
            x = us(x)
            skip = skip_list[i + 1]
            x = torch.cat([x, skip], dim=1)
            x = r1(x, t_emb)
            x = r2(x, t_emb)

        x = F.silu(self.final_norm(x))
        return self.final_conv(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

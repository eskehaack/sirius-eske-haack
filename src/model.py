from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from src.callbacks.timer import log_time

# -------------------------
# Time embeddings
# -------------------------


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


# -------------------------
# UNet blocks
# -------------------------


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, time_dim, dropout)
        self.res2 = ResBlock(out_ch, out_ch, time_dim, dropout)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(
        self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int, dropout: float
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res1 = ResBlock(out_ch + skip_ch, out_ch, time_dim, dropout)
        self.res2 = ResBlock(out_ch, out_ch, time_dim, dropout)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x


class ConditionalUNet(nn.Module):
    """
    Predicts noise epsilon.

    Input:
        noisy target image: [B, target_channels, H, W]
        condition image:    [B, condition_channels, H, W]

    Internally concatenates them along channels:
        [x_t, condition]
    """

    def __init__(
        self,
        target_channels: int = 1,
        condition_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        time_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.target_channels = target_channels
        self.condition_channels = condition_channels

        in_channels = target_channels + condition_channels

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        chs = [base_channels * m for m in channel_mults]

        self.input_conv = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        self.downs = nn.ModuleList()
        in_ch = chs[0]
        for out_ch in chs:
            self.downs.append(DownBlock(in_ch, out_ch, time_dim, dropout))
            in_ch = out_ch

        self.mid1 = ResBlock(chs[-1], chs[-1], time_dim, dropout)
        self.mid2 = ResBlock(chs[-1], chs[-1], time_dim, dropout)

        self.ups = nn.ModuleList()
        rev_chs = list(reversed(chs))
        in_ch = rev_chs[0]
        for skip_ch in rev_chs:
            out_ch = skip_ch
            self.ups.append(UpBlock(in_ch, skip_ch, out_ch, time_dim, dropout))
            in_ch = out_ch

        self.out = nn.Sequential(
            nn.GroupNorm(8, chs[0]),
            nn.SiLU(),
            nn.Conv2d(chs[0], target_channels, 3, padding=1),
        )

    def forward(
        self, x_t: torch.Tensor, condition: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        x = torch.cat([x_t, condition], dim=1)
        x = self.input_conv(x)

        skips = []
        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip, t_emb)

        return self.out(x)


# -------------------------
# Diffusion LightningModule
# -------------------------


class LitConditionalDDPM(pl.LightningModule):
    def __init__(
        self,
        target_channels: int = 1,
        condition_channels: int = 1,
        base_channels: int = 128,
        channel_mults: Sequence[int] = (1, 2, 4, 8, 10),
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        dropout: float = 0.0,
        lr: float = 2e-4,
        image_size: int = 512,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ConditionalUNet(
            target_channels=target_channels,
            condition_channels=condition_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            dropout=dropout,
        )

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.image_size = image_size
        self.lr = lr

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

        self.static = None  # Allocation

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]
        return sqrt_ab * x0 + sqrt_omab * noise

    def _load_batch(self, batch):
        x, y, static, _ = batch

        x0 = torch.stack(
            list(y.values()),
            dim=1,
        )

        x0 = F.interpolate(
            x0,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        condition = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # Only interpolate static once and store it for future use
        if self.static is None:
            self.static = F.interpolate(
                static,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        condition = torch.cat([condition, self.static], dim=1)

        return x0, condition

    @log_time
    def training_step(self, batch, batch_idx, noise_channel: int = 0):
        x0, condition = self._load_batch(batch)

        b = x0.shape[0]
        t = torch.randint(0, self.hparams.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)

        x_t = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_t, condition, t)

        loss = F.mse_loss(noise_pred, noise)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, condition = self._load_batch(batch)

        b = x0.shape[0]
        t = torch.randint(0, self.hparams.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)

        x_t = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_t, condition, t)

        loss = F.mse_loss(noise_pred, noise)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_steps: Optional[int] = None,
        shape: Optional[tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:
        """
        Samples target images conditioned on `condition`.

        Args:
            condition: [B, condition_channels, H, W]
            num_steps: number of reverse diffusion steps. Defaults to training timesteps.
            shape: optional target shape [B, C, H, W]

        Returns:
            samples in roughly [-1, 1]
        """
        self.eval()

        device = condition.device
        b, _, h, w = condition.shape

        if shape is None:
            shape = (b, self.hparams.target_channels, h, w)

        x = torch.randn(shape, device=device)

        steps = num_steps or self.hparams.timesteps
        step_indices = torch.linspace(
            self.hparams.timesteps - 1, 0, steps, device=device
        ).long()

        for i, t_idx in enumerate(step_indices):
            t = torch.full((b,), t_idx.item(), device=device, dtype=torch.long)

            beta_t = self.betas[t][:, None, None, None]
            alpha_t = self.alphas[t][:, None, None, None]
            alpha_bar_t = self.alpha_bars[t][:, None, None, None]

            eps_pred = self.model(x, condition, t)

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
            )

            if t_idx > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean

        return x.clamp(-1, 1)

    def configure_optimizers(self):
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.lr
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

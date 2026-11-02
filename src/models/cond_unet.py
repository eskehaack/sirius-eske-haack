from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from src.callbacks.timer import log_time
from src.models.backbone_unet import ConditionalUNet

class LitConditionalUNet(pl.LightningModule):
    def __init__(
        self,
        target_channels: int = 1,
        condition_channels: int = 1,
        base_channels: int = 128,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule: str = "linear", # Compatibility
        dropout: float = 0.0,
        lr: float = 2e-4,
        image_size: int = 432,
        attention: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ConditionalUNet(
            target_channels=target_channels,
            condition_channels=condition_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            dropout=dropout,
            attention=attention,
        )

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.image_size = image_size
        self.lr = lr

        self.initial_target_shape = (412, 424)
        self.pad_x = (self.image_size - self.initial_target_shape[0]) // 2
        self.pad_y = (self.image_size - self.initial_target_shape[1]) // 2
        self.pad_tuple = (self.pad_y, self.pad_y, self.pad_x, self.pad_x)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

        self.static = None  # Allocation

    def _load_batch(self, batch):
        x, y, static, _ = batch

        x0 = torch.stack(
            list(y.values()),
            dim=1,
        )

        x0 = F.pad(
            x0,
            pad=self.pad_tuple,
            mode="constant",
            value=0.0,
        )

        # Interpolate the condition to match the target shape
        condition = F.interpolate(
            x,
            size=self.initial_target_shape,
            mode="bilinear",
            align_corners=False,
        )
        # Then pad to meet model requirements
        condition = F.pad(
            condition,
            pad=self.pad_tuple,
            mode="constant",
            value=0.0,
        )

        # Only interpolate static once and store it for future use
        if self.static is None:
            # Collapse the batch dimension to allow for broadcasting with other batch sizes
            static = static[0] # Get rid of batch dimension
            static = static.unsqueeze(0)  # Add batch dimension back

            static = F.interpolate(
                static,
                size=self.initial_target_shape,
                mode="bilinear",
                align_corners=False,
            )
            self.static = F.pad(
                static,
                pad=self.pad_tuple,
                mode="constant",
                value=0.0,
            )

        static = self.static.expand(condition.size(0), -1, -1, -1)  # Expand to match batch size
        condition = torch.cat([condition, static], dim=1)

        return x0, condition

    @log_time
    def training_step(self, batch, batch_idx):
        x0, condition = self._load_batch(batch)

        b = x0.shape[0]
        t = torch.full((b,), 0, device=x0.device, dtype=torch.long)
        x_dummy = torch.zeros(
            b, self.hparams.target_channels, condition.shape[2], condition.shape[3],
            device=condition.device,
        )

        pred = self.model(x_dummy, condition, t)
        loss = F.mse_loss(pred, x0)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, condition = self._load_batch(batch)

        b = x0.shape[0]
        t = torch.full((b,), 0, device=x0.device, dtype=torch.long)
        x_dummy = torch.zeros(
            b, self.hparams.target_channels, condition.shape[2], condition.shape[3],
            device=condition.device,
        )

        pred = self.model(x_dummy, condition, t)
        loss = F.mse_loss(pred, x0)

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

        t = torch.full((b,), 0, device=device, dtype=torch.long)
        x_dummy = torch.zeros(
            b, self.hparams.target_channels, condition.shape[2], condition.shape[3],
            device=condition.device,
        )
        return self.model(x_dummy, condition, t)

    def configure_optimizers(self):
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.lr*0.5
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

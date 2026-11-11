from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from src.callbacks.timer import log_time

from src.models.base_diffusion import LitConditionalDiffusion

class LitConditionalUNet(LitConditionalDiffusion):
    def __init__(
        self,
        backbone: str = "cond_unet",
        hyperparameters: dict = {},
    ) -> None:
        super().__init__(backbone, hyperparameters)

    @log_time
    def training_step(self, batch, batch_idx):
        x0, condition = self._load_batch(batch)

        b = x0.shape[0]
        t = torch.full((b,), 0, device=self.device, dtype=torch.long)
        x_dummy = torch.zeros(
            b, self.target_channels, condition.shape[2], condition.shape[3],
            device=self.device,
        )

        pred = self.model(x_dummy, condition, t)
        loss = F.mse_loss(pred, x0)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, condition = self._load_batch(batch)

        b = x0.shape[0]
        t = torch.full((b,), 0, device=self.device, dtype=torch.long)
        x_dummy = torch.zeros(
            b, self.target_channels, condition.shape[2], condition.shape[3],
            device=self.device,
        )

        pred = self.model(x_dummy, condition, t)
        loss = F.mse_loss(pred, x0)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    @torch.inference_mode()
    def sample(
        self,
        condition: torch.Tensor,
        num_steps: Optional[int] = None,
        hparams: dict = {},
    ) -> torch.Tensor:
        """
        Samples target images conditioned on `condition`.

        Args:
            condition: [B, condition_channels, H, W]
            num_steps: number of reverse diffusion steps. Defaults to training timesteps.
            hparams: dictionary of additional hyperparameters.

        Returns:
            samples in roughly [-1, 1]
        """
        self.eval()
        b, _, h, w = condition.shape

        t = torch.full((b,), 0, device=self.device, dtype=torch.long)
        x_dummy = torch.zeros(
            b, self.target_channels, condition.shape[2], condition.shape[3],
            device=self.device,
        )
        return self.model(x_dummy, condition, t)

"""
EDM (Karras et al. 2022) training/sampling for CARRA2 SIC super-resolution.

Wraps the shared SuperResUNet backbone in EDMPrecond and dispatches to the
Karras/Heun sampler — same backbone and conditioning as flow matching, so
any difference in results reflects the training objective, not backbone
capacity.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.models.edm.config import TrainConfig
from src.models.edm.base import DiffusionModel
from src.models.edm.losses import EDMLoss
from src.models.edm.precond import EDMPrecond
from src.models.edm.sampler import edm_sampler
from src.models.edm.unet import SuperResUNet


class EDMModel(DiffusionModel):
    model_type = "edm"

    def __init__(self, net: SuperResUNet, cfg: TrainConfig):
        super().__init__(net, cfg)
        self.precond = EDMPrecond(net, sigma_data=cfg.sigma_data)
        self.loss_fn = EDMLoss(P_mean=cfg.p_mean, P_std=cfg.p_std, sigma_data=cfg.sigma_data)

    @classmethod
    def from_config(cls, cfg: TrainConfig) -> "EDMModel":
        net = SuperResUNet(in_channels=24, out_channels=4)
        return cls(net, cfg)

    def training_loss(self, x1: Tensor, cond: Tensor) -> Tensor:
        return self.loss_fn(self.precond, x0=x1, cond=cond)

    @torch.no_grad()
    def sample(self, cond: Tensor, n_steps: int, device: torch.device) -> Tensor:
        return edm_sampler(
            self.precond, cond, num_steps=n_steps, device=device,
            sigma_min=self.cfg.sigma_min, sigma_max=self.cfg.sigma_max, rho=self.cfg.rho,
        )
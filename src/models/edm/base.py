"""
Common interface both training paradigms implement, so train.py /
inference.py / evaluate.py can stay model-agnostic.

`DiffusionModel.load()` is the single path through which a checkpoint is
ever loaded: it reads config.yaml to learn `model_type` and dispatches to
the matching subclass. This replaces the previous setup where the
checkpoint format (a bare SuperResUNet state dict) was identical between
flow matching and EDM but *not* semantically interchangeable — loading one
into the other's sampler used to run silently and produce meaningless
output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import yaml

import torch
import torch.nn as nn
from torch import Tensor

from src.models.edm.unet import SuperResUNet

_REGISTRY: dict[str, type["DiffusionModel"]] = {}


class DiffusionModel(nn.Module, ABC):
    model_type: str

    def __init__(self, net: SuperResUNet, cfg):
        super().__init__()
        self.net = net
        self.cfg = cfg

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "model_type", None):
            _REGISTRY[cls.model_type] = cls

    @classmethod
    @abstractmethod
    def from_config(cls, cfg, device: torch.device) -> "DiffusionModel":
        ...

    @abstractmethod
    def training_loss(self, x1: Tensor, cond: Tensor, land_mask: Tensor) -> Tensor:
        """x1: (B,1,H,W) clean normalised residual. cond: 2-channel static stack
        from src.common.conditioning.build_cond()."""
        ...

    @abstractmethod
    def sample(self, cond: Tensor, n_steps: int, device: torch.device) -> Tensor:
        """Returns the predicted residual/x0 in normalised space, (B,1,H,W)."""
        ...

    @torch.no_grad()
    def sample_denormalised(
        self,
        lr:            Tensor,
        cond:          Tensor,
        n_steps:       int,
        device:        torch.device,
        residual_mean: float,
        residual_std:  float,
        clamp:         bool = True,
    ) -> Tensor:
        """Shared across both models: sample -> denormalise -> add lr -> clamp to [0,1].

        Land pixels get zero loss weight during training (see training_loss),
        so the network has no signal constraining its output there — force
        them to 0 rather than let unconstrained land predictions leak into
        saved/plotted fields.

        clamp=False skips the [0,1] clip. Needed when averaging several
        ensemble members: clamping each member first and then averaging is
        a biased nonlinear operation near the SIC bounds (whichever bound a
        noisy member overshoots gets truncated before it can cancel out in
        the average), so callers computing an ensemble mean should average
        the raw values and clamp once afterward instead.
        """
        residual = self.sample(cond, n_steps, device)
        sic = lr + residual * residual_std + residual_mean
        if clamp:
            sic = sic.clamp(0.0, 1.0)
        land_mask = cond[:, 1:2]
        return sic * land_mask

    def save_checkpoint(self, run_dir: str | Path, name: str) -> None:
        """Always the bare SuperResUNet state dict — same convention for both paradigms."""
        torch.save(self.net.state_dict(), Path(run_dir) / name)
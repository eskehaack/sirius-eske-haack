"""
EDM (Karras et al. 2022) preconditioning wrapper.

Wraps any backbone with a `forward(x, t) -> Tensor` signature — here,
`src.models.unet.SuperResUNet` — with the EDM input/output scaling
from Karras et al. 2022, Table 1. The backbone owns its own time embedding
(sinusoidal + MLP), so this wrapper has no learnable parameters of its own;
it's pure elementwise arithmetic around a single `self.net(...)` call.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class EDMPrecond(nn.Module):
    """
    EDM preconditioning around a conditional denoiser backbone.

    Parameters
    ----------
    net        : backbone with forward(x, t) -> Tensor, e.g. SuperResUNet.
    sigma_data : approximate std of the clean data distribution (residual_norm
                 is ~N(0,1), so sigma_data ≈ 1 is appropriate).
    """

    def __init__(self, net: nn.Module, sigma_data: float = 1.0):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data

    def forward(self, x_t: Tensor, sigma: Tensor, cond: Tensor) -> Tensor:
        """
        x_t   : (B, 1, H, W) noisy residual channel.
        sigma : (B,) noise level.
        cond  : (B, C_cond, H, W) conditioning channels, concatenated in
                natural units (not EDM-scaled) — [lr, land_mask].

        Returns x0_hat : (B, 1, H, W) — denoised prediction.
        """
        B = x_t.shape[0]
        s2  = sigma ** 2
        sd2 = self.sigma_data ** 2

        c_in   = 1.0 / torch.sqrt(s2 + sd2)
        c_skip = sd2 / (s2 + sd2)
        c_out  = sigma * self.sigma_data / torch.sqrt(s2 + sd2)
        # 0.25*log(sigma) is the Karras-paper convention. There's no pretrained
        # embedding to stay compatible with here — the backbone's sinusoidal
        # embedding makes no assumption about input range — so this is a free
        # choice, not a load-bearing constant.
        c_noise = 0.25 * sigma.log()

        x_in   = c_in.view(B, 1, 1, 1) * x_t
        net_in = torch.cat([x_in, cond], dim=1)
        F_x    = self.net(net_in, c_noise)

        return c_skip.view(B, 1, 1, 1) * x_t + c_out.view(B, 1, 1, 1) * F_x
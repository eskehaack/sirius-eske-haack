"""
Model code from Clara Edmonds (https://github.com/claraedmonds/carra2_superres) with modifications from Eske Haack.
Originally formed by by Thea Qvistgaard (https://github.com/TheaQG/CEDDAR)
Based on the EDM model from Karras et al. 2022 (https://arxiv.org/pdf/2206.00364)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from src.callbacks.timer import log_time

from src.models.base_diffusion import LitConditionalDiffusion

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

    def forward(self, x_t: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
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
        F_x    = self.net(x_in, cond, c_noise)

        return c_skip.view(B, 1, 1, 1) * x_t + c_out.view(B, 1, 1, 1) * F_x


class EDMLoss(nn.Module):
    """
    Karras EDM loss: sample sigma ~ logN(P_mean, P_std), perturb x0 -> x_t,
    predict x0_hat with preconditioning, ocean-masked weighted MSE on x0.
    """
    def __init__(self, P_mean: float = -1.2, P_std: float = 1.2, sigma_data: float = 1.0):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def sample_sigma(self, B: int, device: torch.device, dtype=None):
        return (torch.randn(B, device=device, dtype=dtype) * self.P_std + self.P_mean).exp()

    def forward(
        self,
        edm_model: EDMPrecond,
        x0: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        B = x0.shape[0]
        device = x0.device
        dtype = x0.dtype

        sigma = self.sample_sigma(B, device, dtype=dtype)  # [B]

        n = torch.randn_like(x0)
        x_t = x0 + sigma.view(B, 1, 1, 1) * n
        x0_hat = edm_model(x_t, sigma, cond=cond)

        # EDM per-sample loss weight (Karras et al. 2022 eq. 7)
        s2 = sigma**2
        sd2 = self.sigma_data**2
        w = (s2 + sd2) / ((sigma * self.sigma_data)**2)
        w = w.view(B, 1, 1, 1)

        err2 = (x0_hat - x0)**2  # [B, C, H, W]

        return (w * err2).sum() / x0.numel()  # mean over all pixels and channels


class LitEDM(LitConditionalDiffusion):
    def __init__(
        self,
        backbone: str = "super_unet",
        hyperparameters: dict = {},
    ):
        super().__init__(backbone, hyperparameters)

        self.sigma_min = hyperparameters.get("sigma_min", 0.002)
        self.sigma_max = hyperparameters.get("sigma_max", 160.0)
        self.rho = hyperparameters.get("rho", 7.0)

        self.sigma_data = hyperparameters.get("sigma_data", 1.0)
        self.p_mean = hyperparameters.get("p_mean", -0.4)
        self.p_std = hyperparameters.get("p_std", 1.0)

        self.cfg_dropout = hyperparameters.get("cfg_dropout", 0.1)

        self.precond = EDMPrecond(self.model, sigma_data=self.sigma_data)
        self.loss_fn = EDMLoss(P_mean=self.p_mean, P_std=self.p_std, sigma_data=self.sigma_data)

    def _batch_loss(self, batch) -> torch.Tensor:
        residual, cond = batch
        return self.loss_fn(self.precond, x0=residual, cond=cond)
    
    def _get_sigmas_karras(self, n_steps: int, s_min: float, s_max: float, rho: float, device) -> torch.Tensor:
        """Karras et al. 2022 decreasing noise schedule."""
        i = torch.arange(n_steps, device=device, dtype=torch.float32)
        ramp = i / max(n_steps - 1, 1)  # in [0, 1]
        min_inv = s_min ** (1 / rho)
        max_inv = s_max ** (1 / rho)
        return (max_inv + ramp * (min_inv - max_inv)) ** rho

    @log_time
    def training_step(self, batch, batch_idx):
        x0, condition = self._load_batch(batch)

        # CFG dropout — randomly null the condition
        if torch.rand(1).item() < self.cfg_dropout:
            condition = torch.zeros_like(condition)

        loss = self._batch_loss((x0, condition))

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, condition = self._load_batch(batch)
        loss = self._batch_loss((x0, condition))

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def cfg_forward(self, x_t: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        """
        Classifier-free guidance forward pass.

        Parameters
        ----------
        x_t       : (B, 1, H, W) noisy residual channel.
        sigma     : (B,) noise level.
        cond      : (B, C_cond, H, W) conditioning channels.
        cfg_scale : guidance scale factor.

        Returns
        -------
        x0_hat    : (B, 1, H, W) denoised prediction.
        """
        if cfg_scale == 1.0:
            return self.precond(x_t, sigma, cond=cond)

        null_cond = torch.zeros_like(cond)
        mixed_cond = torch.cat([cond, null_cond], dim=0)
        x_t_mixed = torch.cat([x_t, x_t], dim=0)
        sigma_mixed = torch.cat([sigma, sigma], dim=0)

        denoised_mixed = self.precond(x_t_mixed, sigma_mixed, cond=mixed_cond)
        denoised_cond, denoised_uncond = denoised_mixed.chunk(2, dim=0)

        return denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)

    @torch.inference_mode()
    def sample(
        self,
        cond: torch.Tensor,
        num_steps: int,
        hparams: dict = {},
    ) -> torch.Tensor:
        """
        Karras EDM sampler with Heun (2nd-order) updates.

        Expects model(x_t, sigma, cond=cond) -> x0_hat (see src.models.edm.precond.EDMPrecond).

        Parameters
        ----------
        cond : (B, C_cond, H, W) conditioning stack — [lr, land_mask].

        Returns
        -------
        (B, 4, H, W) denoised sample.
        """
        self.eval()
        self.precond.eval()

        device = self.device
        cond = cond.to(device)
        B, _, H, W = cond.shape

        save_metrics: bool = hparams.get("save_metrics", False)

        S_churn: float = hparams.get("S_churn", 0.0)
        S_min: float = hparams.get("S_min", 0.0)
        S_max: float = hparams.get("S_max", float("inf"))
        S_noise: float = hparams.get("S_noise", 1.0)
        cfg_scale: float = hparams.get("cfg_scale", 1.0)

        sigmas = self._get_sigmas_karras(num_steps, self.sigma_min, self.sigma_max, self.rho, device)
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])  # terminal sigma = 0

        x = torch.randn(B, self.target_channels, H, W, device=device) * self.sigma_max

        means = torch.zeros((num_steps, B, 4), device=device)
        stds = torch.zeros((num_steps, B, 4), device=device)

        for i in range(num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            x_in = x
            if S_min <= float(sigma) <= S_max and S_churn > 0:
                gamma = min(S_churn / num_steps, np.sqrt(2.0) - 1.0)
                eps = torch.randn_like(x) * S_noise
                sigma_hat = sigma * (1 + gamma)
                x_in = x + eps * torch.sqrt(sigma_hat**2 - sigma**2)
            else:
                sigma_hat = sigma

            sigma_hat_vec = torch.full((B,), float(sigma_hat), device=device, dtype=x.dtype)
            denoised = self.cfg_forward(x_in, sigma_hat_vec, cond=cond, cfg_scale=cfg_scale)
            d = (x_in - denoised) / sigma_hat

            x_euler = x_in + (sigma_next - sigma_hat) * d
            if i == num_steps - 1:
                x = x_euler
                break

            sigma_next_vec = torch.full((B,), float(sigma_next), device=device, dtype=x.dtype)
            denoised_next = self.cfg_forward(x_euler, sigma_next_vec, cond=cond, cfg_scale=cfg_scale)
            d_next = (x_euler - denoised_next) / sigma_next

            x = x_in + (sigma_next - sigma_hat) * 0.5 * (d + d_next)

            means[i] = x.mean(dim=(2, 3))
            stds[i] = x.std(dim=(2, 3))

        if save_metrics:
            Path("dist_test").mkdir(parents=True, exist_ok=True)
            np.save("dist_test/edm_means.npy", means.detach().cpu().numpy())
            np.save("dist_test/edm_stds.npy", stds.detach().cpu().numpy())

        return x
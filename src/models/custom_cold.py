from __future__ import annotations

from typing import Optional, Sequence
import math

import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from src.callbacks.timer import log_time
from src.models.backbone_unet import ConditionalUNet


class LitCustomCold(pl.LightningModule):
    def __init__(
        self,
        target_channels: int = 1,
        condition_channels: int = 1,
        base_channels: int = 128,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule: str = "linear",
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

        betas, alphas, alpha_bars = self.schedule(
            schedule=schedule,
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bars[:-1]], dim=0)
        sigmas = (1 - alpha_bars_prev) / (1 - alpha_bars) * betas

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
        self.register_buffer("sigmas", sigmas)

        self.static = None  # Allocation
    
    def schedule(self, schedule: str, timesteps: int, beta_start: float, beta_end: float):
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == "cosine":
            s = 0.008
            t = torch.linspace(0, timesteps, timesteps + 1)
            alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, min=0, max=0.02)
        elif schedule == "exponential":
                betas = torch.logspace(torch.log10(beta_start), torch.log10(beta_end), timesteps)
        elif schedule == "laplace":
            mu = 10.0
            b = 10.0
            t = torch.linspace(0, timesteps - 1, timesteps)
            betas = (1 / (2 * b)) * torch.exp(-torch.abs(t / timesteps - mu) / b)
            betas = torch.clamp(betas, min=0, max=0.999)
        elif schedule == "quadratic":
            t = torch.linspace(0, timesteps - 1, timesteps)
            betas = beta_start + (beta_end - beta_start) * (t / (timesteps - 1)) ** 2
        elif schedule == "sigmoid":
            t = torch.linspace(0, timesteps - 1, timesteps)
            betas = beta_start + (beta_end - beta_start) / (
                1 + torch.exp(-12 * (t / (timesteps - 1) - 0.5))
            )
        elif schedule == "step":
            betas = torch.full((timesteps,), beta_start)
            step_size = timesteps // 10
            for i in range(1, 10):
                betas[i * step_size : (i + 1) * step_size] = beta_start + (
                    (beta_end - beta_start) * i / 10
                )
        else:
            raise ValueError(f"Unknown noise schedule: {schedule}")
        
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bars

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]
        return sqrt_ab * x0 + sqrt_omab * noise

    def _rev_qsample(self, x_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]
        return (x_t - sqrt_omab * noise_pred) / sqrt_ab

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
        t = torch.randint(0, self.hparams.timesteps, (b,), device=x0.device)
        cond_target_variables = condition[:, :self.hparams.target_channels, :, :] # Use the condition as noise

        x_t = self.q_sample(x0, t, cond_target_variables)
        cond_pred = self.model(x_t, condition, t)

        loss = F.mse_loss(cond_pred, cond_target_variables)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, condition = self._load_batch(batch)
        
        b = x0.shape[0]
        t = torch.randint(0, self.hparams.timesteps, (b,), device=x0.device)
        cond_target_variables = condition[:, :self.hparams.target_channels, :, :] # Use the condition as noise

        x_t = self.q_sample(x0, t, cond_target_variables)
        cond_pred = self.model(x_t, condition, t)

        loss = F.mse_loss(cond_pred, cond_target_variables)

        x_0_hat = self._rev_qsample(x_t, t, cond_pred)
        recon_loss = F.mse_loss(x_0_hat, x0)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_recon_loss", recon_loss, prog_bar=True, on_epoch=True)
        return loss

    def sample(
        self,
        condition: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
        shape: Optional[tuple[int, int, int, int]] = None,
        sample_type = "ddpm",
    ) -> torch.Tensor:
        if sample_type == "ddpm":
            return self.sample_ddpm(condition, num_steps=num_steps, shape=shape)
        elif sample_type == "ddim":
            return self.sample_ddim(condition, num_steps=num_steps, eta=eta, shape=shape)
        else:
            raise ValueError(f"Unknown sample_type: {sample_type}. Must be 'ddpm' or 'ddim'.")

    @torch.no_grad()
    def sample_ddpm(
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

        # Use the condition as noise
        x = condition[:, :self.hparams.target_channels, :, :] 

        steps = self.hparams.timesteps if num_steps is None else num_steps
        step_indices = torch.linspace(
            self.hparams.timesteps - 1, 0, steps, device=device
        ).long()

        for i, t_idx in enumerate(step_indices):
            t = torch.full((b,), t_idx.item(), device=device, dtype=torch.long)

            sigma_t = self.sigmas[t][:, None, None, None]
            beta_t = self.betas[t][:, None, None, None]
            alpha_t = self.alphas[t][:, None, None, None]
            alpha_bar_t = self.alpha_bars[t][:, None, None, None]

            eps_pred = self.model(x, condition, t)

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
            )

            if t_idx > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(sigma_t) * noise
            else:
                x = mean

        return x

    @torch.no_grad()
    def sample_ddim(
        self,
        condition: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
        shape: Optional[tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:
        """
        Function generated by Claude Sonnet 4.6

        Deterministic (or partially stochastic) DDIM sampler.
    
        DDIM replaces the per-step Gaussian noise injection of DDPM with a
        deterministic update rule derived from an implicit probabilistic model.
        This dramatically reduces the noise accumulation that makes DDPM ensemble
        spread look like flat random noise.
    
        The update rule at each step tau -> tau_prev is:
    
            x0_pred  = (x_tau - sqrt(1 - alpha_bar_tau) * eps) / sqrt(alpha_bar_tau)
            dir_xt   = sqrt(1 - alpha_bar_prev - sigma^2) * eps
            sigma    = eta * sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_tau))
                        * sqrt(1 - alpha_bar_tau / alpha_bar_prev)
            x_prev   = sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise
    
        Args:
            condition:  [B, condition_channels, H, W]  conditioning tensor.
            num_steps:  number of reverse steps (can be << training timesteps).
                        50-200 steps is usually sufficient for DDIM.
            eta:        stochasticity coefficient in [0, 1].
                        eta=0  -> fully deterministic (no noise injected).
                        eta=1  -> recovers DDPM-equivalent variance.
            shape:      optional explicit output shape [B, C, H, W].
    
        Returns:
            Sampled tensor of shape `shape` (or inferred from condition).
        """
        self.eval()
        device = condition.device
        b, _, h, w = condition.shape
    
        if shape is None:
            shape = (b, self.hparams.target_channels, h, w)
    
        T = self.hparams.timesteps
    
        # Build a subsequence of timesteps, evenly spaced from T-1 down to 0.
        # These are the "tau" indices in the DDIM paper.
        tau = torch.linspace(T - 1, 0, num_steps, device=device).long()
    
        # alpha_bar at each tau, and at the previous tau (tau_{i+1} in paper notation,
        # i.e. the *less noisy* end). For the last step, alpha_bar_prev = 1 (clean).
        alpha_bar = self.alpha_bars          # shape [T]
        alpha_bar_tau      = alpha_bar[tau]  # [num_steps]
        # Shift: alpha_bar at the step *before* each tau in the reverse sequence.
        alpha_bar_tau_prev = torch.cat(
            [alpha_bar[tau[1:]], torch.ones(1, device=device, dtype=alpha_bar.dtype)],
            dim=0,
        )  # [num_steps]
    
        # Use the condition as noise
        x = condition[:, :self.hparams.target_channels, :, :] 
    
        for i in range(num_steps):
            t_idx = tau[i].item()
            ab_t    = alpha_bar_tau[i]
            ab_prev = alpha_bar_tau_prev[i]
    
            t_batch = torch.full((b,), t_idx, device=device, dtype=torch.long)
    
            # 1. Predict noise
            eps = self.model(x, condition, t_batch)
    
            # 2. Predict x0 from the current noisy x and predicted noise
            x0_pred = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
    
            # 3. Compute DDIM sigma (controls stochasticity)
            sigma = (
                eta
                * ((1 - ab_prev) / (1 - ab_t)).sqrt()
                * (1 - ab_t / ab_prev).sqrt()
            )
    
            # 4. Direction pointing to x_tau
            dir_xt = (1 - ab_prev - sigma ** 2).clamp(min=0.0).sqrt() * eps
    
            # 5. Update
            if t_idx > 0:
                noise = torch.randn_like(x) if eta > 0.0 else torch.zeros_like(x)
                x = ab_prev.sqrt() * x0_pred + dir_xt + sigma * noise
            else:
                # Final step: return the clean x0 prediction directly
                x = x0_pred
    
        return x

    def configure_optimizers(self):
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.lr
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
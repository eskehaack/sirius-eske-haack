from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

import numpy as np

from src.callbacks.timer import log_time

from src.models.base_diffusion import LitConditionalDiffusion


class LitConditionalDDPM(LitConditionalDiffusion):
    def __init__(
        self,
        backbone: str = "cond_unet",
        hyperparameters: dict = {},
    ):
        super().__init__(backbone, hyperparameters)

        schedule = hyperparameters.get("schedule", "linear")
        timesteps = hyperparameters.get("timesteps", 1000)
        beta_start = hyperparameters.get("beta_start", 1e-4)
        beta_end = hyperparameters.get("beta_end", 2e-2)

        self.schedule(
            schedule=schedule,
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        self.cfg_dropout = hyperparameters.get("cfg_dropout", 0.1)

    @log_time
    def training_step(self, batch, batch_idx):
        x0, condition = self._load_batch(batch)

        # CFG dropout — randomly null the condition
        if torch.rand(1).item() < self.cfg_dropout:
            condition = torch.zeros_like(condition)

        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)

        x_t = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_t, condition, t)

        noise_no_pad = self._crop(noise)
        noise_pred_no_pad = self._crop(noise_pred)

        loss = F.mse_loss(noise_pred_no_pad, noise_no_pad)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, condition = self._load_batch(batch)

        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.device)
        noise = torch.randn_like(x0)

        x_t = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_t, condition, t)

        noise_no_pad = self._crop(noise)
        noise_pred_no_pad = self._crop(noise_pred)

        loss = F.mse_loss(noise_pred_no_pad, noise_no_pad)

        x_0_hat = self._rev_qsample(x_t, t, noise_pred)
        x_0_no_pad = self._crop(x0)
        x_0_hat_no_pad = self._crop(x_0_hat)
        recon_loss = F.mse_loss(x_0_hat_no_pad, x_0_no_pad)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_recon_loss", recon_loss, prog_bar=True, on_epoch=True)
        return loss

    @torch.inference_mode()
    def sample(
        self,
        condition: torch.Tensor,
        num_steps: int = 50,
        hparams: dict = {},
    ) -> torch.Tensor:
        self.eval()
        sample_type = hparams.get("sample_type", "ddpm")
        if sample_type == "ddpm":
            cfg_scale = hparams.get("cfg_scale", 1.0)
            save_metrics = hparams.get("save_metrics", False)
            return self.sample_ddpm(condition, num_steps=num_steps, cfg_scale=cfg_scale, save_metrics=save_metrics)
        elif sample_type == "ddim":
            eta = hparams.get("eta", 0.0)
            return self.sample_ddim(condition, num_steps=num_steps, eta=eta)
        else:
            raise ValueError(f"Unknown sample_type: {sample_type}. Must be 'ddpm' or 'ddim'.")
        
    def sample_ddpm(
        self,
        condition: torch.Tensor,
        num_steps: Optional[int] = None,
        cfg_scale: float = 1.0,
        save_metrics: bool = False
    ) -> torch.Tensor:
        """
        Samples target images conditioned on `condition`.

        Args:
            condition: [B, condition_channels, H, W]
            num_steps: number of reverse diffusion steps. Defaults to training timesteps.
            cfg_scale: scale for classifier-free guidance. Defaults to 1.0.
            shape: optional target shape [B, C, H, W]

        Returns:
            samples in roughly [-1, 1]
        """
        self.eval()

        device = condition.device
        b, _, h, w = condition.shape
        shape = (b, self.target_channels, h, w)
        x = torch.randn(shape, device=device)

        steps = self.timesteps if num_steps is None else num_steps
        step_indices = torch.linspace(
            self.timesteps - 1, 0, steps, device=device
        ).long()

        # Null condition for CFG
        null_condition = torch.zeros_like(condition)

        means = np.zeros((steps, b, self.target_channels))
        stds = np.zeros((steps, b, self.target_channels))

        for i, t_idx in enumerate(step_indices):
            t = torch.full((b,), t_idx.item(), device=device, dtype=torch.long)

            sigma_t = self.sigmas[t][:, None, None, None]
            beta_t = self.betas[t][:, None, None, None]
            alpha_t = self.alphas[t][:, None, None, None]
            alpha_bar_t = self.alpha_bars[t][:, None, None, None]

            noise_pred_cond = self.model(x, condition, t)

            # CFG: combine conditional and unconditional predictions
            if cfg_scale != 1.0:
                noise_pred_uncond = self.model(x, null_condition, t)
                noise_pred = noise_pred_uncond + cfg_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                noise_pred = noise_pred_cond

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * noise_pred
            )

            mean_no_pad = self._crop(mean)
            means[i] = mean_no_pad.mean(dim=[2, 3]).detach().cpu().numpy()
            stds[i] = mean_no_pad.std(dim=[2, 3]).detach().cpu().numpy()

            if t_idx > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise

            else:
                x = mean

        if save_metrics:
            np.save("dist_test/means_black.npy", means)
            np.save("dist_test/stds_black.npy", stds)

        return x

    def sample_ddim(
        self,
        condition: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
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

        Returns:
            Sampled tensor of shape `shape` (or inferred from condition).
        """
        self.eval()
        device = self.device
        b, _, h, w = condition.shape
        shape = (b, self.target_channels, h, w)
    
        T = self.timesteps
    
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
    
        # Start from pure noise
        x = torch.randn(shape, device=device)
    
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
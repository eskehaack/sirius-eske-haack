from __future__ import annotations
from abc import abstractmethod

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from src.models.backbones import BACKBONEREGISTRY

class LitConditionalDiffusion(LightningModule):
    def __init__(
        self,
        backbone: str = "cond_unet",
        hyperparameters: dict = {},
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.target_channels = hyperparameters.get("target_channels", 4)
        self.condition_channels = hyperparameters.get("condition_channels", 20)

        self.model = BACKBONEREGISTRY[backbone](hyperparameters)

        self.training_image_size: tuple[int, int] = hyperparameters['image_size']
        self.lr: float = hyperparameters['lr']
        self.max_epochs: int = hyperparameters['max_epochs']
        self.timesteps: int = hyperparameters.get("timesteps", 1000)

        self.initial_target_shape = (412, 424)
        dh = self.training_image_size[0] - self.initial_target_shape[0]
        dw = self.training_image_size[1] - self.initial_target_shape[1]
        assert dh >= 0 and dw >= 0, "Target shape exceeds image_size"

        pad_top, pad_bottom = dh // 2, dh - dh // 2   # 4, 4
        pad_left, pad_right = dw // 2, dw - dw // 2   # 4, 4
        self.pad_tuple = (pad_left, pad_right, pad_top, pad_bottom)  # (4, 4, 4, 4)

        self.crop_slices = (
            slice(pad_top, pad_top + self.initial_target_shape[0]),    # rows 4:416
            slice(pad_left, pad_left + self.initial_target_shape[1]),  # cols 4:428
        )

        self.static = None  # Allocation

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bars.to(self.device)[t][:, None, None, None]
        sqrt_omab = self.sqrt_one_minus_alpha_bars.to(self.device)[t][:, None, None, None]
        return sqrt_ab * x0 + sqrt_omab * noise

    def _rev_qsample(self, x_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bars.to(self.device)[t][:, None, None, None]
        sqrt_omab = self.sqrt_one_minus_alpha_bars.to(self.device)[t][:, None, None, None]
        return (x_t - sqrt_omab * noise_pred) / sqrt_ab

    def _load_batch(self, batch):
        x, y, static, _ = batch
        device = self.device
        dtype = torch.float32

        # Target: stack the 4 dict entries along channels -> [B, 4, 412, 424]
        x0 = torch.stack(list(y.values()), dim=1).to(device, dtype)
        x0 = F.pad(x0, pad=self.pad_tuple, mode="constant", value=0.0)  # [B, 4, *training_image_size]

        # Condition: interpolate x from 59x131 to 412x424, then pad
        condition = F.interpolate(
            x.to(device, dtype),
            size=self.initial_target_shape,  # (412, 424)
            mode="bilinear",
            align_corners=False,
        )
        condition = F.pad(condition, pad=self.pad_tuple, mode="constant", value=0.0)  # [B, 18, *training_image_size]

        # Static: already 412x424, so only pad. Cache once (assumes static is identical across samples).
        if self.static is None:
            static_single = static[:1].to(device, dtype)  # [1, 2, 412, 424]
            self.static = F.pad(static_single, pad=self.pad_tuple, mode="constant", value=0.0)
        elif self.static.device != device:
            self.static = self.static.to(device)

        static_b = self.static.expand(condition.size(0), -1, -1, -1)  # [B, 2, *training_image_size]
        condition = torch.cat([condition, static_b], dim=1)           # [B, 20, *training_image_size]

        return x0, condition

    def _crop(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.crop_slices[0], self.crop_slices[1]]  # crop to (412, 424)

    def schedule(self, schedule: str, timesteps: int, beta_start: float, beta_end: float) -> None:
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
            alphas = 1.0 - betas

            alpha_bars = torch.cumprod(alphas, dim=0).to(self.device)
            sqrt_alpha_bars = torch.sqrt(alpha_bars).to(self.device)
            sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars).to(self.device)

        elif schedule == "cosine":
            s = 0.008
            t = torch.linspace(0, timesteps, timesteps + 1, device=self.device)
            alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            alphas = 1.0 - betas
            
            alpha_bars = (alphas_cumprod[1:]).to(self.device)
            sqrt_alpha_bars = torch.sqrt(alpha_bars).to(self.device)
            sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars).to(self.device)

            betas = torch.clamp(betas, min=1e-8, max=0.999).to(self.device)

        elif schedule == "laplace":
            mu = 10.0
            b = 10.0
            t = torch.linspace(0, timesteps - 1, timesteps, device=self.device)

            betas = (1 / (2 * b)) * torch.exp(-torch.abs(t / timesteps - mu) / b)
            alphas = 1.0 - betas
            
            alpha_bars = torch.cumprod(alphas, dim=0).to(self.device)
            sqrt_alpha_bars = torch.sqrt(alpha_bars).to(self.device)
            sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars).to(self.device)

            betas = torch.clamp(betas, min=1e-8, max=0.999).to(self.device)

        else:
            raise ValueError(f"Unknown noise schedule: {schedule}")

        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bars[:-1]], dim=0)
        sigmas = ((1 - alpha_bars_prev) / (1 - alpha_bars) * betas).to(self.device)

        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_bars", alpha_bars, persistent=False)
        self.register_buffer("sqrt_alpha_bars", sqrt_alpha_bars, persistent=False)
        self.register_buffer("sqrt_one_minus_alpha_bars", sqrt_one_minus_alpha_bars, persistent=False)
        self.register_buffer("sigmas", sigmas, persistent=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.lr * 0.5
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    @abstractmethod
    def training_step(self, batch, batch_idx):
        ...

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        ...

    @torch.inference_mode()
    @abstractmethod
    def sample(self, cond: torch.Tensor, n_steps: int = 1000, hparams: dict = {}) -> torch.Tensor:
        ...
    
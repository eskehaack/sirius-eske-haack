import torch
import torch.nn as nn
from src.models.edm.precond import EDMPrecond


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
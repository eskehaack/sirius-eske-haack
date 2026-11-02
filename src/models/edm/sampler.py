import math

import torch
import numpy as np


def _get_sigmas_karras(n_steps: int, s_min: float, s_max: float, rho: float, device) -> torch.Tensor:
    """Karras et al. 2022 decreasing noise schedule."""
    i = torch.arange(n_steps, device=device, dtype=torch.float32)
    ramp = i / max(n_steps - 1, 1)  # in [0, 1]
    min_inv = s_min ** (1 / rho)
    max_inv = s_max ** (1 / rho)
    return (max_inv + ramp * (min_inv - max_inv)) ** rho


@torch.no_grad()
def edm_sampler(
    model,
    cond: torch.Tensor,
    num_steps: int,
    device: torch.device | str,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    S_churn: float = 0.0,
    S_min: float = 0.0,
    S_max: float = float("inf"),
    S_noise: float = 1.0,
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
    device = torch.device(device)
    cond = cond.to(device)
    B, _, H, W = cond.shape

    sigmas = _get_sigmas_karras(num_steps, sigma_min, sigma_max, rho, device)
    sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])  # terminal sigma = 0

    x = torch.randn(B, 4, H, W, device=device) * sigma_max

    means = torch.zeros((num_steps, B, 4), device=device)
    stds = torch.zeros((num_steps, B, 4), device=device)

    for i in range(num_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        x_in = x
        if S_min <= float(sigma) <= S_max and S_churn > 0:
            gamma = min(S_churn / num_steps, math.sqrt(2.0) - 1.0)
            eps = torch.randn_like(x) * S_noise
            sigma_hat = sigma * (1 + gamma)
            x_in = x + eps * torch.sqrt(sigma_hat**2 - sigma**2)
        else:
            sigma_hat = sigma

        sigma_hat_vec = torch.full((B,), float(sigma_hat), device=device, dtype=x.dtype)
        denoised = model(x_in, sigma_hat_vec, cond=cond)
        d = (x_in - denoised) / sigma_hat

        x_euler = x_in + (sigma_next - sigma_hat) * d
        if i == num_steps - 1:
            x = x_euler
            break

        sigma_next_vec = torch.full((B,), float(sigma_next), device=device, dtype=x.dtype)
        denoised_next = model(x_euler, sigma_next_vec, cond=cond)
        d_next = (x_euler - denoised_next) / sigma_next

        x = x_in + (sigma_next - sigma_hat) * 0.5 * (d + d_next)

        means[i] = x.mean(dim=(2, 3))
        stds[i] = x.std(dim=(2, 3))

    np.save("dist_test/edm_means.npy", means.detach().cpu().numpy())
    np.save("dist_test/edm_stds.npy", stds.detach().cpu().numpy())

    return x
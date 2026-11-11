"""
Simple 4-level UNet for sea ice super-resolution.

Design notes
------------
- Conditioning is handled purely by channel concatenation at the input.
  Build the channel stack externally (e.g. [x_t, lr_sit, sic, mask, deform])
  and pass the total count as `in_channels`.
- Time / noise-level embedding is injected as an additive channel bias inside
  each ResBlock (easy to upgrade to full FiLM scale+shift later).
- Multi-head self-attention at the bottleneck only (between the two MidBlock
  ResBlocks). At a 1000×1000 crop that is 125×125 = 15.6k tokens, so it relies
  on the fused SDPA kernels to avoid materialising the N×N score matrix.
- GroupNorm throughout — stable for the small batch sizes typical in patch-based
  diffusion training.
- All building blocks are independent classes; swap or extend without touching
  the top-level SuperResUNet.

Spatial flow (default channels [32, 64, 96, 128], 3 downsampling steps)
------------------------------------------------------------------------
  init_conv          in_ch → C0            H × W
  DownBlock 0        C0    → C1,  skip@C1  H × W   → H/2 × W/2
  DownBlock 1        C1    → C2,  skip@C2  H/2      → H/4
  DownBlock 2        C2    → C3,  skip@C3  H/4      → H/8
  MidBlock           C3    → C3            H/8   (ResBlock → Attention → ResBlock)
  UpBlock 0          C3+C3 → C2            H/4
  UpBlock 1          C2+C2 → C1            H/2
  UpBlock 2          C1+C1 → C0            H
  final_conv         C0    → out_ch        H × W
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_groups(channels: int, desired: int = 8) -> int:
    """Largest divisor of `channels` that is <= `desired` (for GroupNorm)."""
    g = min(channels, desired)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


# ---------------------------------------------------------------------------
# Time / noise-level embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for a scalar time / sigma value.

    Accepts t of shape (B,) — either discrete timestep indices or continuous
    noise-level values (e.g. log-sigma).  Output shape: (B, dim).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1)
        )                                   # (half,)
        args = t[:, None].float() * freqs[None, :]   # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)


class CosineAttention(nn.Module):
    """
    From: https://github.com/vinuni-vishc/Few-Shot-Cosine-Transformer
    Presented in: Enhancing Few-Shot Image Classification With Cosine Transformer (https://ieeexplore.ieee.org/document/10190567)
    """
    def __init__(self, dim, heads, dim_head, variant):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.sm = nn.Softmax(dim = -1)
        self.variant = variant
        self.input_linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias = False))
        
        self.output_linear = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def cosine_distance(self, x1, x2):
        '''
        x1      =  [b, h, n, k]
        x2      =  [b, h, k, m]
        output  =  [b, h, n, m]
        '''
        dots = torch.matmul(x1, x2)
        scale = torch.einsum('bhi, bhj -> bhij', 
                (torch.norm(x1, 2, dim = -1), torch.norm(x2, 2, dim = -2)))
        return (dots / scale)
        
    def forward(self, q, k, v):
        f_q, f_k, f_v = map(
            lambda t: self.input_linear(t)
                .reshape(t.shape[0], t.shape[1], self.heads, -1)  # (q, n, h*d) -> (q, n, h, d)
                .permute(2, 0, 1, 3), # (q, n, h, d) -> (h, q, n, d)
            (q, k, v)
        )   
        
        if self.variant == "cosine":
            dots = self.cosine_distance(f_q, f_k.transpose(-1, -2)) # (h, q, n, 1)
            out = torch.matmul(dots, f_v) # (h, q, n, d_h)
        
        else: # self.variant == "softmax"
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale            
            out = torch.matmul(self.sm(dots), f_v)
        
        out = out.permute(1, 2, 0, 3).reshape(out.shape[1], out.shape[2], -1) # (q, n, d)
        return self.output_linear(out)  


class AttentionBlock(nn.Module):
    """
    Multi-head spatial self-attention with a residual connection.

    The output projection is zero-initialised so the block starts as the
    identity and the network initially behaves like the attention-free UNet.

    Parameters
    ----------
    channels  : feature width (must be divisible by `groups`).
    groups    : GroupNorm group count, already resolved by the caller.
    num_heads : attention heads; falls back to 1 if `channels` is not divisible.
    variant   : "cosine" or "normal" attention.
    """

    def __init__(self, channels: int, groups: int, num_heads: int = 4, variant: str = "cosine"):
        super().__init__()
        if channels % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.variant = variant

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        if self.variant == "cosine":
            # Learnable per-head temperature, initialised to 10, clamped at 100.
            self.logit_scale = nn.Parameter(torch.full((num_heads, 1, 1), math.log(10.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = (t.transpose(-1, -2).contiguous() for t in qkv.unbind(1))  # each (B, heads, HW, d)
        
        if self.variant == "cosine":
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            # Fold the per-head temperature into q, since SDPA's `scale` must be a float.
            q = q * self.logit_scale.clamp(max=math.log(100.0)).exp()
            scale = 1.0
        else:
            scale = None

        if x.is_cuda:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                out = F.scaled_dot_product_attention(q, k, v, scale=scale)
        else:
            out = F.scaled_dot_product_attention(q, k, v, scale=scale)

        out = out.transpose(-1, -2).reshape(b, c, h, w)                   # (B, C, H, W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# Core building block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Two-layer residual block with GroupNorm and time-embedding injection.

    Time is projected to `out_channels` scalars and added as a channel bias
    after the first convolution — a lightweight alternative to full FiLM that
    can be upgraded later by replacing the additive term with scale + shift.

    Parameters
    ----------
    in_channels, out_channels : spatial feature widths.
    time_dim : dimension of the time embedding vector fed in at forward time.
    groups   : target number of groups for GroupNorm (reduced automatically
               if `out_channels` is not divisible).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1   = nn.GroupNorm(_valid_groups(in_channels,  groups), in_channels)
        self.conv1   = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2   = nn.GroupNorm(_valid_groups(out_channels, groups), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2   = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act     = nn.SiLU()
        self.skip    = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        # Additive time bias: (B, out_ch) -> (B, out_ch, 1, 1)
        h = h + self.time_proj(self.act(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Encoder / decoder blocks
# ---------------------------------------------------------------------------

class DownBlock(nn.Module):
    """
    Two ResBlocks followed by stride-2 conv downsampling.

    Returns (downsampled_features, skip_connection).
    The skip is taken *before* downsampling so the decoder can recover the
    pre-downsampled spatial detail.
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float):
        super().__init__()
        self.res1      = ResBlock(in_channels,  out_channels, time_dim, dropout=dropout)
        self.res2      = ResBlock(out_channels, out_channels, time_dim, dropout=dropout)
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x    = self.res1(x, t_emb)
        x    = self.res2(x, t_emb)
        skip = x                          # (B, out_ch, H, W)
        x    = self.downsample(x)         # (B, out_ch, H/2, W/2)
        return x, skip


class MidBlock(nn.Module):
    """Bottleneck: ResBlock → self-attention → ResBlock at the lowest resolution."""

    def __init__(self, channels: int, time_dim: int, dropout: float, attention: bool, num_heads: int):
        super().__init__()
        self.attention: bool = attention
        
        self.res1 = ResBlock(channels, channels, time_dim, dropout=dropout)
        if self.attention:
            self.attn = AttentionBlock(channels, _valid_groups(channels), num_heads)
        self.res2 = ResBlock(channels, channels, time_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, t_emb)
        if self.attention:
            x = self.attn(x)
        x = self.res2(x, t_emb)
        return x


class UpBlock(nn.Module):
    """
    Bilinear 2× upsample, skip-connection concatenation, then two ResBlocks.

    Parameters
    ----------
    in_channels   : channels coming from the block below (before upsample).
    skip_channels : channels of the matching encoder skip connection.
    out_channels  : desired output channels.
    """

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, time_dim: int, dropout: float
    ):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.res1 = ResBlock(in_channels + skip_channels, out_channels, time_dim, dropout=dropout)
        self.res2 = ResBlock(out_channels, out_channels, time_dim, dropout=dropout)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x

# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SuperResUNet(nn.Module):
    """
    N-level UNet for conditional super-resolution.

    Conditioning is handled internally by channel concatenation
    forward.  A typical input stack might be::

        x_in = torch.cat([x_t, lr_gcm], dim=1)
        pred = model(x_in, t)

    Parameters
    ----------
    in_channels   : total channels in the pre-stacked conditioning tensor.
    out_channels  : output channels (usually 1 for a scalar residual field).
    base_dim      : channel width at the coarsest encoder level.
    channel_mults : multipliers applied to `base_dim` at each level.
                    Default (1, 2, 3, 4) → [32, 64, 96, 128] with base_dim=32.
    time_dim      : internal dimension of the time-embedding MLP.
    groups        : target GroupNorm group count (reduced per layer if needed).
    """

    def __init__(
        self,
        hyperparameters: dict = {},
    ):
        super().__init__()

        self.target_channels = hyperparameters.get("target_channels", 4)
        self.condition_channels = hyperparameters.get("condition_channels", 20)
        in_channels = self.target_channels + self.condition_channels

        base_channels = hyperparameters.get("base_channels", 32)
        channel_mults = hyperparameters.get("channel_mults", [1, 2, 3, 4])
        time_dim = hyperparameters.get("time_dim", 256)
        groups = hyperparameters.get("groups", 8)
        dropout = hyperparameters.get("dropout", 0.0)
        attention = hyperparameters.get("attention", False)
        num_heads = hyperparameters.get("num_heads", 4)

        dims = [base_channels * m for m in channel_mults]   # e.g. [32, 64, 96, 128]
        n_levels = len(dims) - 1                        # number of down/up pairs

        # Time embedding: sinusoidal positional encoding → 2-layer MLP
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial channel expansion (no spatial change)
        self.init_conv = nn.Conv2d(in_channels, dims[0], 3, padding=1)

        # Encoder
        self.downs = nn.ModuleList([
            DownBlock(dims[i], dims[i + 1], time_dim, dropout)
            for i in range(n_levels)
        ])

        # Bottleneck
        self.mid = MidBlock(
            dims[-1], time_dim, 
            dropout, attention, num_heads
        )

        # Decoder (reversed: deepest first)
        self.ups = nn.ModuleList([
            UpBlock(
                in_channels=dims[n_levels - i],
                skip_channels=dims[n_levels - i],
                out_channels=dims[n_levels - i - 1],
                time_dim=time_dim,
                dropout=dropout
            )
            for i in range(n_levels)
        ])

        # Final projection to output residual
        self.final_norm = nn.GroupNorm(_valid_groups(dims[0], groups), dims[0])
        self.final_conv = nn.Conv2d(dims[0], self.target_channels, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W)  — pre-stacked noisy input + conditioning.
        t : (B,)                    — time steps or noise levels (float).

        Returns
        -------
        (B, out_channels, H, W)  — predicted residual.
        """
        x = torch.cat([x, cond], dim=1)  # (B, in_ch, H, W)
        t_emb = self.time_embed(t)      # (B, time_dim)

        x = self.init_conv(x)           # (B, C0, H, W)

        skips: list[torch.Tensor] = []
        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)

        x = self.mid(x, t_emb)

        for up in self.ups:
            x = up(x, skips.pop(), t_emb)

        x = self.final_conv(F.silu(self.final_norm(x)))
        return x
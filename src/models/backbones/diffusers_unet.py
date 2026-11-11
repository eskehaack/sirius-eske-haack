import torch

from diffusers import UNet2DModel

class ClimateUNet(torch.nn.Module):
    """
    Wraps diffusers UNet2DModel to support channel-concatenation conditioning.

    The noisy target (4 channels) and condition (20 channels) are concatenated
    along the channel dimension before being passed to the UNet. The UNet is
    configured with attention at the bottleneck and intermediate resolutions,
    replacing the attention-free custom UNet.

    Args:
        target_channels:    Number of target output channels (default: 4).
        condition_channels: Number of conditioning input channels (default: 20).
        base_channels:      Base channel width (default: 128; wider than before
                            to compensate for removing explicit skip connections
                            from the old architecture).
        channel_mults:      Channel multipliers per resolution level.
        attention_levels:   Which resolution levels get self-attention.
                            0 = highest resolution, len-1 = bottleneck.
        num_heads:          Number of attention heads.
        dropout:            Dropout probability inside ResBlocks.
    """

    def __init__(
        self,
        hyperparameters: dict = {},
    ):
        super().__init__()

        self.target_channels = hyperparameters.get("target_channels", 4)
        self.condition_channels = hyperparameters.get("condition_channels", 20)
        in_channels = self.target_channels + self.condition_channels

        base_channels = hyperparameters.get("base_channels", 64)
        channel_mults = hyperparameters.get("channel_mults", [1, 2, 4, 8])
        block_out_channels = tuple(base_channels * m for m in channel_mults)

        # Attention is enabled at the levels listed in attention_levels.
        # diffusers expects a tuple of booleans, one per resolution level.
        num_heads = hyperparameters.get("num_heads", 8)
        attention_levels = hyperparameters.get("attention_levels", [])
        attention_head_dim = block_out_channels[-1] // num_heads
        down_block_types = []
        up_block_types = []
        for i in range(len(channel_mults)):
            if i in attention_levels:
                down_block_types.append("AttnDownBlock2D")
                up_block_types.insert(0, "AttnUpBlock2D")
            else:
                down_block_types.append("DownBlock2D")
                up_block_types.insert(0, "UpBlock2D")

        self.unet = UNet2DModel(
            in_channels=in_channels,
            out_channels=self.target_channels,
            block_out_channels=block_out_channels,
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
            attention_head_dim=attention_head_dim,
            norm_num_groups=8,
            dropout=hyperparameters.get("dropout", 0.0),
            downsample_type="resnet",
            upsample_type="resnet",
            # Time embedding is handled internally by diffusers
        )

    def forward(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t:       Noisy target,  [B, target_channels, H, W]
            condition: Condition,     [B, condition_channels, H, W]
            t:         Timesteps,     [B]
        Returns:
            Predicted noise,          [B, target_channels, H, W]
        """
        x = torch.cat([x_t, condition], dim=1)
        return self.unet(x, t).sample
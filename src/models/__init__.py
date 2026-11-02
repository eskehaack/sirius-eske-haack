from .backbone_unet import ConditionalUNet
from .cond_unet import LitConditionalUNet
from .ddpm import LitConditionalDDPM
from .cold_diffusion import LitColdDiffusion
from .custom_cold import LitCustomCold
from .diffusers_ddpm import LitClimateDiffusion
from .edm.lit_edm import LitEDM

__all__ = [
    "ConditionalUNet",
    "LitConditionalUNet",
    "LitConditionalDDPM",
    "LitColdDiffusion",
    "LitCustomCold",
    "LitClimateDiffusion",
    "LitEDM",
]

MODELREGISTRY = {
    "conditional_ddpm": LitConditionalDDPM,
    "conditional_unet": LitConditionalUNet,
    "cold_diffusion": LitColdDiffusion,
    "custom_cold": LitCustomCold,
    "diffusers_ddpm": LitClimateDiffusion,
    "edm": LitEDM,
}
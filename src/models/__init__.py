from .backbones.cond_unet import ConditionalUNet
from .cond_unet import LitConditionalUNet
from .ddpm import LitConditionalDDPM
from .cold_diffusion import LitColdDiffusion
from .custom_cold import LitCustomCold
from .diffusers_ddpm import LitClimateDiffusion
from .lit_edm import LitEDM
from .lit_edm2 import LitEDM2

__all__ = [
    "ConditionalUNet",
    "LitConditionalUNet",
    "LitConditionalDDPM",
    "LitColdDiffusion",
    "LitCustomCold",
    "LitClimateDiffusion",
    "LitEDM",
    "LitEDM2",
]

MODELREGISTRY = {
    "conditional_ddpm": LitConditionalDDPM,
    "conditional_unet": LitConditionalUNet,
    "cold_diffusion": LitColdDiffusion,
    "custom_cold": LitCustomCold,
    "diffusers_ddpm": LitClimateDiffusion,
    "edm": LitEDM,
    "edm2": LitEDM2,
}
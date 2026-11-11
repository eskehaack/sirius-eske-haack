from .cond_unet import ConditionalUNet
from .diffusers_unet import ClimateUNet
from .super_unet import SuperResUNet

__all__ = [
    "ConditionalUNet",
    "SuperResUNet",
    "ClimateUNet"
]

BACKBONEREGISTRY = {
    "conditional_unet": ConditionalUNet,
    "super_unet": SuperResUNet,
    "climate_unet": ClimateUNet,
}

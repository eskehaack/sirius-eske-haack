"""
Run configuration for CARRA2 SIC super-resolution training/inference/evaluation.

One `TrainConfig` is built per run (CLI args -> dataclass), saved as
`saved_models/<experiment_name>/config.yaml`, and reloaded by
`src.inference`, `src.evaluate`, and `DiffusionModel.load()` so the exact
settings that produced a checkpoint are always recoverable.
"""

import argparse
import dataclasses
import types
import typing
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml

# Single source of truth — src/data/dataset.py imports these rather than
# redefining them.
YEAR_MIN = 2012
YEAR_MAX = 2017

MODEL_TYPES = ("flow_matching", "edm", "deterministic", "flow_matching_residual")

# Per-model defaults for fields whose "currently working" value differs
# between the two training paradigms (see train.py / train_edm.py history).
# Left as None on the dataclass and resolved here so a fresh TrainConfig()
# doesn't silently pick one model's tuning for the other.
_MODEL_DEFAULTS: dict[str, dict[str, object]] = {
    "flow_matching":          {"lr": 3e-4, "epochs": 50, "weight_decay": 0.0, "use_cosine_schedule": False},
    "edm":                    {"lr": 3e-4, "epochs": 50, "weight_decay": 0.0, "use_cosine_schedule": False},
    "deterministic":          {"lr": 3e-4, "epochs": 50, "weight_decay": 0.0, "use_cosine_schedule": False},
    "flow_matching_residual": {"lr": 3e-4, "epochs": 50, "weight_decay": 0.0, "use_cosine_schedule": False},
}


@dataclass
class DataConfig:
    train_zarr: str = "data/pre_processed_2/train.zarr"
    val_zarr: str = "data/pre_processed_2/validate.zarr"
    stats_npz: str = "data/pre_processed_2/train_stats.npz"
    # Stats for the flow_matching_residual model's second-stage target
    # (r2 = x1 - det_pred). Computed by src.data.compute_stage2_stats.
    # Required (non-None) only when model_type == "flow_matching_residual".
    stage2_stats_npz: str | None = None

    train_crop_size: tuple[int, int] = (1000, 1000)
    train_crop_y0:   int = 0
    train_crop_x0:   int = 0

    # None = fall back to the matching train_crop_* value — see
    # resolved_inference_crop(). 
    inference_crop_size: tuple[int, int] | None = None
    inference_crop_y0:   int | None = None
    inference_crop_x0:   int | None = None

    ice_mask_threshold: float = 0.15
    year_min: int = YEAR_MIN
    year_max: int = YEAR_MAX

    def resolved_inference_crop(self) -> tuple[tuple[int, int], int, int]:
        """(crop_size, y0, x0) to use for eval/inference, falling back to the
        train crop for any field left unset."""
        size = self.inference_crop_size if self.inference_crop_size is not None else self.train_crop_size
        y0   = self.inference_crop_y0   if self.inference_crop_y0   is not None else self.train_crop_y0
        x0   = self.inference_crop_x0   if self.inference_crop_x0   is not None else self.train_crop_x0
        return size, y0, x0


@dataclass
class TrainConfig:
    experiment_name: str
    model_type: Literal["flow_matching", "edm", "deterministic", "flow_matching_residual"] = "flow_matching"

    # flow_matching_residual only: run dir of the frozen stage-1 DeterministicModel.
    # Required (non-None) only when model_type == "flow_matching_residual".
    stage1_run_dir: str | None = None

    # Resolved per model_type at construction time if left as None — see
    # _MODEL_DEFAULTS above.
    epochs: int | None = None
    lr: float | None = None
    weight_decay: float | None = None
    use_cosine_schedule: bool | None = None

    batch_size: int = 4
    grad_clip: float = 1.0
    seed: int = 0
    num_workers: int = 0
    n_infer_steps: int = 20
    require_gpu: bool = False

    # flow-matching-only (ignored when model_type == "edm")
    edge_threshold: float = 0.05

    # EDM-only (ignored when model_type == "flow_matching")
    sigma_data: float = 1.0
    p_mean: float = -1.2
    p_std: float = 1.2
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0

    data: DataConfig = field(default_factory=DataConfig)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def __post_init__(self) -> None:
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {self.model_type!r}")
        defaults = _MODEL_DEFAULTS[self.model_type]
        for key, val in defaults.items():
            if getattr(self, key) is None:
                setattr(self, key, val)


# ---------------------------------------------------------------------------
# YAML persistence
# ---------------------------------------------------------------------------

def save_config(cfg: TrainConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(dataclasses.asdict(cfg), fh, sort_keys=False)


def load_config(path: str | Path) -> TrainConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    data_raw = dict(raw.pop("data"))
    data_raw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in data_raw.items()}
    data_known = {f.name for f in fields(DataConfig)} # silently drop stale keys
    data_raw = {k: v for k, v in data_raw.items() if k in data_known}
    train_known = {f.name for f in fields(TrainConfig)}
    raw = {k: v for k, v in raw.items() if k in train_known}
    return TrainConfig(data=DataConfig(**data_raw), **raw)


# ---------------------------------------------------------------------------
# CLI: one --field flag per TrainConfig/DataConfig field, introspected from
# the dataclasses so new fields don't need a matching hand-written flag.
# ---------------------------------------------------------------------------

def _str2bool(s: str) -> bool:
    if s.lower() in ("1", "true", "yes", "y"):
        return True
    if s.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {s!r}")


def _add_field_arg(
    parser: argparse.ArgumentParser,
    name: str,
    ftype: object,
    default: object,
    required: bool = False,
) -> None:
    flag = f"--{name}"

    # Unwrap `X | None` (used for per-model-resolved fields).
    if isinstance(ftype, types.UnionType):
        non_none = [t for t in typing.get_args(ftype) if t is not type(None)]
        ftype = non_none[0] if non_none else str

    origin = typing.get_origin(ftype)
    if origin is tuple:
        elem_type = typing.get_args(ftype)[0]
        parser.add_argument(flag, type=elem_type, nargs=len(typing.get_args(ftype)),
                             default=default, required=required)
    elif origin is Literal:
        parser.add_argument(flag, type=str, choices=typing.get_args(ftype),
                             default=default, required=required)
    elif ftype is bool:
        parser.add_argument(flag, type=_str2bool, default=default, required=required)
    else:
        parser.add_argument(flag, type=ftype, default=default, required=required)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/load a CARRA2 SIC super-resolution model (flow matching or EDM)."
    )
    for f in fields(TrainConfig):
        if f.name in ("data", "created_at"):
            continue
        required = f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        default = None if required else (
            f.default if f.default is not dataclasses.MISSING else f.default_factory()
        )
        _add_field_arg(parser, f.name, f.type, default, required=required)
    for f in fields(DataConfig):
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        _add_field_arg(parser, f"data_{f.name}", f.type, default)
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    data_kwargs = {}
    for f in fields(DataConfig):
        val = getattr(args, f"data_{f.name}")
        data_kwargs[f.name] = tuple(val) if isinstance(val, list) else val

    train_kwargs = {}
    for f in fields(TrainConfig):
        if f.name in ("data", "created_at"):
            continue
        train_kwargs[f.name] = getattr(args, f.name)

    return TrainConfig(data=DataConfig(**data_kwargs), **train_kwargs)


def parse_train_config(argv: list[str] | None = None) -> tuple[TrainConfig, bool]:
    """Parse CLI args into (cfg, overwrite) — overwrite is a run-mode flag,
    not a hyperparameter, so it isn't part of TrainConfig itself."""
    parser = build_arg_parser()
    parser.add_argument("--overwrite", action="store_true",
                         help="Allow reusing an existing experiment_name directory.")
    args = parser.parse_args(argv)
    return config_from_args(args), args.overwrite
"""
The original author of this code is: Mikhail Ivanov (https://github.com/mikhail-a-ivanov) as part of his work for the Swedish Meteorological and Hydrological Institute (SMHI). 
The code has been modified by the current author to suit specific needs, under the approval of Mikhail Ivanov.
"""

import os
from pathlib import Path
import toml
import torch
import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset

from src.data_builders import data_utils as d_utils

class ClimateDataBuilder:
    def __init__(
        self,
        date_config,
        preprocessing_config,
        predictor_config,
        regridded_predictor_config=None,
        static_features_config=None,
        target_config=None
    ):
        self.date_config = date_config
        self.preprocessing_config = preprocessing_config
        self.predictor_config = predictor_config
        self.regridded_predictor_config = regridded_predictor_config
        self.static_features_config = static_features_config
        self.target_config = target_config
        
        self.predictors = None
        self.targets = None
        self.static_features = None
        self.train_idx = None
        self.val_idx = None
        self.infer_idx = None
        self.stats = None

    def load_set(self, config):
        datasets = []
        timesteps_per_chunk = 'auto' # Seems to be optimal for preprocessing
        vertical_dim_name = config.get("vertical_dim_name", "plev")
        for source_name, source_config in config["source"].items():
            for scenario_name, scenario_config in source_config["scenario"].items():
                for member_name, member_config in scenario_config["member"].items():
                    basepath = member_config["basepath"]
                    files = member_config["files"]
                    mode = member_config["mode"]
                    ds = d_utils.merge_dataset(
                        basepath, 
                        files, 
                        mode,
                        timesteps_per_chunk,
                        vertical_dim_name
                    )
                    realization_id = f"{source_name}|{scenario_name}|{member_name}"
                    ds = ds.expand_dims(realization=[realization_id])
                    ds = self._normalize_time(ds)
                    datasets.append(ds)
                    
        return datasets

    def load_predictors(self):
        return self.load_set(self.predictor_config)
    
    def load_regridded_predictors(self):
        return self.load_set(self.regridded_predictor_config)

    def load_targets(self):
        return self.load_set(self.target_config)

    def load_static_features(self):
        if self.static_features_config is None:
            return None

        mode = self.static_features_config["mode"]

        ds = d_utils.merge_dataset(
            self.static_features_config["basepath"],
            self.static_features_config["files"],
            mode,
            None
        )
        return ds

    def _normalize_time(self, ds):
        if self.date_config.get("daily_frequency", False):
            ds = ds.assign_coords(time=ds.time.dt.floor("D"))
        return ds

    def align(self, predictors, targets):
        if targets is None:
            return predictors, None

        predictors, targets = xr.align(
            predictors,
            targets,
            join="inner",
        )

        if not predictors.time.equals(targets.time):
            raise ValueError("Time mismatch after alignment")

        return predictors, targets

    def _residualize_targets(self):
        if self.targets is None:
            print("No targets provided, skipping residualization.")
            return
        
        if self.preprocessing_config["residualize_targets"]:
            regridded_predictors = self.load_regridded_predictors()
            regridded_predictors = xr.concat(
                [regridded.squeeze("realization", drop=True) for regridded in regridded_predictors],
                dim='time'
            )
            for var in self.targets.data_vars:
                self.targets[var] = self.targets[var] - regridded_predictors[var]
        else:
            print("Residualization of targets is disabled.")
            
    def _get_global_stats(self):
        path = self.preprocessing_config["normalization_stats_path"]
        if os.path.isfile(path):
            print(f"Loading global stats from {path}")
            stats = xr.open_dataset(path)
    
            self.stats = stats
            return

        print(f"Computing global stats from the training predictors...")
        predictors_training = self.predictors.isel(time=self.train_idx)
        targets_training = self.targets.isel(time=self.train_idx)
        
        stats = xr.Dataset(
            {
                var: xr.concat(
                    [
                        predictors_training[var].mean(),
                        predictors_training[var].std(),
                        targets_training[var].mean() if var in targets_training.data_vars else xr.DataArray(np.nan),
                        targets_training[var].std()  if var in targets_training.data_vars else xr.DataArray(np.nan),
                    ],
                    dim="statistic",
                    coords="minimal",
                ).assign_coords(statistic=["pred_mean", "pred_std", "targ_mean", "targ_std"])
                for var in predictors_training.data_vars
            }
        )
    
        print(f"Saving global stats to {path}")
        Path(path).absolute().parent.mkdir(parents=True, exist_ok=True)
        stats.to_netcdf(path)
        self.stats = stats

    def _global_standardization(self):
        excluded = {"pr_mask"}
        means = xr.Dataset({var: self.stats[var].sel(statistic="pred_mean") for var in self.stats.data_vars})
        stds  = xr.Dataset({var: self.stats[var].sel(statistic="pred_std") for var in self.stats.data_vars})

        self.predictors = (self.predictors - means) / stds

        if self.targets is not None:
            means = xr.Dataset({var: self.stats[var].sel(statistic="targ_mean") for var in self.stats.data_vars})
            stds  = xr.Dataset({var: self.stats[var].sel(statistic="targ_std") for var in self.stats.data_vars})
            target_vars = [var for var in self.targets.data_vars if var not in excluded]
    
            missing = set(target_vars) - set(self.predictors.data_vars)
            assert not missing, f"Missing predictors for targets: {missing}"

            self.targets.update(
                (self.targets[target_vars] - means[target_vars]) /
                stds[target_vars]
            )

    def _scale_static_features(self):
        self.static_features = self.static_features.fillna(0)
        mins = self.static_features.min()
        maxs = self.static_features.max()
        self.static_features = (self.static_features - mins) / (maxs - mins)

    def _check_static_alignment(self):
        if self.targets is None or self.static_features is None:
            return
    
        spatial_dims = [
            dim for dim in self.static_features.dims
            if dim in self.targets.dims
        ]
    
        for dim in spatial_dims:
            assert self.targets.sizes[dim] == self.static_features.sizes[dim], (
                f"Dimension '{dim}' has different size "
                f"({self.targets.sizes[dim]} vs {self.static_features.sizes[dim]})"
            )
        
    def preprocessing(self):
        self._residualize_targets()
        self._check_static_alignment()
        self._scale_static_features()
        if self.preprocessing_config["normalization"] == 'global':
            self._get_global_stats()
            self._global_standardization()
        else:
            print(f"No normalization is applied.")
            
    def build_training_set(self):
        print(f"Loading predictors...")
        predictors_list = self.load_predictors()
        print(f"Loading targets...")
        targets_list = self.load_targets()
        print(f"Loading static features...")
        static_features = self.load_static_features()
        self.static_features = static_features

        predictors_list_aligned = []
        targets_list_aligned = []

        print(f"Aligning the time coordinate...")
        assert len(predictors_list) == len(targets_list), "Mismatch predictors/targets realizations"
        for predictors, targets in zip(predictors_list, targets_list):
            predictors_aligned, targets_aligned = self.align(predictors, targets)
            predictors_list_aligned.append(predictors_aligned)
            targets_list_aligned.append(targets_aligned)

        print(f"Concatenating the predictors...")
        predictors_combined = xr.concat(
            [predictors.squeeze("realization", drop=True) for predictors in predictors_list_aligned],
            dim='time'
        )
        self.predictors = predictors_combined

        print(f"Concatenating the targets...")
        targets_combined = xr.concat(
            [targets.squeeze("realization", drop=True) for targets in targets_list_aligned],
            dim='time'
        )
        self.targets = targets_combined

        print(f"Building the training indices...")
        train_idx = d_utils.build_indices(
            pd.to_datetime(self.predictors.time.values),
            self.date_config["train_blocks"]
        )
        self.train_idx = train_idx
        print("Training indices:")
        print("shape", train_idx.shape)
        print("max", train_idx.max())
        print("min", train_idx.min())
        print("blocks", self.date_config["train_blocks"])

        print(f"Building the validation indices...")
        val_idx = d_utils.build_indices(
            pd.to_datetime(self.predictors.time.values),
            self.date_config["val_blocks"]
        )
        self.val_idx = val_idx

        print(f"Building the inference indices...")
        infer_idx = d_utils.build_indices(
            pd.to_datetime(self.predictors.time.values),
            self.date_config["inference_blocks"]
        )
        self.infer_idx = infer_idx

        print(f"Preprocessing the data...")
        self.preprocessing()

    def write_preprocessed_data(self, indices):
        cfg = self.preprocessing_config
        predictors_output = cfg.get("predictors_output")
        targets_output = cfg.get("targets_output")
        static_output = cfg.get("static_output")

        chunk_t = cfg.get("timesteps_per_chunk", 8)
        shard_t = cfg.get("timesteps_per_shard", 256)
        if shard_t % chunk_t != 0:
            raise ValueError(f"timesteps_per_shard ({shard_t}) must be a multiple of "
                             f"timesteps_per_chunk ({chunk_t})")

        if static_output and not os.path.isdir(static_output):
            print(f"Writing static data to {static_output}")
            static = self.static_features
            if isinstance(static, xr.DataArray):
                static = static.to_dataset(name=static.name or "data")
            for var in static.variables.values():
                var.encoding = {}
            static.to_zarr(static_output, zarr_format=3, mode="w", consolidated=False)

        for label, obj, out in (("predictors", self.predictors, predictors_output),
                                ("targets", self.targets, targets_output)):
            if obj is None or not out or os.path.isdir(out):
                continue
            print(f"Writing {label} to {out} (chunk={chunk_t}, shard={shard_t} timesteps)")
            ds, encoding = d_utils._to_sharded_dataset(obj, indices, chunk_t, shard_t)
            ds.to_zarr(out, zarr_format=3, mode="w", encoding=encoding, consolidated=False)

class TrainingDataset(Dataset):
    def __init__(self, predictors_path, static_features_path, targets_path, blocks):
        self.predictors = xr.open_dataarray(predictors_path)
        self.predictors_variables = list(self.predictors.channel.values)
        predictors_spatial_dims = [
            dim for dim in self.predictors.dims
            if dim not in ("time", "channel")
        ]
        self.predictors_grid = tuple(
            [len(self.predictors[dim]) for dim in predictors_spatial_dims]
        )
        
        self.targets = xr.open_dataarray(targets_path)
        self.target_variables = list(self.targets.channel.values)
        
        static_features = xr.open_dataset(static_features_path)
        self.static_variables = list(static_features.data_vars)
        self.static_features = torch.from_numpy(
            static_features.to_array().values.astype(np.float32)
        )
        self.target_grid = tuple(static_features.sizes.values())

        indices = d_utils.build_indices(pd.to_datetime(self.predictors.time.values), blocks)
        self.indices = np.array(indices)
        self.length = len(self.indices)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        t = self.indices[idx]
        x = torch.from_numpy(
            self.predictors.isel(time=t).values.astype(np.float32)
        )
        y = torch.from_numpy(
            self.targets.isel(time=t).values.astype(np.float32)
        )
        y = {
            var: y[i]
            for i, var in enumerate(self.target_variables)
        }

        return x, y, self.static_features, idx


def parse_config(config_keyword="input", config_path="config.toml"):
    """Parses and returns TOML input config.
    
    Example for input_config fields:

    path = input_config["path"]
    files = input_config["files"]
    levels = input_config["levels"]
    """
    try:
        with open(config_path, 'r') as config_file:
            config = toml.load(config_file)
        
        return config[config_keyword]
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
    except toml.TomlDecodeError:
        print(f"Error: Failed to parse the configuration file '{config_path}'. Ensure it's a valid TOML file.")

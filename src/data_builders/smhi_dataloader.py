import os
import glob
from pathlib import Path
import toml
import torch
import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset
import torch.nn.functional as F

class ClimateDataBuilder:
    def __init__(
        self,
        date_config,
        preprocessing_config,
        predictor_config,
        static_features_config=None,
        target_config=None
    ):
        self.date_config = date_config
        self.preprocessing_config = preprocessing_config
        self.predictor_config = predictor_config
        self.static_features_config = static_features_config
        self.target_config = target_config
        
        self.predictors = None
        self.targets = None
        self.static_features = None
        self.train_idx = None
        self.val_idx = None
        self.infer_idx = None
        self.stats = None

    def load_predictors(self):
        datasets = []
        #timesteps_per_chunk = self.preprocessing_config.get("timesteps_per_chunk")
        timesteps_per_chunk = 'auto' # Seems to be optimal for preprocessing
        vertical_dim_name = self.predictor_config["vertical_dim_name"]
        for source_name, source_config in self.predictor_config["source"].items():
            for scenario_name, scenario_config in source_config["scenario"].items():
                for member_name, member_config in scenario_config["member"].items():
                    basepath = member_config["basepath"]
                    files = member_config["files"]
                    mode = member_config["mode"]
                    ds = merge_dataset(
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

    def load_targets(self):
        if self.target_config is None:
            return None

        datasets = []
        #timesteps_per_chunk = self.preprocessing_config.get("timesteps_per_chunk")
        timesteps_per_chunk = 'auto' # Seems to be optimal for preprocessing
        for source_name, source_config in self.target_config["source"].items():
            for scenario_name, scenario_config in source_config["scenario"].items():
                for member_name, member_config in scenario_config["member"].items():
                    basepath = member_config["basepath"]
                    files = member_config["files"]
                    mode = member_config["mode"]
                    ds = merge_dataset(
                        basepath, 
                        files, 
                        mode,
                        timesteps_per_chunk,
                    )
                    realization_id = f"{source_name}|{scenario_name}|{member_name}"
                    ds = ds.expand_dims(realization=[realization_id])
                    ds = self._normalize_time(ds)
                    datasets.append(ds)
                    
        return datasets

    def load_static_features(self):
        if self.static_features_config is None:
            return None

        mode = self.static_features_config["mode"]

        ds = merge_dataset(
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

    def _apply_temperature_constraints(self):
        if self.preprocessing_config["temperature_constraints"]:
            temperatures_vars = ['tas', 'tasmin', 'tasmax']
            for var in temperatures_vars:
                assert var in self.predictors.data_vars, f"{var} is missing in the predictor set"
                if self.targets:
                    assert var in self.targets.data_vars, f"{var} is missing in the target set"
                    
            self.predictors['tas_diff_min'] = self.predictors['tas'] - self.predictors['tasmin']
            self.predictors['tas_diff_max'] = self.predictors['tasmax'] - self.predictors['tas']
            self.predictors = self.predictors.drop_vars(['tas', 'tasmax'])

            if self.targets:
                self.targets['tas_diff_min'] = self.targets['tas'] - self.targets['tasmin']
                self.targets['tas_diff_max'] = self.targets['tasmax'] - self.targets['tas']
                self.targets = self.targets.drop_vars(['tas', 'tasmax'])
        else:
            print(f"No temperature constraints applied.")

    def _precipitation_preprocessing(self):
        if self.preprocessing_config["precipitation_transform"] == 'log1p':
            assert 'pr' in self.predictors, 'pr variable is missing in the predictor set'
            pr_scaling = float(self.preprocessing_config["precipitation_scaling"])
            wet_day_threshold = float(self.preprocessing_config["wet_day_threshold"])
            self.predictors['pr_log_mm_day'] = np.log1p(self.predictors['pr'] * pr_scaling)
            self.predictors = self.predictors.drop_vars(['pr'])
            if self.targets:
                assert 'pr' in self.targets, 'pr variable is missing in the target set'
                self.targets['pr'] *= pr_scaling
                if wet_day_threshold > 0:
                    self.targets['pr_mask'] = self.targets['pr'] >= wet_day_threshold
                self.targets['pr_log_mm_day'] = np.log1p(self.targets['pr'])
                self.targets = self.targets.drop_vars(['pr'])
        else:
            print(f'No precipitation preprocessing applied.')

    def _get_global_stats(self):
        path = self.preprocessing_config["normalization_stats_path"]
        if os.path.isfile(path):
            print(f"Loading global stats from {path}")
            stats = xr.open_dataset(path)
            missing = set(self.predictors.data_vars) - set(stats.data_vars)
            assert not missing, f"Missing data variables: {missing}"
    
            self.stats = stats
            return

        print(f"Computing global stats from the training predictors...")
        predictors_training = self.predictors.isel(time=self.train_idx)
        
        stats = xr.Dataset(
            {
                var: xr.concat(
                    [
                        predictors_training[var].mean(),
                        predictors_training[var].std(),
                    ],
                    dim="statistic",
                ).assign_coords(statistic=["mean", "std"])
                for var in predictors_training.data_vars
            }
        )
    
        print(f"Saving global stats to {path}")
        Path(path).absolute().parent.mkdir(parents=True, exist_ok=True)
        stats.to_netcdf(path)
        self.stats = stats

    def _global_standardization(self):
        excluded = {"pr_mask"}
        means = xr.Dataset({var: self.stats[var].sel(statistic="mean") for var in self.stats.data_vars})
        stds  = xr.Dataset({var: self.stats[var].sel(statistic="std") for var in self.stats.data_vars})

        self.predictors = (self.predictors - means) / stds

        if self.targets is not None:
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
        self._apply_temperature_constraints()
        self._precipitation_preprocessing()
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
        train_idx = build_indices(
            self.predictors,
            self.date_config["train_blocks"]
        )
        self.train_idx = train_idx
        print("Training indices:")
        print("shape", train_idx.shape)
        print("max", train_idx.max())
        print("min", train_idx.min())
        print("blocks", self.date_config["train_blocks"])

        print(f"Building the target indices...")
        val_idx = build_indices(
            self.predictors,
            self.date_config["val_blocks"]
        )
        self.val_idx = val_idx

        print(f"Preprocessing the data...")
        self.preprocessing()

        print("Training indices...... again:")
        print("shape", train_idx.shape)
        print("max", train_idx.max())
        print("min", train_idx.min())
        print("blocks", self.date_config["train_blocks"])

    def build_inference_set(self):
        print(f"Loading predictors...")
        predictors_list = self.load_predictors()
        print(f"Loading static features...")
        static_features = self.load_static_features()
        self.static_features = static_features

        print(f"Concatenating the predictors...")
        predictors_combined = xr.concat(
            [predictors.squeeze("realization", drop=True) for predictors in predictors_list],
            dim='time'
        )
        self.predictors = predictors_combined

        print(f"Building the inference indices...")
        infer_idx = build_indices(
            self.predictors,
            self.date_config["inference_blocks"]
        )
        self.infer_idx = infer_idx
        
        print(f"Preprocessing the data...")
        self.preprocessing()

    def write_preprocessed_data(self, indices):
        predictors_output = self.preprocessing_config.get("predictors_output")
        targets_output = self.preprocessing_config.get("targets_output")
        static_output = self.preprocessing_config.get("static_output")

        timesteps_per_chunk = self.preprocessing_config.get("timesteps_per_chunk")
        
        if static_output and not os.path.isdir(static_output):
            print(f"Writing static data {static_output}")
            self.static_features.to_zarr(static_output, zarr_format=2, mode='w')
            
        if predictors_output and not os.path.isdir(predictors_output):
            print(f"Writing predictors data to {predictors_output}")
            predictors = rechunk_dataset(self.predictors, indices, timesteps_per_chunk)    
            predictors.to_zarr(
                predictors_output,
                zarr_format=2,
                mode="w"
            )
            
        if self.targets is not None and targets_output and not os.path.isdir(targets_output):
            print(f"Writing targets data to {targets_output}")
            targets = rechunk_dataset(self.targets, indices, timesteps_per_chunk)
            
            targets.to_zarr(
                targets_output,
                zarr_format=2,
                mode="w"
            )

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

        indices = build_indices(self.predictors, blocks)
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

def load_variable(varname, path, mode, timesteps_per_chunk=None, vertical_dim_name="plev"):
    if mode == 'mfdataset':
        path, suffix = path.split("*", 1)
        path = Path(path).expanduser().resolve()
        assert path.exists(), f"Path does not exist: {path}"
        files = sorted(path.rglob("*.nc"))  # Ensure the path is valid for globbing
        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            compat="no_conflicts",
            data_vars="minimal",
            coords="minimal",
            chunks={"time": timesteps_per_chunk} if timesteps_per_chunk else "auto",
            parallel=False,
        )
    else:
        ds = xr.open_dataset(
            path, 
            chunks={"time": timesteps_per_chunk} if timesteps_per_chunk else "auto",
        )
        
    da = ds[varname]

    # Case 1: has vertical dimension
    if vertical_dim_name in da.dims:
        out = {}
        levels = da.coords[vertical_dim_name].values

        for lev in levels:
            name = f"{varname}_{int(lev)}"
            out[name] = da.sel({vertical_dim_name: lev}).drop_vars(vertical_dim_name)

        return xr.Dataset(out)

    # Case 2: single level variable
    return da.to_dataset(name=varname)

def merge_dataset(input_path, input_files, mode, timesteps_per_chunk=None, vertical_dim_name='plev'):
    datasets = []
    for var, file_path in input_files.items():
        ds = load_variable(
            var, 
            os.path.join(input_path, file_path),
            mode, 
            timesteps_per_chunk=timesteps_per_chunk if timesteps_per_chunk else None,
            vertical_dim_name=vertical_dim_name
        )
        datasets.append(ds)

    if timesteps_per_chunk:
        return xr.merge(
            datasets,
            compat="no_conflicts",
            join="exact",
        ).chunk({"time": timesteps_per_chunk})
    else:
        return xr.merge(
            datasets,
            compat="no_conflicts",
            join="exact",
        )

def build_mask(ds, blocks):
    time = pd.to_datetime(ds.time.values)
    mask = np.zeros(len(time), dtype=bool)

    for start, end in blocks:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        mask |= (time >= start) & (time <= end)

    return mask

def build_indices(ds, blocks):
    mask = build_mask(ds, blocks)
    return np.flatnonzero(mask)

def rechunk_dataset(ds, indices, timesteps_per_chunk=None):
    da = ds.isel(time=indices)
    da = da.to_array(dim="channel")

    spatial_dims = [
        dim for dim in da.dims
        if dim not in ("time", "channel")
    ]

    da = da.transpose(
        "time",
        "channel",
        *spatial_dims
    )

    chunks = {
        "channel": -1,
        "time": timesteps_per_chunk if timesteps_per_chunk else 'auto',
    }

    for dim in spatial_dims:
        chunks[dim] = -1

    return da.chunk(chunks)    

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

def toml_dump(output_path, config_path="config.toml"):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"The source file does not exist: {config_path}")
    
    if not os.path.isdir(output_path):
        raise NotADirectoryError(f"The destination directory does not exist: {output_path}")
    
    filename = os.path.basename(config_path)
    destination_path = os.path.join(output_path, filename)

    # Load and re-serialize TOML
    print(f"Saving a copy of the config file to {destination_path}...")
    data = toml.load(config_path)
    with open(destination_path, 'w', encoding='utf-8') as dst_file:
        toml.dump(data, dst_file)
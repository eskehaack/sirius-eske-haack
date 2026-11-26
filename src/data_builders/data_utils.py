"""
Data loader for climate downscaling sampler.
Loads predictors, regridded, targets, and static features for a specific
date/member/scenario from a TOML config, using xarray lazy loading.

Generated on 2026-09-30 by Claude Sonnet 4.6 Low
"""

import re
import os
from datetime import datetime
from pathlib import Path

import xarray as xr
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_covers_date(filename: str, date: datetime) -> bool:
    """
    Quick pre-filter: check whether a filename's year-range suffix
    (e.g. '_1951-2014.nc') covers the requested date.
    Falls back to True (let xarray decide) if no range is found.
    """
    match = re.search(r"_(\d{4})-(\d{4})\.nc$", filename)
    if match:
        start_year, end_year = int(match.group(1)), int(match.group(2))
        return start_year <= date.year <= end_year
    return True  # no year range in name → include and let xarray filter


def _sel_date(ds: xr.Dataset, date: str) -> xr.Dataset:
    """Select a single date from the time dimension, if present."""
    if "time" in ds.dims:
        return ds.sel(time=date, method="nearest")
    return ds


def _load_source_block(block: dict, date: str, date_dt: datetime) -> xr.Dataset:
    """
    Load a single source block (predictors / regridded / targets).

    block keys expected:
        basepath  – root directory for the files
        mode      – 'single' (one file per variable) or
                    'mfdataset' (glob pattern, multiple files per variable)
        files     – {variable: filename_or_glob}
    """
    basepath = Path(block["basepath"])
    mode = block.get("mode", "single")
    files_map: dict[str, str] = block.get("files", {})

    datasets: list[xr.Dataset] = []

    for var, pattern in files_map.items():
        path = basepath / pattern

        if mode == "mfdataset":
            # Pattern is a glob (e.g. "tas/*.nc")
            matched = sorted(basepath.glob(pattern))
            if not matched:
                raise FileNotFoundError(
                    f"No files matched glob '{path}' for variable '{var}'"
                )
            # Open lazily, decode times, then immediately select the date
            ds = xr.open_mfdataset(
                matched,
                compat="no_conflicts",
                combine="by_coords",
                data_vars="minimal",
                chunks={},          # lazy / dask-backed
            )

        else:  # mode == 'single'
            fname = str(pattern)
            if not _file_covers_date(fname, date_dt):
                continue            # file doesn't cover this date – skip

            full_path = basepath / fname
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {full_path}")

            ds = xr.open_dataset(
                full_path,
                chunks={},          # lazy / dask-backed
            )

        # Select the requested date immediately so nothing else is loaded
        ds = _sel_date(ds, date)

        # Keep only the target variable to avoid pulling in extras
        vars_to_keep = [v for v in ds.data_vars if v == var or v == "rotated_latitude_longitude"]
        if vars_to_keep:
            ds = ds[vars_to_keep]

        datasets.append(ds)

    if not datasets:
        raise ValueError(
            f"No data loaded from basepath '{basepath}' for date '{date}'"
        )

    return xr.merge(datasets, compat='equals')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_sample(
    config: dict,
    date: str,
    scenario: str,
    member: str,
    model: str = "EC-Earth3-Veg",
) -> dict[str, xr.Dataset]:
    """
    Load all data for a single date / scenario / member.

    Parameters
    ----------
    config   : parsed TOML dict (e.g. from `tomllib.load(...)`)
    date     : ISO date string, e.g. "2010-06-30"
    scenario : e.g. "historical", "ssp126", "ssp370"
    member   : e.g. "r1i1p1f1"
    model    : GCM name (default "EC-Earth3-Veg")

    Returns
    -------
    dict with keys "predictors", "regridded", "targets", "static"
    """
    date_dt = datetime.strptime(date, "%Y-%m-%d")
    key_path = ["source", model, "scenario", scenario, "member", member]

    def _resolve(section: dict) -> dict:
        """Walk the nested key path to the leaf block."""
        node = section
        for k in key_path:
            node = node[k]
        return node

    # ---- predictors --------------------------------------------------------
    pred_block = _resolve(config["predictors"])
    predictors = _load_source_block(pred_block, date, date_dt)

    # ---- regridded ---------------------------------------------------------
    rg_block = _resolve(config["regridded"])
    regridded = _load_source_block(rg_block, date, date_dt)

    # ---- targets -----------------------------------------------------------
    tgt_block = _resolve(config["targets"])
    targets = _load_source_block(tgt_block, date, date_dt)

    # ---- static features (no date dimension) -------------------------------
    sf_cfg = config["static_features"]
    static_path = Path(sf_cfg["basepath"])
    static_datasets: list[xr.Dataset] = []

    for var, fname in sf_cfg["files"].items():
        full_path = static_path / fname
        if not full_path.exists():
            raise FileNotFoundError(f"Static file not found: {full_path}")
        ds = xr.open_dataset(full_path, chunks={})
        static_datasets.append(ds[[var]] if var in ds.data_vars else ds)

    static = xr.merge(static_datasets, compat="equals")

    return {
        "predictors": predictors,
        "regridded":  regridded,
        "targets":    targets,
        "static":     static,
    }


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

def build_mask(time, blocks):
    mask = np.zeros(len(time), dtype=bool)

    for start, end in blocks:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        mask |= (time >= start) & (time <= end)

    return mask

def build_indices(time, blocks):
    mask = build_mask(time, blocks)
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

def _to_sharded_dataset(obj, indices, chunk_t, shard_t, dtype="float32", default_name="data"):
    """Select valid timesteps, trim to whole chunks, stack fields into (time, channel, ...),
    cast to dtype, and build zarr v3 chunk/shard encoding."""
    indices = np.sort(np.asarray(indices))
    n = (len(indices) // chunk_t) * chunk_t
    if n < len(indices):
        print(f"  dropping last {len(indices) - n} timesteps so every chunk is full")

    if isinstance(obj, xr.DataArray):
        da = obj if "channel" in obj.dims else obj.expand_dims(channel=[obj.name or default_name])
    else:
        fields = [v for v in obj.data_vars if "time" in obj[v].dims]   # skip grid-mapping etc.
        da = obj[fields].to_array(dim="channel")

    da = da.isel(time=indices[:n]).astype(dtype)
    spatial_dims = [d for d in da.dims if d not in ("time", "channel")]
    da = da.transpose("time", "channel", *spatial_dims)

    # One dask chunk = one shard along time, full extent elsewhere
    da = da.chunk({d: (shard_t if d == "time" else -1) for d in da.dims})

    ds = da.to_dataset(name=default_name)            # back to a Dataset with a known name

    for var in ds.variables.values():                # drop inherited v2 encoding
        var.encoding = {}

    shape = ds[default_name].shape                   # (time, channel, y, x)
    encoding = {default_name: {
        "chunks": (chunk_t, *shape[1:]),             # one read
        "shards": (shard_t, *shape[1:]),             # one file
    }}
    return ds, encoding

# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import toml

    config = toml.load("./src/configs/training_config.toml")

    sample = load_sample(
        config,
        date="1952-03-15",
        scenario="historical",
        member="r1i1p1f1",
    )

    for name, ds in sample.items():
        print(f"\n=== {name} ===")
        print(ds)
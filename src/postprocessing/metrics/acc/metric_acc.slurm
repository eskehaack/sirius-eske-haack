#!/bin/bash

#SBATCH --job-name=ec_earth_data_postprocess_acc
#SBATCH --account=project_465002687
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=./hpc_outputs/%x-%j.out
#SBATCH --error=./hpc_outputs/%x-%j.err

member=r1i1p1f1
scenario=ssp370

OUTDIR=/scratch/project_465002687/ec_earth/metrics

if [[ ! -d "${OUTDIR}" ]]; then
    mkdir -p "${OUTDIR}"
fi

# Variables to process
VARS=("tasmax" "tasmin" "pr" "tas")

for x in "${VARS[@]}"; do
    echo "Processing variable: ${x}"

    HIST=/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg/historical/${member}/day/${x}
    SSP=/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg/${scenario}/${member}/day/${x}

    HCLIMFILES=("${HIST}"/*.nc)
    HCLIMFILES+=("${SSP}"/*.nc)

    REGRIDHIS="/scratch/project_465002687/ec_earth/predictors/regridded/historical/${member}/${x}_EUR-12_day_EC-Earth3-Veg_historical_${member}_regridded.nc"
    REGRIDSSP="/scratch/project_465002687/ec_earth/predictors/regridded/${scenario}/${member}/${x}_EUR-12_day_EC-Earth3-Veg_${scenario}_${member}_regridded.nc"
    REGRIDFILES=("${REGRIDHIS}" "${REGRIDSSP}")

    # Merge into files
    cdo mergetime -del29feb ${HCLIMFILES[@]} -detrend ${OUTDIR}/hclim_pipe_${x}.nc
    cdo mergetime -del29feb ${REGRIDFILES[@]} -detrend ${OUTDIR}/regrid_pipe_${x}.nc

    cdo ymonsub ${OUTDIR}/hclim_pipe_${x}.nc  -ymonmean ${OUTDIR}/hclim_pipe_${x}.nc  ${OUTDIR}/anom_hclim_${x}.nc
    cdo ymonsub ${OUTDIR}/regrid_pipe_${x}.nc -ymonmean ${OUTDIR}/regrid_pipe_${x}.nc ${OUTDIR}/anom_regrid_${x}.nc

    # Compute anomalies and ACC, reading from files
    cdo timcor ${OUTDIR}/anom_hclim_${x}.nc ${OUTDIR}/anom_regrid_${x}.nc \
        ${OUTDIR}/acc_${x}_${scenario}_${member}.nc

    # Clean up merges
    rm ${OUTDIR}/hclim_pipe_${x}.nc ${OUTDIR}/regrid_pipe_${x}.nc
    rm ${OUTDIR}/anom_hclim_${x}.nc ${OUTDIR}/anom_regrid_${x}.nc

    echo "Done: ${x}"
done

echo "All variables processed."
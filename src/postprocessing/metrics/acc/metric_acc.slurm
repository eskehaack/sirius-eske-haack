#!/bin/bash

#SBATCH --job-name=ec_earth_data_postprocess_acc
#SBATCH --account=project_465002687
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=./hpc_outputs/%x-%j.out
#SBATCH --error=./hpc_outputs/%x-%j.err

member=r1i1p1f1
scenario=ssp370

OUTDIR=/scratch/project_465002687/ec_earth/metrics

if [[ ! -d "${OUTDIR}" ]]; then
    mkdir -p "${OUTDIR}"
fi

# Variables to process
VARS=("pr")

for x in "${VARS[@]}"; do
    echo "Processing variable: ${x}"

    HIST=/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg/historical/${member}/day/${x}
    SSP=/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg/${scenario}/${member}/day/${x}

    HCLIMFILES=("${HIST}"/*.nc)
    HCLIMFILES+=("${SSP}"/*.nc)

    REGRIDHIS="/scratch/project_465002687/ec_earth/predictors/regridded/historical/${member}/${x}_EUR-12_day_EC-Earth3-Veg_historical_${member}_regridded.nc"
    REGRIDSSP="/scratch/project_465002687/ec_earth/predictors/regridded/${scenario}/${member}/${x}_EUR-12_day_EC-Earth3-Veg_${scenario}_${member}_regridded.nc"
    REGRIDFILES=("${REGRIDHIS}" "${REGRIDSSP}")

    # Merge files
    cdo mergetime ${HCLIMFILES[@]}  ${OUTDIR}/hclim_raw_${x}.nc
    cdo mergetime ${REGRIDFILES[@]} ${OUTDIR}/regrid_raw_${x}.nc

    echo "Merged files"

    # Remove seasonal cycle first
    cdo ydaysub ${OUTDIR}/hclim_raw_${x}.nc -ydaymean ${OUTDIR}/hclim_raw_${x}.nc  ${OUTDIR}/anom_hclim_${x}.nc
    cdo ydaysub ${OUTDIR}/hclim_raw_${x}.nc -ydaymean ${OUTDIR}/regrid_raw_${x}.nc ${OUTDIR}/anom_regrid_${x}.nc

    echo "Removed seasonal cycle"

    # Then detrend the anomalies
    cdo detrend,equal=false ${OUTDIR}/anom_hclim_${x}.nc  ${OUTDIR}/anom_hclim_detrended_${x}.nc
    cdo detrend,equal=false ${OUTDIR}/anom_regrid_${x}.nc ${OUTDIR}/anom_regrid_detrended_${x}.nc

    echo "Detrended anomalies"

    NTIME_HCLIM=$(cdo ntime ${OUTDIR}/anom_hclim_detrended_${x}.nc)
    NTIME_REGRID=$(cdo ntime ${OUTDIR}/anom_regrid_detrended_${x}.nc)

    if [ "${NTIME_HCLIM}" != "${NTIME_REGRID}" ]; then
        echo "ERROR: Time steps do not match for variable ${x}!"
        echo "  HCLIM:  ${NTIME_HCLIM}"
        echo "  REGRID: ${NTIME_REGRID}"
        exit 1
    fi

    echo "Time steps match (${NTIME_HCLIM}), proceeding with timcor..."

    # Compute ACC
    cdo timcor ${OUTDIR}/anom_hclim_detrended_${x}.nc ${OUTDIR}/anom_regrid_detrended_${x}.nc \
        ${OUTDIR}/acc_${x}_${scenario}_${member}.nc

    echo "Computed ACC for variable ${x}, saved to ${OUTDIR}/acc_${x}_${scenario}_${member}.nc"

    # Clean up
    rm ${OUTDIR}/hclim_noleap_${x}.nc ${OUTDIR}/regrid_noleap_${x}.nc
    rm ${OUTDIR}/hclim_raw_${x}.nc ${OUTDIR}/regrid_raw_${x}.nc
    rm ${OUTDIR}/hclim_merge_${x}.nc ${OUTDIR}/regrid_merge_${x}.nc
    rm ${OUTDIR}/anom_hclim_${x}.nc  ${OUTDIR}/anom_regrid_${x}.nc
    rm ${OUTDIR}/anom_hclim_detrended_${x}.nc ${OUTDIR}/anom_regrid_detrended_${x}.nc

    echo "Done: ${x}"
done

echo "All variables processed."
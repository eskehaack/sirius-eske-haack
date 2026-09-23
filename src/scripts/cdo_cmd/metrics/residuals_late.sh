#!/bin/bash

SCENARIO=ssp370
MEMBER=r1i1p1f1
X=pr
period=late

# Compute time mean for late century ec earth data
REGRIDDED=/scratch/project_465002687/ec_earth/predictors/regridded
INFILE="${REGRIDDED}/${SCENARIO}/${MEMBER}/${X}_EUR-12_day_EC-Earth3-Veg_${SCENARIO}_${MEMBER}_regridded.nc"

REGRID_OUTFILE=/scratch/project_465002687/ec_earth/metrics/regridded_${MEMBER}_${X}_mean_${period}.nc

if [ ! -e "${REGRID_OUTFILE}" ]; then
    echo "Computing time mean for late century ec earth data"
    cdo timmean -selyear,2070/2100 ${INFILE} ${REGRID_OUTFILE}
else
    echo "Output file ${REGRID_OUTFILE} already exists. Skipping time mean computation."
fi

# Compute time mean for late century hclim data
HCLIM=/scratch/project_465002687/ec_earth/targets/HCLIM/merged
INFILE="${HCLIM}/${X}_historical_${SCENARIO}_${MEMBER}_yearly.nc"
HCLIM_OUTFILE=/scratch/project_465002687/ec_earth/metrics/hclim_${MEMBER}_${X}_mean_${period}.nc

if [ ! -e "${HCLIM_OUTFILE}" ]; then
    echo "Computing time mean for late century hclim data"
    cdo timmean -selyear,2070/2100 ${INFILE} ${HCLIM_OUTFILE}
else
    echo "Output file ${HCLIM_OUTFILE} already exists. Skipping time mean computation."
fi

# Compute residuals
OUTFILE=/scratch/project_465002687/ec_earth/metrics/residuals_${MEMBER}_${X}_mean_${period}.nc

if [ ! -e "${OUTFILE}" ]; then
    echo "Computing residuals for late century data"
    cdo sub ${REGRID_OUTFILE} ${HCLIM_OUTFILE} ${OUTFILE}
else
    echo "Output file ${OUTFILE} already exists. Skipping residual computation."
fi

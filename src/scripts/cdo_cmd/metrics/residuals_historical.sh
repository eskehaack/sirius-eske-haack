#!/bin/bash

SCENARIO=historical
MEMBER=r1i1p1f1
X=pr
period=historical

# Compute time mean for historical ec earth data
REGRIDDED=/scratch/project_465002687/ec_earth/predictors/regridded
INFILE="${REGRIDDED}/${SCENARIO}/${MEMBER}/${X}_EUR-12_day_EC-Earth3-Veg_${SCENARIO}_${MEMBER}_regridded.nc"

REGRID_OUTFILE=/scratch/project_465002687/ec_earth/metrics/regridded_${MEMBER}_${X}_mean_${period}.nc

if [ ! -e "${REGRID_OUTFILE}" ]; then
    echo "Computing time mean for historical ec earth data"
    cdo timmean -selyear,1985/2014 ${INFILE} ${REGRID_OUTFILE}
else
    echo "Output file ${REGRID_OUTFILE} already exists. Skipping time mean computation."
fi

# Compute time mean for historical hclim data
HCLIM=/scratch/project_465002687/ec_earth/targets/HCLIM/merged
INFILE="${HCLIM}/${X}_historical_ssp370_${MEMBER}_yearly.nc"
HCLIM_OUTFILE=/scratch/project_465002687/ec_earth/metrics/hclim_${MEMBER}_${X}_mean_${period}.nc

if [ ! -e "${HCLIM_OUTFILE}" ]; then
    echo "Computing time mean for historical hclim data"
    cdo timmean -selyear,1985/2014 ${INFILE} ${HCLIM_OUTFILE}
else
    echo "Output file ${HCLIM_OUTFILE} already exists. Skipping time mean computation."
fi

# Compute residuals
OUTFILE=/scratch/project_465002687/ec_earth/metrics/residuals_${MEMBER}_${X}_mean_${period}.nc

if [ ! -e "${OUTFILE}" ]; then
    echo "Computing residuals for historical data"
    cdo sub ${REGRID_OUTFILE} ${HCLIM_OUTFILE} ${OUTFILE}
else
    echo "Output file ${OUTFILE} already exists. Skipping residual computation."
fi

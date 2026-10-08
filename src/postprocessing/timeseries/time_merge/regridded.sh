#!/bin/bash

X='pr'
MERGED="/scratch/project_465002687/ec_earth/predictors/regridded/merged"

if [ ! -d "${MERGED}" ]; then
    mkdir -p "${MERGED}"
fi

for member in r1i1p1f1 r2i1p1f1 r3i1p1f1; do
    MEMBER=${member}

    HIS="/scratch/project_465002687/ec_earth/predictors/regridded/historical/"
    HISFILE="/${X}_EUR-12_day_EC-Earth3-Veg_historical_${MEMBER}_regridded.nc"

    for scenario in ssp126 ssp370; do
        SSP="/scratch/project_465002687/ec_earth/predictors/regridded/${scenario}/"
        SSPFILE="/${X}_EUR-12_day_EC-Earth3-Veg_${scenario}_${MEMBER}_regridded.nc"

        OUTFILE=${MERGED}/${X}_historical_${scenario}_${MEMBER}_yearly.nc

        if [ -e "${OUTFILE}" ]; then
            echo "File ${OUTFILE} already exists, skipping."
        else
            cdo yearmean -mergetime ${HIS}${MEMBER}${HISFILE} ${SSP}${MEMBER}${SSPFILE} ${OUTFILE}
        fi
    done
done

#!/bin/bash

X=pr

for scenario in historical ssp126 ssp370; do
    for member in r1i1p1f1 r2i1p1f1 r3i1p1f1; do
        GRIDFILE=/users/haackesk/Desktop/sirius/src/scripts/cdo_cmd/grid_models/hclim_grid.txt
        INDIR="/scratch/project_465002687/ec_earth/predictors/EC-Earth3-Veg-v2/${scenario}/${member}"
        INFILES=("${INDIR}/${X}_*.nc")

        OUTDIR="/scratch/project_465002687/ec_earth/predictors/regridded/${scenario}/${member}"
        OUTFILE="${OUTDIR}/${X}_EUR-12_day_EC-Earth3-Veg_${scenario}_${member}_regridded.nc"

        if [ ! -d "${OUTDIR}" ]; then
            mkdir -p "${OUTDIR}"
        fi

        if [ -e "${OUTFILE}" ]; then
            echo "Output file ${OUTFILE} already exists. Overwriting"
            rm "${OUTFILE}"
            cdo remapcon,${GRIDFILE} ${INFILES[@]} ${OUTFILE}
        else
            echo "Regridding files for scenario: ${scenario}, member: ${member}, variable: ${X}"
            cdo remapcon,${GRIDFILE} ${INFILES[@]} ${OUTFILE}
        fi
    done
done
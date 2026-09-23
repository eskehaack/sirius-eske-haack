#!/bin/bash
set -e

SCENARIO=ssp126
MEMBER=r1i1p1f1
X=tasmax

HIST=/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg/historical/${MEMBER}/day/${X}
SSP=/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg/${SCENARIO}/${MEMBER}/day/${X}

MERGEDIR=/scratch/project_465002687/ec_earth/targets/HCLIM/merged
MERGEFILE="${MERGEDIR}/tasmax_historical_${SCENARIO}_r1i1p1f1.nc"

OUT=/scratch/project_465002687/ec_earth/metrics
SUFILE="${OUT}/su_hclim_r1i1p1f1_${SCENARIO}.nc"

FILES=("${HIST}"/*.nc)
FILES+=("${SSP}"/*.nc)

# Make one big merged file for variable X
echo "Merging ${X}: member=${MEMBER} scenario=${SCENARIO} (${#FILES[@]} files)"
mkdir -p "${MERGEDIR}" "${OUT}"
if [[ ! -e "${MERGEFILE}" ]]; then
    echo "Merging files into ${MERGEFILE}"
    cdo shifttime,2mon -mergetime "${FILES[@]}" "${MERGEFILE}"
else
    echo "Merged file ${MERGEFILE} already exists, skipping."
fi

# Compute the "summer days" metric (eca su)
if [[ ! -e "${SUFILE}" ]]; then
    echo "Calculating SU metric for ${MEMBER} ${SCENARIO} using ${X} into ${SUFILE}"

    cdo mergetime \
        -eca_su -selyear,1985/2014 "${MERGEFILE}" \
        -eca_su -selyear,2020/2049 "${MERGEFILE}" \
        -eca_su -selyear,2070/2099 "${MERGEFILE}" \
        "${SUFILE}"
else
    echo "SU metric file ${SUFILE} already exists, skipping."
fi


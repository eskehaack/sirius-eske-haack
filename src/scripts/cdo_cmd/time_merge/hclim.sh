#!/bin/bash

# set -euo pipefail

X='pr'
BASE="/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg"
OUT="/scratch/project_465002687/ec_earth/targets/HCLIM/merged"

mkdir -p "${OUT}"

for member in r1i1p1f1 r2i1p1f1 r3i1p1f1; do
    for scenario in ssp126 ssp370; do

        SCENARIO_DIR="${BASE}/${scenario}/${member}/day/${X}"
        HISTORICAL_DIR="${BASE}/historical/${member}/day/${X}"

        # Glob all time-slice files for this realisation/scenario
        SCENARIO_FILES=("${SCENARIO_DIR}"/${X}_EUR-12_EC-Earth3-Veg_${scenario}_${member}_*.nc)
        HISTORY_FILES=("${HISTORICAL_DIR}"/${X}_EUR-12_EC-Earth3-Veg_historical_${member}_*.nc)
    
        FILES=("${SCENARIO_FILES[@]}")
        FILES+=("${HISTORY_FILES[@]}")

        # Safety check — skip if no files found
        if [[ ! -e "${FILES[0]}" ]]; then
            echo "WARNING: No files found in ${IN_DIR}, skipping."
            continue
        fi

        echo "Processing member=${member} scenario=${scenario} (${#FILES[@]} files)"

        OUTFILE="${OUT}/${X}_historical_${scenario}_${member}_yearly.nc"

        # Merge all time slices, pipe directly into yearmean — no temp file needed
        cdo yearmean -mergetime "${FILES[@]}" "${OUTFILE}"

    done
done
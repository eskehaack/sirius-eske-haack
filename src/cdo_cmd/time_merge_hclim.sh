#!/bin/bash

set -euo pipefail

X='tas'
BASE="/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg"
OUT="/scratch/project_465002687/ec_earth/targets/HCLIM/merged"

mkdir -p "${OUT}"

for scenario in historical ssp126 ssp370; do
    for member in r1i1p1f1 r2i1p1f1 r3i1p1f1; do

        IN_DIR="${BASE}/${scenario}/${member}/day/${X}"

        # Glob all time-slice files for this realisation/scenario
        FILES=("${IN_DIR}"/${X}_EUR-12_EC-Earth3-Veg_${scenario}_${member}_*.nc)

        # Safety check — skip if no files found
        if [[ ! -e "${FILES[0]}" ]]; then
            echo "WARNING: No files found in ${IN_DIR}, skipping."
            continue
        fi

        echo "Processing scenario=${scenario} member=${member} (${#FILES[@]} files)"

        OUTFILE="${OUT}/${X}_${scenario}_${member}_yearly.nc"

        # Merge all time slices, pipe directly into yearmean — no temp file needed
        cdo yearmean -mergetime "${FILES[@]}" "${OUTFILE}"

    done
done
#!/bin/bash

#SBATCH --job-name=ec_earth_data_postprocess_r10mm
#SBATCH --account=project_465002687
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=./hpc_outputs/%x-%j.out
#SBATCH --error=./hpc_outputs/%x-%j.err

x=pr

for member in r1i1p1f1 r2i1p1f1 r3i1p1f1; do
    for scenario in ssp126 ssp370; do
        HIST=/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg/historical/${member}/day/${x}
        SSP=/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg/${scenario}/${member}/day/${x}

        MERGEDIR=/scratch/project_465002687/ec_earth/targets/HCLIM/merged
        MERGEFILE="${MERGEDIR}/${x}_historical_${scenario}_${member}.nc"

        OUT=/scratch/project_465002687/ec_earth/metrics
        R10FILE="${OUT}/r10mm_hclim_${member}_${scenario}.nc"

        FILES=("${HIST}"/*.nc)
        FILES+=("${SSP}"/*.nc)

        # Make one big merged file for variable x
        echo "Merging ${x}: member=${member} scenario=${scenario} (${#FILES[@]} files)"
        mkdir -p "${MERGEDIR}" "${OUT}"
        if [[ ! -e "${MERGEFILE}" ]]; then
            echo "Merging files into ${MERGEFILE}"
            cdo shifttime,2mon -mergetime "${FILES[@]}" "${MERGEFILE}"
        else
            echo "Merged file ${MERGEFILE} already exists, skipping."
        fi

        # Compute the "wet days" metric (eca r10mm)
        if [[ ! -e "${R10FILE}" ]]; then
            echo "Calculating R10mm metric for ${member} ${scenario} using ${x} into ${R10FILE}"

            cdo mergetime \
                -eca_r10mm -selyear,1985/2014 -mulc,86400 "${MERGEFILE}" \
                -eca_r10mm -selyear,2020/2049 -mulc,86400 "${MERGEFILE}" \
                -eca_r10mm -selyear,2070/2099 -mulc,86400 "${MERGEFILE}" \
                "${R10FILE}"
        else
            echo "R10mm metric file ${R10FILE} already exists, skipping."
        fi

        if [[ -e "${R10FILE}" && -e "${MERGEFILE}" ]]; then
            echo "All done. R10mm metric file: ${R10FILE}"
            echo "Removing merged file to save storage: ${MERGEFILE}"
            rm -f "${MERGEFILE}"
        else
            echo "Error: R10mm metric file or merged file not found."
            exit 1
        fi
    done
done
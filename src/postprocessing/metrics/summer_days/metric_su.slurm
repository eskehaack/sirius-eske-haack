#!/bin/bash

#SBATCH --job-name=ec_earth_data_postprocess_su
#SBATCH --account=project_465002687
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=./hpc_outputs/%x-%j.out
#SBATCH --error=./hpc_outputs/%x-%j.err

x=tasmax

for member in r1i1p1f1 r2i1p1f1 r3i1p1f1; do
    for scenario in ssp126 ssp370; do
        HIST=/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg/historical/${member}/day/${x}
        SSP=/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg/${scenario}/${member}/day/${x}

        MERGEDIR=/scratch/project_465002687/ec_earth/targets/HCLIM/merged
        MERGEFILE="${MERGEDIR}/${x}_historical_${scenario}_${member}.nc"

        OUT=/scratch/project_465002687/ec_earth/metrics
        SUFILE="${OUT}/su_hclim_${member}_${scenario}.nc"

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

        # Compute the "summer days" metric (eca su)
        if [[ ! -e "${SUFILE}" ]]; then
            echo "Calculating SU metric for ${member} ${scenario} using ${x} into ${SUFILE}"

            cdo mergetime \
                -eca_su -selyear,1985/2014 "${MERGEFILE}" \
                -eca_su -selyear,2020/2049 "${MERGEFILE}" \
                -eca_su -selyear,2070/2099 "${MERGEFILE}" \
                "${SUFILE}"
        else
            echo "SU metric file ${SUFILE} already exists, skipping."
        fi

        if [[ -e "${SUFILE}" && -e "${MERGEFILE}" ]]; then
            echo "All done. SU metric file: ${SUFILE}"
            echo "Removing merged file to save storage: ${MERGEFILE}"
            rm -f "${MERGEFILE}"
        else
            echo "Error: SU metric file or merged file not found."
            exit 1
        fi
    done
done
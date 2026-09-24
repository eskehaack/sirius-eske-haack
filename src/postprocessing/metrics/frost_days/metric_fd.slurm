#!/bin/bash

#SBATCH --job-name=ec_earth_data_postprocess_fd
#SBATCH --account=project_465002687
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=./hpc_outputs/%x-%j.out
#SBATCH --error=./hpc_outputs/%x-%j.err

x=tasmin

for member in r1i1p1f1 r2i1p1f1 r3i1p1f1; do
    for scenario in ssp126 ssp370; do
        HIST=/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg/historical/${member}/day/${x}
        SSP=/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg/${scenario}/${member}/day/${x}

        MERGEDIR=/scratch/project_465002687/ec_earth/targets/HCLIM/merged
        MERGEFILE="${MERGEDIR}/${x}_historical_${scenario}_${member}.nc"

        OUT=/scratch/project_465002687/ec_earth/metrics
        FDFILE="${OUT}/fd_hclim_${member}_${scenario}.nc"

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

        # Compute the "frost days" metric (eca fd)
        if [[ ! -e "${FDFILE}" ]]; then
            echo "Calculating FD metric for ${member} ${scenario} using ${x} into ${FDFILE}"

            cdo mergetime \
                -eca_fd -selyear,1985/2014 "${MERGEFILE}" \
                -eca_fd -selyear,2020/2049 "${MERGEFILE}" \
                -eca_fd -selyear,2070/2099 "${MERGEFILE}" \
                "${FDFILE}"
        else
            echo "FD metric file ${FDFILE} already exists, skipping."
        fi

        if [[ -e "${FDFILE}" && -e "${MERGEFILE}" ]]; then
            echo "All done. FD metric file: ${FDFILE}"
            echo "Removing merged file to save storage: ${MERGEFILE}"
            rm -f "${MERGEFILE}"
        else
            echo "Error: FD metric file or merged file not found."
            exit 1
        fi
    done
done
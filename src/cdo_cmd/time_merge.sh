set -euo pipefail

X='pr'

for member in r1i1p1f1 r2i1p1f1 r3i1p1f1; do
    MEMBER=${member}

    HIS="/scratch/project_465002687/ec_earth/predictors/EC-Earth3-Veg-v2/historical/"
    HISFILE="/${X}_EUR-12_day_EC-Earth3-Veg_historical_${MEMBER}_r360x180_1951-2014.nc"

    MERGED="/scratch/project_465002687/ec_earth/predictors/EC-Earth3-Veg-v2/merged"

    SSP126="/scratch/project_465002687/ec_earth/predictors/EC-Earth3-Veg-v2/ssp126/"
    SSP126FILE="/${X}_EUR-12_day_EC-Earth3-Veg_ssp126_${MEMBER}_r360x180_2015-2100.nc"

    cdo mergetime ${HIS}${MEMBER}${HISFILE} ${SSP126}${MEMBER}${SSP126FILE} ${MERGED}/${X}_historical_ssp126_${MEMBER}.nc
    cdo yearmean ${MERGED}/${X}_historical_ssp126_${MEMBER}.nc ${MERGED}/${X}_historical_ssp126_${MEMBER}_yearly.nc

    SSP370="/scratch/project_465002687/ec_earth/predictors/EC-Earth3-Veg-v2/ssp370/"
    SSP370FILE="/${X}_EUR-12_day_EC-Earth3-Veg_ssp370_${MEMBER}_r360x180_2015-2100.nc"

    cdo mergetime ${HIS}${MEMBER}${HISFILE} ${SSP370}${MEMBER}${SSP370FILE} ${MERGED}/${X}_historical_ssp370_${MEMBER}.nc
    cdo yearmean ${MERGED}/${X}_historical_ssp370_${MEMBER}.nc ${MERGED}/${X}_historical_ssp370_${MEMBER}_yearly.nc
done

srun \
    --account=project_465002687 \
    --partition=small \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=256G \
    --time=02:00:00 \
    bash -lc '
    module purge
    module load CrayEnv
    module load PRoot

    mkdir -p "/tmp/$USER/singularity-build"
    export SINGULARITY_TMPDIR="/tmp/$USER/singularity-build"
    export SINGULARITY_CACHEDIR="/tmp/$USER/singularity-build"

    mkdir -p "/tmp/$USER/mpl"
	export MPLCONFIGDIR="/tmp/$USER/mpl"

	cd ~/Desktop/sirius
	singularity build /scratch/project_465002687/ec_earth/containers/docker.sif singularity/docker.def
    '

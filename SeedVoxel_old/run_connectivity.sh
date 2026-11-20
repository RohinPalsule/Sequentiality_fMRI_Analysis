#!/bin/bash

#SBATCH -c 2
#SBATCH --job-name=fmri_preproc
#SBATCH --array=1-3
#SBATCH -o %x-%A-%a.out
#SBATCH -e %x-%A-%a.err
#SBATCH --mail-user=rapalsul@uci.edu
#SBATCH --mail-type=END

# Get subject corresponding to this array index
sub=$(sed -n "${SLURM_ARRAY_TASK_ID}p" sublist.txt)

# Remove "sub-" so Python gets just 01, 02, etc.
sub=${sub#sub-}

echo "Running subject: ${sub}"

singularity exec \
    --bind /mnt/data1:/mnt/data1 \
    mypython.sif \
    python seed_voxel_connectivity.py "${sub}"

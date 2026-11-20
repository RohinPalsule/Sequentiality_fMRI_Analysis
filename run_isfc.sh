#!/bin/bash

#SBATCH -c 2
#SBATCH --job-name=fmri_preproc
#SBATCH -o %x-%A.out
#SBATCH -e %x-%A.err
#SBATCH --mail-user=rapalsul@uci.edu
#SBATCH --mail-type=END

echo "Running isfc"

singularity exec \
    --bind /mnt/data1:/mnt/data1 \
    mypython.sif \
    python ISFC_pipeline.py isfc





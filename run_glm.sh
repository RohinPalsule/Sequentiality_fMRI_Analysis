#!/bin/bash

#SBATCH -c 2
#SBATCH --array=1-15
#SBATCH --job-name=fmri_preproc
#SBATCH -o %x-%A.out
#SBATCH -e %x-%A.err
#SBATCH --mail-user=rapalsul@uci.edu
#SBATCH --mail-type=END

LINE_NUM=$SLURM_ARRAY_TASK_ID
SUBJ=$(awk "NR==$LINE_NUM" /home/rapalsul/projects/LeeChen/glm_sublist.txt)

echo "Running glm"

singularity exec \
    --bind /mnt/data1:/mnt/data1 \
    mypython.sif \
    python FilmFest_GLM.py $SUBJ





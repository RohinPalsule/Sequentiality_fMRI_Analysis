#!/bin/bash

#SBATCH -c 8
#SBATCH --job-name=leechen_preproc
#SBATCH --array=2-20
#SBATCH -o %x-%A-%a.out
#SBATCH -e %x-%A-%a.err
#SBATCH --mail-user=rapalsul@uci.edu
#SBATCH --mail-type=END

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=4

# Parse subject from sublist
LINE_NUM=$SLURM_ARRAY_TASK_ID
SUBJ=$(awk "NR==$LINE_NUM" /home/rapalsul/projects/LeeChen/sublist.txt)

# Directories
BIDS_DIR=/mnt/data1/bornstein/LeeChenRaw/ds004042/
OUT_DIR=/mnt/data1/bornstein/LeeChenfMRIPrep/
WORK_DIR=/mnt/data1/bornstein/LeeChenRaw/fmriprep_working_dir/sub-${SUBJ}

mkdir -p ${OUT_DIR}
mkdir -p ${WORK_DIR}

# Singularity image and FreeSurfer license
SING_IMG=/home/rapalsul/mycontainers/fmriprep_23.0.0_20230315.simg
FS_LICENSE=/home/rapalsul/licenses/freesurfer_license.txt

# Create a unique writable directory for this Slurm job
TMP_DIR=/mnt/data1/bornstein/LeeChenRaw/fmriprep_tmp/${SLURM_JOB_ID}
mkdir -p ${TMP_DIR}

# Writable matplotlib/cache directory
export MPLCONFIGDIR=${TMP_DIR}/matplotlib
mkdir -p ${MPLCONFIGDIR}

# fMRIPrep options
FMRIPREP_OPTS="--output-spaces MNI152NLin2009cAsym anat fsaverage5 --fs-license-file ${FS_LICENSE} --cifti-output --nthreads 8 --omp-nthreads 4"

# Run fMRIPrep
unset PYTHONPATH
export FS_LICENSE=${FS_LICENSE}

singularity run --cleanenv \
    -B /mnt/data1/bornstein:/mnt/data1/bornstein \
    -B ${TMP_DIR}:/home/fmriprep \
    ${SING_IMG} \
    ${BIDS_DIR} \
    ${OUT_DIR} \
    participant \
    --participant-label ${SUBJ} \
    -w ${WORK_DIR} \
    --clean-workdir \
    ${FMRIPREP_OPTS}

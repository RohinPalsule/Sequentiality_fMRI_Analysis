#!/bin/bash
#SBATCH --job-name=datalad_install
#SBATCH --output=datalad_%j.out
#SBATCH --error=datalad_%j.err
#SBATCH --time=48:00:00

SIF=/mnt/data1/bornstein/datalad-wrappers/datalad-openneuro.sif
DATA=/mnt/data1/bornstein/LeeChenRaw
DSDIR=$DATA/ds004042

mkdir -p "$DATA"
cd "$DATA"

apptainer exec \
    --bind "$DATA":/mnt/data1/bornstein/LeeChenRaw \
    "$SIF" \
    datalad clone \
    https://github.com/OpenNeuroDatasets/ds004042.git \
    "$DSDIR"

apptainer exec \
    --bind "$DATA":/mnt/data1/bornstein/LeeChenRaw \
    --pwd "$DSDIR" \
    "$SIF" \
    datalad get -r -d "$DSDIR" .

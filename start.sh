#!/bin/bash
module load git
module load vim

module load conda
conda activate TorchEnv

echo "Starting..."
squeue --me

cd ~/rotation1/pipeline/
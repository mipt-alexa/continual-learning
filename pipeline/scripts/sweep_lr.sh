#!/bin/bash 
#SBATCH --job-name=sweep_lr                # Job name
#SBATCH --output=out/%x_%a.out               # Standard output (%A for job ID, %a for array index)
#SBATCH --error=out/%x_%a.err                # Standard error
#SBATCH --array=0-5                            # Array of jobs  
#SBATCH --time=24:00:00                       # Time limit
#SBATCH --ntasks=1                            # Number of tasks per job
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --mem=64G                             # Memory per job
#SBATCH --gres=gpu:1                          # Request 1 GPU
#SBATCH --partition=gpu                       # Use GPU partition (adjust if needed)

# Define the hyperparameter grid (combinations of learning rate and batch size)
MODE="sweep_lr"
# FREEZE=false
MODEL="resnet"
OPTIM="adam"
LEARNING_RATES=(5e-3 1e-3)
WEIGHT_DECAYS=(1e-4)
NUM_EPOCHS=20


num_learning_rates=${#LEARNING_RATES[@]}

# Calculate the index for each hyperparameter using the SLURM_ARRAY_TASK_ID
lr_idx=$((SLURM_ARRAY_TASK_ID % num_learning_rates))
w_idx=$((SLURM_ARRAY_TASK_ID / num_learning_rates))

# Set the hyperparameters for this job
LEARNING_RATE=${LEARNING_RATES[$lr_idx]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$w_idx]}

# Print the selected hyperparameters for logging/debugging
echo "Running $SLURM_ARRAY_TASK_ID task of job $SLURM_ARRAY_JOB_ID"     

# Load any necessary modules (if needed), activate virtual environments, etc.
module load cuda
module load conda
conda activate TorchEnv

# Run the Python training script with the selected hyperparameters
python src/main.py --mode $MODE --model $MODEL --lr $LEARNING_RATE --weight_decay $WEIGHT_DECAY --num_epochs $NUM_EPOCHS
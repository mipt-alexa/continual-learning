#!/bin/bash
#SBATCH --job-name=cnn_benchmark       # Job name
#SBATCH --output=out/output_%A_%a.out             # Standard output (%A for job ID, %a for array index)
#SBATCH --error=out/error_%A_%a.err               # Standard error
#SBATCH --array=0-2                           # Array of jobs  
#SBATCH --time=01:00:00                       # Time limit
#SBATCH --ntasks=1                            # Number of tasks per job
#SBATCH --cpus-per-task=4                     # Number of CPU cores per task
#SBATCH --mem=16G                             # Memory per job
#SBATCH --gres=gpu:1                          # Request 1 GPU
#SBATCH --partition=gpu                       # Use GPU partition (adjust if needed)

# Define the hyperparameter grid (combinations of learning rate and batch size)
LEARNING_RATES=(1e-4 1e-3 1e-2)
NUM_EPOCHS=(30)

# Calculate total number of combinations
num_learning_rates=${#LEARNING_RATES[@]}
num_epochs=${#NUM_EPOCHS[@]}

# Calculate the index for each hyperparameter using the SLURM_ARRAY_TASK_ID
lr_idx=$((SLURM_ARRAY_TASK_ID % num_learning_rates))
ep_idx=$((SLURM_ARRAY_TASK_ID / num_learning_rates))

# Set the hyperparameters for this job
LEARNING_RATE=${LEARNING_RATES[$lr_idx]}
NUM_EPOCHS=${NUM_EPOCHS[$ep_idx]}

ep_idx=$((ep_idx % num_epochs))

# Print the selected hyperparameters for logging/debugging
echo "Running job with lr = $LEARNING_RATE, num_epochs = $NUM_EPOCHS"

# Load any necessary modules (if needed), activate virtual environments, etc.
module load cuda
module load conda
conda activate TorchEnv

# Run the Python training script with the selected hyperparameters
python main.py --lr $LEARNING_RATE --num_epochs $NUM_EPOCHS

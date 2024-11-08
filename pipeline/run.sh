#!/bin/bash
#SBATCH --job-name=vit_bench       # Job name
#SBATCH --output=out/%x_%a.out             # Standard output (%A for job ID, %a for array index)
#SBATCH --error=out/%x_%a.err               # Standard error
#SBATCH --array=0-5                           # Array of jobs  
#SBATCH --time=06:00:00                       # Time limit
#SBATCH --ntasks=1                            # Number of tasks per job
#SBATCH --cpus-per-task=4                     # Number of CPU cores per task
#SBATCH --mem=64G                             # Memory per job
#SBATCH --gres=gpu:L40S:1                          # Request 1 GPU
#SBATCH --partition=gpu                       # Use GPU partition (adjust if needed)

# Define the hyperparameter grid (combinations of learning rate and batch size)
MODEL="vit"
MODE="full"
OPTIM="adam"
LEARNING_RATES=(5e-4 1e-4 5e-5)
WEIGHT_DECAYS=(1e-4 1e-3)
NUM_EPOCHS=60

# Calculate total number of combinations
num_learning_rates=${#LEARNING_RATES[@]}
num_decays=${#WEIGHT_DECAYS[@]}

# Calculate the index for each hyperparameter using the SLURM_ARRAY_TASK_ID
lr_idx=$((SLURM_ARRAY_TASK_ID % num_learning_rates))
w_idx=$((SLURM_ARRAY_TASK_ID / num_learning_rates))

# Set the hyperparameters for this job
LEARNING_RATE=${LEARNING_RATES[$lr_idx]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$w_idx]}


# Print the selected hyperparameters for logging/debugging
echo "Running $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID \n for $MODE task with $MODEL, optimizer $OPTIM, lr = $LEARNING_RATE, weight decay = $WEIGHT_DECAY for $NUM_EPOCHS epochs"

# Load any necessary modules (if needed), activate virtual environments, etc.
module load cuda
module load conda
conda activate TorchEnv

# Run the Python training script with the selected hyperparameters
python main.py --model $MODEL --mode $MODE --lr $LEARNING_RATE --weight_decay $WEIGHT_DECAY --num_epochs $NUM_EPOCHS

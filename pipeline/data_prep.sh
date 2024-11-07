#!/bin/bash
#SBATCH --job-name=cil_data_prep      # Job name
#SBATCH --output=out/output_%A.out             # Standard output (%A for job ID, %a for array index)
#SBATCH --error=out/error_%A.err               # Standard error
#SBATCH --time=01:00:00                       # Time limit
#SBATCH --ntasks=1                            # Number of tasks per job
#SBATCH --cpus-per-task=4                     # Number of CPU cores per task
#SBATCH --mem=16G                             # Memory per job
#SBATCH --partition=gpu                       # Use GPU partition (adjust if needed)


# Print the selected hyperparameters for logging/debugging
echo "Downloading and preprocessing datasets..."

# Load any necessary modules (if needed), activate virtual environments, etc.
module load cuda
module load conda
conda activate TorchEnv

# Run the Python training script with the selected hyperparameters
python init_data_proc.py 

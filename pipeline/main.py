import argparse
import torch
from torch import optim, nn

from data_proc import *
from train import *
from classes import *
from visualize import *
from initialize import *


def parse_args():
    # Define the hyperparameters as command-line arguments
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Model Training")
    
    parser.add_argument('--lr', type=float, required=True, help="Learning rate")
    parser.add_argument('--optimizer', type=str, default="adam", help="Optimizer (default: adam)")
    # parser.add_argument('--gpu', action='store_true', help="Use GPU if available")
    
    # Parse the arguments
    args = parser.parse_args()
    
    return args

def main():
    # Parse the arguments passed from the SLURM script
    args = parse_args()
    
    # Display the hyperparameters (for logging/debugging purposes)
    print(f"Hyperparameters:")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Optimizer: {args.optimizer}")
    # print(f"  Using GPU: {args.gpu}")
    
    # Here you can add the code for setting up the model, dataset, optimizer, etc.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"
    print("Device:", device)

    dataloaders = create_loaders()
    print("Dataloaders created...")
    
    model = create_model().to(device)
    optim = setup_optimizer(model.parameters(), lr=args.lr)
    print("Model created...")
    
    trainer = ExperimentTrainer(dataloaders,
                                model.to(device), 
                                optim,
                                nn.NLLLoss())

    print("Starting training process...")

    print(trainer.train_full(num_epochs=20))
    
    # Placeholder for actual model training logic

    

if __name__ == "__main__":
    main()

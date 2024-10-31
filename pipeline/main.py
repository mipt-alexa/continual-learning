import argparse
import torch
from torch import optim, nn

import json  

from data_proc import *
from train import *
from classes import *
from visualize import *
from initialize import *
from visualize import *


def custom_serializer(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # Convert PyTorch tensor to list


def parse_args():
    # Define the hyperparameters as command-line arguments
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Model Training")

    parser.add_argument('--model', type=str, default="", help="Model")   
    parser.add_argument('--mode', type=str, default="full", help="Mode of training")
    parser.add_argument('--optimizer', type=str, default="adam", help="Optimizer (default: adam)")
    parser.add_argument('--lr', type=float, required=True, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=30, help="Number of epochs")
    # parser.add_argument('--gpu', action='store_true', help="Use GPU if available")
    
    # Parse the arguments
    args = parser.parse_args()
    
    return vars(args)

def main():
    # Parse the arguments passed from the SLURM script
    args = parse_args()
    
    # Display the hyperparameters (for logging/debugging purposes)
    print(f"Hyperparameters:")
    for k, v in args.items():
        print(k, v)
            
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"
    print("Device:", device, torch.cuda.get_device_name(0))

    dataloaders = create_loaders()
    print("Dataloaders created...")
    
    model = create_model(args["model"]).to(device)
    optim = setup_optimizer(model.parameters(), lr=args["lr"])
    print("Model created...")

    trainer = ExperimentTrainer(dataloaders,
                                model.to(device), 
                                optim,
                                nn.NLLLoss(),
                                device)
    
    args["lr"] = "{:.0e}".format(args["lr"])
    run_name = '_'.join([str(value) for value in args.values()])
    print(run_name)

    print("Starting training...")
    if args["mode"] == "cil":
        acc_last, acc_0 = trainer.train_class_inc(num_epochs_per_task=args["num_epochs"])
        args["acc_last"] = acc_last
        args["acc_0"] = acc_0
    else:
        loss, acc_train, acc_val = trainer.train_full(num_epochs=args["num_epochs"])
        args["loss"] = loss
        args["acc_train"] = acc_train
        args["acc_val"] = acc_val

    print("Training finished")

    with open(f"./results/{run_name}.json", "w") as outfile:
        json.dump(args, outfile, default=custom_serializer)

    print("Results saved!")

    # save_file = open("./results/"+run_name, "w")  
    # save_file.close()  
    
    # torch.save(acc_val, f"./results/{run_name}.pt")

    # visualize(loss, [acc_train, acc_val], run_name)
    

if __name__ == "__main__":
    main()

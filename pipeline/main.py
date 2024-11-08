import argparse
import torch

import json  

from data_proc import *
from train import *
from classes import *
from visualize import *
from initialize import *


import torch._dynamo
torch._dynamo.config.suppress_errors = True

torch.set_float32_matmul_precision('high')


def custom_serializer(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # Convert PyTorch tensor to list


def parse_args():
    # Define the hyperparameters as command-line arguments
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Model Training")

    parser.add_argument('--mode', type=str, required=False, default="", help="Mode of training")
    parser.add_argument('--model', type=str, default="", help="Model")   
    parser.add_argument('--optimizer', type=str, default="adam", help="Optimizer (default: adam)")
    parser.add_argument('--lr', type=float, required=True, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0., help="Weights decay rate for optimizer")
    parser.add_argument('--num_epochs', type=int, default=30, help="Number of epochs")
    # parser.add_argument('--gpu', action='store_true', help="Use GPU if available")
    
    # Parse the arguments
    args = parser.parse_args()
    
    return vars(args)
    

def main():
    # Parse the arguments passed from the SLURM script
    args = parse_args()
    
    # Display the hyperparameters (for logging/debugging purposes)
    for key, value in args.items():
        print(f"{key}: {value}")
            
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"
    print("Device:", device, torch.cuda.get_device_name(0))

    
    if args["mode"] == "full":
        #for benchmarking on full cifar 100 dataloaders are different
        dataloaders = create_full_ds_loaders()  
    else:
        # regular [10] loaders with labels division
        dataloaders = create_loaders()
    print("Dataloaders created...")
    
    model = create_model(args["model"]).to(device)
    model = torch.compile(model)
    print("Model created...")

    optim = setup_optimizer(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    scheduler = setup_scheduler(optim, mode=args["mode"], num_epochs=args["num_epochs"])
    print("Optimizer and scheduler created..")
    
    trainer = ExperimentTrainer(dataloaders,
                                model.to(device), 
                                optim,
                                scheduler,
                                nn.NLLLoss(),
                                device)

    # Create readable filename for saving results containing hyperparameters
    args["lr"] = "{:.0e}".format(args["lr"]).replace('e-0', 'e-').replace('e+0', 'e+')
    args["weight_decay"] = "{:.0e}".format(args["weight_decay"]).replace('e-0', 'e-').replace('e+0', 'e+')
    run_name = '_'.join([str(value) for value in args.values()])

    print("Starting training...")
    
    if args["mode"] == "full":
        loss, acc_train, acc_val = trainer.train_full(num_epochs=args["num_epochs"])
        args["loss"] = loss
        args["acc_train"] = acc_train
        args["acc_val"] = acc_val
        
    elif args["mode"] == "cil":
        loss, acc_last, acc_0 = trainer.train_class_inc(num_epochs_per_task=args["num_epochs"])
        args["loss"] = loss
        args["acc_last"] = acc_last
        args["acc_0"] = acc_0

    elif args["mode"] == "cil_hook":
        loss, acc_last, acc_0 = trainer.train_class_inc_hook(num_epochs_per_task=args["num_epochs"])
        args["loss"] = loss
        args["acc_last"] = acc_last
        args["acc_0"] = acc_0

    elif args["mode"] == "cil_replay":
        loss, acc_last, acc_0 = trainer.train_class_inc_replay(num_epochs_per_task=args["num_epochs"])
        args["loss"] = loss
        args["acc_last"] = acc_last
        args["acc_0"] = acc_0        
 

    print("Training finished")
    
    # Saving results as json file in /results/ directory
    with open(f"./results/{run_name}.json", "w") as outfile:
        json.dump(args, outfile, default=custom_serializer)

    print("Results saved!")
    

if __name__ == "__main__":
    main()

import argparse
import json  
import wandb

from data_proc import *
from train import *
from classes import *
from initialize import *

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

TORCHDYNAMO_VERBOSE=0 

torch.set_float32_matmul_precision('high')


def custom_serializer(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # Convert PyTorch tensor to list


def parse_args():
    # Define the hyperparameters as command-line arguments
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Model Training")

    parser.add_argument('--mode', type=str, required=False, default="", help="Mode of training")
    parser.add_argument('--buffer', type=int, required=False, default=0, help="Replay buffer size")
    
    parser.add_argument('--freeze', type=bool, default=False, help="Flag to freeze partually the last linear layer")

    parser.add_argument('--model', type=str, default="", help="Model")   
    parser.add_argument('--optimizer', type=str, default="adam", help="Optimizer (default: adam)")
    parser.add_argument('--lr', type=float, required=True, help="Learning rate")
    parser.add_argument('--lr_mode', type=str, default="", required=False)

    parser.add_argument('--weight_decay', type=float, required=False, default=0., help="Weights decay rate for optimizer")
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
        if isinstance(value, float):
            print(f"{key}: {value:.0e}")
        else:
            print(f"{key}: {value}")

    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        args["device"] = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        args["device"] = device

    print("Device:", device, args["device"])

    num_tasks = 10
    if "full" in args["mode"]:
        # for benchmarking on full cifar 100 dataloaders are different
        dataloaders = create_full_ds_loaders()  
        num_tasks = 1
    else:
        # dataloaders with data of all previous tasks reshuffled
        # buffer size 0 corresponds to no replay
        dataloaders = create_loaders(buffer_size=args["buffer"])
        
    print("Dataloaders created...")


    num_epochs = args["num_epochs"] # FIX for replay mode - number epochs is larger
    num_tasks = 1 if "full" in args["mode"] else 10

    
    model = create_model(args["model"]).to(device)
    if torch.cuda.get_device_name(0) not in ["Tesla P100-PCIE-16GB", "NVIDIA GeForce GTX 1080 Ti"]:
        model = torch.compile(model)
    print("Model created...")

    optim = setup_optimizer(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    scheduler = setup_scheduler(optim, mode=args['lr_mode'], num_epochs_per_task=args['num_epochs'], num_tasks=num_tasks)
    print("Optimizer and scheduler created..")

    
    trainer = ExperimentTrainer(dataloaders,
                                model, 
                                optim,
                                scheduler,
                                nn.NLLLoss(),
                                device,
                                num_tasks
                               )

    
    args_copy = args.copy()
    
    # Create readable filename for saving results containing hyperparameters
    args["lr"] = "{:.0e}".format(args["lr"]).replace('e-0', 'e-').replace('e+0', 'e+')
    args["weight_decay"] = "{:.0e}".format(args["weight_decay"]).replace('e-0', 'e-').replace('e+0', 'e+')
    
    run_name = '_'.join([str(args[key]) for key in args.keys()])

    # Log results to WandB
    wandb.init(project=f'{args["mode"]}_{args["lr_mode"]}', 
               config=args_copy)

    print("Starting training...")

    
    if args["mode"] == "full":
        loss, acc_train, acc_val = trainer.train(tasks=[0], num_epochs_per_task=args["num_epochs"])
 
    elif args["freeze"]:
        loss, acc_train, acc_val = trainer.train_hook(num_epochs_per_task=args["num_epochs"])
        
    elif "replay" in args["mode"]:
        loss, acc_train, acc_val = trainer.train(tasks=range(0, 10), num_epochs_per_task=args["num_epochs"])

    else:
        loss, acc_train, acc_val = trainer.train(tasks=[0], num_epochs_per_task=args["num_epochs"])


    args["loss"] = loss
    args["acc_train"] = acc_train
    args["acc_val"] = acc_val
    
    print("Training finished")

    
    # Saving results as json file in /results/ directory
    with open(f"results/{run_name}.json", "w") as outfile:
        json.dump(args, outfile, default=custom_serializer)

    print("Results saved!")
    

if __name__ == "__main__":
    main()

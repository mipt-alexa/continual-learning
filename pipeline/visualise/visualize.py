import torch 
import json
import numpy as np 

import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True


def plot(run_name):
    if run_name[-5:] != ".json":
        run_name += ".json"
    with open(f'../results/{run_name}', 'r') as f:
      data = json.load(f)

    fig, ax = plt.subplots(1, 2, figsize=(8,2))

    for key, value in data.items():

        if key == "acc_train":
            value = np.array(value)
            for task_id in range(value.shape[1]):
                ax[1].plot(value[:, task_id], color=plt.get_cmap("tab20")(2*task_id+1), label=f"{key} {task_id}")

        elif key == "acc_val":
            value = np.array(value)
            for task_id in range(value.shape[1]):
                ax[1].plot(value[:, task_id], color=plt.get_cmap("tab20")(2*task_id), label=f"{key} {task_id}")

        elif key == "loss":
            ax[0].plot(value, label=key)
            
        elif isinstance(value, str):
            print(f"{key}: {value}")
            
        else:
            print(f"{key}: {value}")

    plt.title(run_name)
    ax[0].legend()
    # ax[1].legend()
    plt.show()
    
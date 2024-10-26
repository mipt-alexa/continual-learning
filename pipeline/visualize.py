import torch 

import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True


def visualize(losses, acc_scores, filename):
        
    fig, axs = plt.subplots(2, 1, figsize=(5,5), layout="tight")
    axs[0].plot(losses)
    axs[0].title.set_text("Loss")
    axs[1].plot(acc_scores[0], label="train")
    axs[1].plot(acc_scores[1], label="val")
    axs[1].title.set_text("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].legend()

    plt.savefig(f"./plots/{filename}.png")
    
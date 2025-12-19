import matplotlib.pyplot as plt
import os

def plot_learning_curves(log_history, output_dir):
    train_loss = []
    steps = []

    eval_loss = []
    eval_epochs = []
    # Extract train and eval loss from log_history
    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            train_loss.append(entry["loss"])
            steps.append(entry["epoch"])
        if "eval_loss" in entry and "epoch" in entry:
            eval_loss.append(entry["eval_loss"])
            eval_epochs.append(entry["epoch"])
    # Plot the learning curves
    plt.figure()
    plt.plot(steps[:len(train_loss)], train_loss, label="Train Loss")
    if eval_loss:
        plt.plot(eval_epochs, eval_loss, label="Eval Loss", color='orange', linewidth=2, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Learning Curves")
    plt.savefig(os.path.join(output_dir, "learning_curves.png"))
    plt.close()

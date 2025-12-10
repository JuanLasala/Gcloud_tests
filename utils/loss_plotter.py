import matplotlib.pyplot as plt
import os

def plot_learning_curves(log_history, output_dir):
    train_loss = []
    eval_loss = []
    steps = []

    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            train_loss.append(entry["loss"])
            steps.append(entry["epoch"])
        if "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])

    plt.figure()
    plt.plot(steps[:len(train_loss)], train_loss, label="Train Loss")
    plt.plot(steps[:len(eval_loss)], eval_loss, label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Learning Curves")
    plt.savefig(os.path.join(output_dir, "learning_curves.png"))
    plt.close()

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import os

def plot_confusion(y_true, y_pred, class_names, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

def save_classification_report(y_true, y_pred, class_names, output_dir):
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

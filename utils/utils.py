import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, auc, roc_curve
import numpy as np

def get_classification_reports(pred_probs, pred_labels, true_labels):
    # Compute AUC
    false_positive_rate, true_positive_rate, thresholds = roc_curve(true_labels, pred_probs)
    auc_ = auc(false_positive_rate, true_positive_rate)
    auc_ = round(auc_*100, 2)
    # Compute Precision
    prec = precision_score(true_labels, pred_labels)
    prec = round(prec*100, 2)
    # Compute Recall
    recall = recall_score(true_labels, pred_labels)
    recall = round(recall*100, 2)
    # Compute F1-score
    f1 = f1_score(true_labels, pred_labels)
    f1 = round(f1*100, 2)
    # Compute Accuracy
    acc = accuracy_score(true_labels, pred_labels)
    acc = round(acc*100, 2)
    return prec, recall, f1, auc_, acc


def plot_training(loss_list, metric_list, title):
    # %matplotlib inline
    # clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5) )
    fig.subplots_adjust(wspace=.2)
    plotLoss(ax1, np.array(loss_list), title)
    plotMetric(ax2, np.array(metric_list), title)
    plt.show()


def plotLoss(ax, loss_list, title):
    ax.plot(loss_list[:, 0], label="Train Loss")
    ax.plot(loss_list[:, 1], label="Validation Loss")
    ax.set_title("Loss Curves - " + title, fontsize=12)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.legend(prop={'size': 10})
    ax.grid()


def plotMetric(ax, metric_list, title):
    ax.plot(metric_list[:, 0], label="Train F1")
    ax.plot(metric_list[:, 1], label="Validation F1")
    ax.set_title("Metric Curve - " + title, fontsize=12)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.legend(prop={'size': 10})
    ax.grid()
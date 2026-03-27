import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    
    # Auto-generate labels if not provided
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    
def plot_confusion_matrix_top_devices(y_true, y_pred, top_n=15):
    counts = np.bincount(y_true)
    top_labels = np.argsort(counts)[-top_n:]

    mask = np.isin(y_true, top_labels)
    y_true_top = y_true[mask]
    y_pred_top = y_pred[mask]

    cm = confusion_matrix(
        y_true_top,
        y_pred_top,
        labels=top_labels,
        normalize="true"
    )

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=top_labels
    )
    disp.plot(cmap="Blues")
    for text in disp.ax_.texts:
        text.set_text("")
    plt.title(f"Confusion Matrix (Top {top_n} Devices)")
    plt.tight_layout()
    plt.show()

# def per_device_classification_report(y_true, y_pred, target_names=None):
#     report = classification_report(
#         y_true,
#         y_pred,
#         target_names=target_names,
#         output_dict=True,
#         zero_division=0,
#     )


#     return report


def per_device_classification_report(y_true, y_pred, target_names=None):
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    df = pd.DataFrame(report).transpose()

    # Remove non-device rows
    df = df[~df.index.isin(["accuracy", "macro avg", "weighted avg"])]

    # Rename columns for consistency
    df = df.rename(columns={"f1-score": "f1"})

    # Move device_id into column
    df["device_id"] = df.index

    # Optional: convert to int if labels are numeric
    try:
        df["device_id"] = df["device_id"].astype(int)
    except:
        pass

    return df.reset_index(drop=True)


def plot_metric_bars(metrics: dict, title="Ranking Metrics"):
    names = list(metrics.keys())
    values = list(metrics.values())


    plt.figure(figsize=(10, 5))
    plt.bar(names, values)
    plt.xticks(rotation=45)
    plt.ylabel("Score")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    

def plot_top_predicted_devices(device_priority_df, top_n=15):
    df = device_priority_df.head(top_n)

    plt.figure(figsize=(10, 5))
    plt.bar(df["device_id"].astype(str), df["pred_top1_count"])
    plt.title(f"Top-{top_n} Predicted Devices (Top-1)")
    plt.xlabel("Device ID")
    plt.ylabel("Top-1 Prediction Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_per_device_f1(per_device_df, top_n=15):
    df = per_device_df.head(top_n)

    plt.figure(figsize=(10, 5))
    plt.bar(df["device_id"].astype(str), df["f1"])
    plt.title(f"Top-{top_n} Devices by F1 Score")
    plt.xlabel("Device ID")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def print_classification_report(y_true, y_pred, target_names=None):
    print(classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
        )
    )

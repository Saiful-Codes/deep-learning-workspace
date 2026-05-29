from datetime import datetime
import numpy as np
import torch
import tqdm.notebook as tq
import wandb
import torchmetrics

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


# Determine which device to use
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)


def plot_confusion_matrix(cm, class_names, title=None, save_name=None):
    """
    Plot and optionally save a normalised confusion matrix.
    """

    # Normalise confusion matrix row-wise
    cm = cm.astype(np.float32) / cm.sum(axis=1)[:, None]

    df_cm = pd.DataFrame(cm, class_names, class_names)

    plt.figure(figsize=(8, 6))
    ax = sn.heatmap(df_cm, annot=True, cmap="flare")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if title is not None:
        plt.title(title)

    plt.tight_layout()

    if save_name is not None:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved as: {save_name}")

    plt.show()


def count_classes(preds):
    """
    Count the number of predictions per class.
    """

    pred_classes = preds.argmax(dim=1)
    n_classes = preds.shape[1]

    return [(pred_classes == c).sum().item() for c in range(n_classes)]


def train_epoch(epoch, model, optimizer, criterion, loader, num_classes, device):
    """
    Train the model for one full epoch.
    """

    model.train()

    loss_metric = torchmetrics.MeanMetric().to(device)
    acc_metric = torchmetrics.Accuracy(
        task="multiclass",
        num_classes=num_classes
    ).to(device)
    uar_metric = torchmetrics.Recall(
        task="multiclass",
        num_classes=num_classes,
        average="macro"
    ).to(device)

    for inputs, lbls in loader:
        inputs, lbls = inputs.to(device), lbls.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, lbls)

        loss.backward()
        optimizer.step()

        loss_metric.update(loss)
        acc_metric.update(outputs, lbls)
        uar_metric.update(outputs, lbls)

    metrics_dict = {
        "Loss_train": loss_metric.compute().item(),
        "Accuracy_train": acc_metric.compute().item(),
        "UAR_train": uar_metric.compute().item(),
    }

    return metrics_dict


def val_epoch(epoch, model, criterion, loader, num_classes, device):
    """
    Evaluate the model for one full validation epoch.
    """

    model.eval()

    loss_metric = torchmetrics.MeanMetric().to(device)
    acc_metric = torchmetrics.Accuracy(
        task="multiclass",
        num_classes=num_classes
    ).to(device)
    uar_metric = torchmetrics.Recall(
        task="multiclass",
        num_classes=num_classes,
        average="macro"
    ).to(device)
    cm_metric = torchmetrics.ConfusionMatrix(
        task="multiclass",
        num_classes=num_classes
    ).to(device)

    for inputs, lbls in loader:
        inputs, lbls = inputs.to(device), lbls.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, lbls)

        loss_metric.update(loss)
        acc_metric.update(outputs, lbls)
        uar_metric.update(outputs, lbls)
        cm_metric.update(outputs, lbls)

    metrics_dict = {
        "Loss_val": loss_metric.compute().item(),
        "Accuracy_val": acc_metric.compute().item(),
        "UAR_val": uar_metric.compute().item(),
    }

    cm = cm_metric.compute().cpu().numpy()

    return metrics_dict, cm


def train_model(model, train_loader, val_loader, optimizer, criterion,
                class_names, n_epochs, project_name, ident_str=None):
    """
    Train model, log metrics to W&B, print summary, and save final confusion matrix.
    """

    num_classes = len(class_names)
    model.to(device)

    if ident_str is None:
        ident_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_name = f"{model.__class__.__name__}_{ident_str}"

    run = wandb.init(project=project_name, name=exp_name)

    val_metrics_dict = {}
    cm = None

    try:
        for epoch in tq.tqdm(range(n_epochs), total=n_epochs, desc="Epochs"):

            train_metrics_dict = train_epoch(
                epoch,
                model,
                optimizer,
                criterion,
                train_loader,
                num_classes,
                device
            )

            val_metrics_dict, cm = val_epoch(
                epoch,
                model,
                criterion,
                val_loader,
                num_classes,
                device
            )

            wandb.log({
                **train_metrics_dict,
                **val_metrics_dict,
                "epoch": epoch + 1
            })

    finally:
        run.finish()

    print("\n========== RUN SUMMARY ==========")
    print(f"Run Name: {exp_name}")
    print(f"Epochs: {n_epochs}")

    if len(val_metrics_dict) > 0:
        print(f"Final Val Accuracy: {val_metrics_dict['Accuracy_val']:.4f}")
        print(f"Final Val UAR: {val_metrics_dict['UAR_val']:.4f}")

    print("Values to be copied into Excel spreadsheet.")
    print("=================================\n")

    if cm is not None:
        plot_confusion_matrix(
            cm,
            class_names,
            title=ident_str,
            save_name=f"{ident_str}_cm.png"
        )

    return val_metrics_dict, cm
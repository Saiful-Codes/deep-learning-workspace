import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import tqdm.notebook as tq
import wandb
import torchmetrics
from torch.utils.data import DataLoader

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# Determine which device on import, and then use that elsewhere.
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)


def plot_confusion_matrix(cm, class_names, save_path=None, title=None):
    '''
        cm: the confusion matrix that we wish to plot
        class_names: the names of the classes
        save_path: optional path. If given, the plot is written there
                   (overwriting any existing file), creating parent
                   folders if missing. Default None keeps the old behaviour.
        title: optional plot title (defaults to a generic caption).
    '''

    # this normalizes the confusion matrix
    cm = cm.astype(np.float32) / cm.sum(axis=1)[:, None]

    # Use a fresh figure/axes per call so repeated calls in one process don't
    # stack extra colorbars or overlap labels.
    df_cm = pd.DataFrame(cm, class_names, class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    sn.heatmap(df_cm, annot=True, fmt='.2f', cmap='flare', ax=ax,
               cbar=True, square=True)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title if title is not None
                 else 'Normalised confusion matrix (validation)')
    fig.tight_layout()

    # Save before show(): some backends clear the figure on show().
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def print_val_metrics(metrics_dict, cm, class_names):
    '''
    Print the final validation accuracy, UAR and per-class recall.
    Per-class recall is read straight off the confusion matrix
    (diagonal / row total), so nothing is hardcoded.
    '''
    row_sums = cm.sum(axis=1)
    recalls = np.divide(np.diag(cm).astype(np.float32), row_sums,
                        out=np.zeros(len(class_names), dtype=np.float32),
                        where=row_sums != 0)

    print("\n=== Final validation metrics (last epoch) ===")
    print(f"Validation Accuracy: {metrics_dict['Accuracy_val']:.4f}")
    print(f"Validation UAR     : {metrics_dict['UAR_val']:.4f}")
    print("Per-class recall:")
    for name, recall in zip(class_names, recalls):
        print(f"  {name:6s}: {recall:.4f}")

def count_classes(preds):
    '''
    Counts the number of predictions per class given preds, a tensor
    shaped [batch, n_classes], where the maximum per preds[i]
    is considered the "predicted class" for batch element i.
    '''
    pred_classes = preds.argmax(dim=1)
    n_classes = preds.shape[1]
    return [(pred_classes == c).sum().item() for c in range(n_classes)]

def train_epoch(epoch, model, optimizer, criterion, loader, num_classes, device):
    '''
    Train the model on the entire training set precisely once (one epoch).
    Lab 6 has a very similar function.
    '''

    model.train()

    # Initialize metrics: mean loss per example, accuracy, and UAR (macro recall).
    loss_metric = torchmetrics.MeanMetric().to(device)
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)
    uar_metric = torchmetrics.Recall(task='multiclass', num_classes=num_classes,
                                     average='macro').to(device)

    for i, (inputs, lbls) in enumerate(loader):
        inputs, lbls = inputs.to(device), lbls.to(device)

        # Standard training step: zero grads -> forward -> loss -> backward -> step.
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        # Accumulate metrics for this batch.
        loss_metric.update(loss)
        acc_metric.update(outputs, lbls)
        uar_metric.update(outputs, lbls)

    # Calculate epoch metrics, and store in a dictionary for wandb
    metrics_dict = {
        'Loss_train': loss_metric.compute().item(),
        'Accuracy_train': acc_metric.compute().item(),
        'UAR_train': uar_metric.compute().item(),
    }

    return metrics_dict

def val_epoch(epoch, model, criterion, loader, num_classes, device):
    '''
    Evaluate the model on the entire validation set.
    '''
    model.eval()

    # Initialize metrics: mean loss per example, accuracy, and UAR (macro recall).
    loss_metric = torchmetrics.MeanMetric().to(device)
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)
    uar_metric = torchmetrics.Recall(task='multiclass', num_classes=num_classes,
                                     average='macro').to(device)

    # Task 1c - confusion matrix accumulated over the whole validation set.
    cm_metric = torchmetrics.ConfusionMatrix(task='multiclass',
                                             num_classes=num_classes).to(device)

    with torch.no_grad():
        for inputs, lbls in loader:
            inputs, lbls = inputs.to(device), lbls.to(device)

            # No backward / no optimizer step during validation.
            outputs = model(inputs)
            loss = criterion(outputs, lbls)

            # Accumulate metrics for this batch.
            loss_metric.update(loss)
            acc_metric.update(outputs, lbls)
            uar_metric.update(outputs, lbls)
            cm_metric.update(outputs, lbls)

    # Calculate epoch metrics, and store in a dictionary for wandb
    metrics_dict = {
        'Loss_val': loss_metric.compute().item(),
        'Accuracy_val': acc_metric.compute().item(),
        'UAR_val': uar_metric.compute().item(),
    }

    # Compute the confusion matrix (raw counts; normalised at plot time).
    cm = cm_metric.compute().cpu().numpy()

    return metrics_dict, cm
    


def train_model(model, train_loader, val_loader, optimizer, criterion,
                class_names, n_epochs, project_name, ident_str=None):
                
    num_classes = len(class_names)
    model.to(device)
    
    # Initialise Weights and Biases (wandb) project
    if ident_str is None:
      ident_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model.__class__.__name__}_{ident_str}"
    run = wandb.init(project=project_name, name=exp_name)

    try:
        # Train by iterating over epochs
        for epoch in tq.tqdm(range(n_epochs), total=n_epochs, desc='Epochs'):
            train_metrics_dict = train_epoch(epoch, model, optimizer, criterion,
                    train_loader, num_classes, device)
                    
            val_metrics_dict, cm = val_epoch(epoch, model, criterion, 
                    val_loader, num_classes, device)
            wandb.log({**train_metrics_dict, **val_metrics_dict})
    finally:
        run.finish()

    # Report final validation metrics (from the last validation epoch)
    print_val_metrics(val_metrics_dict, cm, class_names)

    # Plot confusion matrix from results of last val epoch, and save a copy
    # to logs/ (overwritten on every run).
    plot_confusion_matrix(cm, class_names, save_path="logs/baseline_cm.png")

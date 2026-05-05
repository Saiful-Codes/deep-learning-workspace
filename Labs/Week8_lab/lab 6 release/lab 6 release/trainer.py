import torch
import torchmetrics
import tqdm.notebook as tq
import numpy as np
import wandb

import utils

"""
You should be familiar with all of the steps shown in this code.
Read through the functions and comments to understand the changes.
"""

def train_epoch(model,optimizer, loss_func, train_loader, device):
    """
    Perform one epoch of training on the provided model.
    """
    # Initialize metrics
    epoch_loss = torchmetrics.MeanMetric().to(device)
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes= 2).to(device)

    # Training loop
    model.train() # ensure the model is in train mode before training loop
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Perform inference, calculate gradients and update weights
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        epoch_loss(loss)
        acc_metric(outputs, labels)

    return epoch_loss.compute(), acc_metric.compute()


def test_epoch(model, loss_func, test_loader, device):
    """
    Evaluate the model on the entire test set.
    """
    # Initialize metrics
    epoch_loss = torchmetrics.MeanMetric().to(device)
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes= 2).to(device)

    model.eval() # ensure the model is in eval mode before testing loop
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Perform inference without gradients.
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

        # Accumulate metrics
        epoch_loss(loss)
        acc_metric(outputs, labels)

    return epoch_loss.compute(), acc_metric.compute()
    
    
def train_model(model, train_loader, test_loader, num_epochs, loss_func, optimizer, exp_name):
    """
    Completely train and evaluate the model over all epochs.

    Uses only a train and test split, so test data is used for validation.
    """

    # TODO: Uncomment below when you get to the wandb visualization section
    # run = wandb.init(project ="CSE5DL Transformer Lab", name=exp_name)

    device = utils.get_training_device()
    model.to(device)

    # Train by iterating over epochs
    for epoch in tq.tqdm(range(num_epochs), total=num_epochs, desc='Epochs'):
        train_epoch_loss, train_epoch_accuracy = \
            train_epoch(model, optimizer, loss_func, train_loader, device)
        test_epoch_loss, test_epoch_accuracy = \
            test_epoch(model, loss_func, test_loader, device)

        print("Epoch: {}. Train loss: {:.3f},  accuracy: {:.3f}. Test loss: {:.3f} accuracy: {:.3f}".format(
            epoch, train_epoch_loss, train_epoch_accuracy, test_epoch_loss, test_epoch_accuracy))

        #TODO: Uncomment below when you get to the wandb visualization section
        #wandb.log({
        #         "train_loss": train_epoch_loss,
        #         "train_accuracy": train_epoch_accuracy,
        #         "test_loss": test_epoch_loss,
        #         "test_accuracy": test_epoch_accuracy})
    
    # TODO: Uncomment below when you get to the wandb visualization section
    # Finish wandb run
    # run.finish()


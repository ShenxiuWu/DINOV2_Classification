"""
This script contains functions for training, validating, and testing a model using a mix of precision training.
It also includes a function for finding the best threshold for each class that maximizes the F1 score.

Functions:
- train_one_epoch_mix: Train the model for one epoch and calculate training metrics.
- validate_one_epoch_mix: Validate the model on the validation dataset for one epoch.
- test_loop_mix: Perform the testing loop for a given model on the test dataset.
- train_validation: Train and validate the model for one epoch.
- find_best_threshold_base: Find the best threshold for each class that maximizes the F1 score.
"""
import tqdm
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from utils.Utils import get_metrics
import numpy as np


def train_one_epoch_mix(
    model, train_loader, optimizer, loss_fn, scaler, device, metrics
):
    """
    Train the model for one epoch and calculate training metrics.

    Args:
        model: The model to train.
        train_loader: DataLoader for the training data.
        optimizer: Optimizer for updating model weights.
        loss_fn: Loss function.
        scaler: Gradient scaler for mixed precision training.
        device: Device to run the training on (CPU or GPU).
        metrics: List of metric objects to calculate during training.

    Returns:
        tuple: Average training loss and calculated metrics (accuracy, precision, recall, F1 score).
    """
    model.train()
    total_loss = 0
    accuracy_top1, precision, recall, f1_score = metrics

    for data, target in tqdm.tqdm(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(
            device, non_blocking=True
        )
        optimizer.zero_grad()

        with autocast():
            output = model(data)
            loss = loss_fn(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * data.size(0)
        preds = torch.argmax(output, dim=1).detach()
        accuracy_top1.update(preds, target)
        precision.update(preds, target)
        recall.update(preds, target)
        f1_score.update(preds, target)

    avg_train_loss = total_loss / len(train_loader.dataset)
    train_acc = accuracy_top1.compute()
    train_prec = precision.compute()
    train_rec = recall.compute()
    train_f1 = f1_score.compute()

    accuracy_top1.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()

    return avg_train_loss, train_acc, train_prec, train_rec, train_f1


def validate_one_epoch_mix(model, valid_loader, loss_fn, device, metrics):
    """
    Validate the model on the validation dataset for one epoch.

    Args:
        model (torch.nn.Module): The model to be validated.
        valid_loader (torch.utils.data.DataLoader): The validation data loader.
        loss_fn (torch.nn.Module): The loss function used for validation.
        device (torch.device): The device on which the validation will be performed.
        metrics (tuple): A tuple containing the metrics to be computed during validation.

    Returns:
        tuple: A tuple containing the average validation loss, validation accuracy,
            validation precision, validation recall, and validation F1 score.
    """

    model.eval()
    total_val_loss = 0
    # accuracy_top5
    accuracy_top1, precision, recall, f1_score = metrics

    with torch.no_grad():
        for data, target in tqdm.tqdm(valid_loader):
            data, target = data.to(device, non_blocking=True), target.to(
                device, non_blocking=True
            )

            with autocast():
                output = model(data)
                loss = loss_fn(output, target)

            total_val_loss += loss.item() * data.size(0)
            preds = torch.argmax(output, dim=1).detach()

            accuracy_top1.update(preds, target)
            # accuracy_top5.update(preds, target)
            precision.update(preds, target)
            recall.update(preds, target)
            f1_score.update(preds, target)

    avg_val_loss = total_val_loss / len(valid_loader.dataset)
    val_acc = accuracy_top1.compute()
    # val_acc_top5 = accuracy_top5.compute()
    val_prec = precision.compute()
    val_rec = recall.compute()
    val_f1 = f1_score.compute()

    # Reset metrics for next epoch
    accuracy_top1.reset()
    # accuracy_top5.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()
    # val_acc_top5
    return avg_val_loss, val_acc, val_prec, val_rec, val_f1


def test_loop_mix(model, test_loader, loss_fn, device, metrics):
    """
    Perform the testing loop for a given model on the test dataset.

    Args:
        model (torch.nn.Module): The model to be tested.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        loss_fn (torch.nn.Module): The loss function used for evaluation.
        device (torch.device): The device on which the computation will be performed.

    Returns:
        tuple: A tuple containing the average test loss, test accuracy, test precision,
                test recall, and test F1 score.
    """
    model.eval()
    total_val_loss = 0
    accuracy_top1, precision, recall, f1_score = metrics
    with torch.no_grad():
        for data, target in tqdm.tqdm(test_loader):
            data, target = data.to(device, non_blocking=True), target.to(
                device, non_blocking=True
            )

            with autocast():
                output = model(data)
                loss = loss_fn(output, target)

            total_val_loss += loss.item() * data.size(0)
            preds = torch.argmax(output, dim=1).detach()

            accuracy_top1.update(preds, target)
            precision.update(preds, target)
            recall.update(preds, target)
            f1_score.update(preds, target)

    avg_test_loss = total_val_loss / len(test_loader.dataset)
    test_acc = accuracy_top1.compute()
    test_prec = precision.compute()
    test_rec = recall.compute()
    test_f1 = f1_score.compute()

    accuracy_top1.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()
    return avg_test_loss, test_acc, test_prec, test_rec, test_f1


def train_validation(
    model, train_loader, valid_loader, optimizer, loss_fn, epoch, device
):
    """
    Train and validate the model
    Args:
        model (torch.nn.Module): The model to be trained and validated.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        valid_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        loss_fn (torch.nn.Module): Loss function.
        epoch (int): The current epoch number.
        device (torch.device): Device to run the training on (CPU or GPU).
    Returns:
        tuple: A tuple containing the average validation loss, validation accuracy,
            validation precision, validation recall, validation F1 score, training accuracy,
            average training loss, training precision, training recall, and training F1 score.
    """
    results_metrics = {}
    scaler = torch.amp.GradScaler("cuda")

    model.to(device)
    metrics = get_metrics(task="multiclass", num_class=10, device=device)
    avg_train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch_mix(
        model, train_loader, optimizer, loss_fn, scaler, device, metrics
    )
    if valid_loader is not None:
        
        avg_val_loss, val_acc, val_prec, val_rec, val_f1 = validate_one_epoch_mix(
            model, valid_loader, loss_fn, device, metrics
        )

    else:
        avg_val_loss, val_acc, val_prec, val_rec, val_f1 = None, None, None, None, None
    
    results_metrics["train_acc"] = train_acc
    results_metrics["train_loss"] = avg_train_loss
    results_metrics["train_prec"] = train_prec
    results_metrics["train_rec"] = train_rec
    results_metrics["train_f1"] = train_f1
    results_metrics["val_acc"] = val_acc
    results_metrics["val_loss"] = avg_val_loss
    results_metrics["val_prec"] = val_prec
    results_metrics["val_rec"] = val_rec
    results_metrics["val_f1"] = val_f1

    return results_metrics

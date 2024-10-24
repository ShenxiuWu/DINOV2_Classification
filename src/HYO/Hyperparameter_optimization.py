"""
Hyperparameter Optimization for DINOV2 using Different Training Split

This script performs hyperparameter optimization for the DINOV2 model using different training splits. It uses the Optuna library for the optimization process. The script takes command-line arguments for specifying the model name, experiment name, training dataset file path, and checkpoint path.

The script defines an objective function that is used by Optuna to evaluate different hyperparameter configurations. The objective function trains the DINOV2 model with the specified hyperparameters and returns the average F1 score obtained on the validation set.

Usage:
    python Dino_hyperparameter_optimization.py [--model_name MODEL_NAME] [--experiment_name EXPERIMENT_NAME] [--training_dataset TRAINING_DATASET] [--checkpoint_path CHECKPOINT_PATH]

Arguments:
    --model_name (str): Name of the model (default: "DINOv2_L")
    --experiment_name (str): Name of the experiment (default: "Quater_dataset")
    --training_dataset (str): Path to the training dataset file (default: "/home/vault/iwfa/iwfa100h/ai-faps-badar-alam/Data-split/Quater_dataset.csv")
    --checkpoint_path (str): Path to the checkpoint file
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loading import MNISTFashionDataLoader
from models.classifier import DinoVisionTransformerClassifier
from models.train_validation_test import train_validation, test_loop_mix
from utils.Utils import EarlyStopping, get_metrics, dino_backbones
from optuna.pruners import MedianPruner, PatientPruner
import numpy as np
import torch
import tqdm
import torch.optim as optim
from torch import nn
from optuna.trial import TrialState
import optuna
import argparse


# Ensure the script can find the necessary modules

# Set the device to GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Hyperparameter Optimization")
parser.add_argument("--model_name", type=str, default="dinov2_s")
parser.add_argument("--experiment_name", type=str, default="FashionMNIST")

args = parser.parse_args()

# Load the DINOV2 model

Static_path = "facebookresearch/dinov2"
model_name = dino_backbones["dinov2_s"]["name"]
embedding_size = dino_backbones["dinov2_s"]["embedding_size"]
backbone = torch.hub.load(Static_path, model_name)

def objective(trial):
    """
    Objective function for the hyperparameter optimization process.
    Args:
        trial: Optuna trial object
    returns:
        float: Average F1 score for the validation dataset

    """
    print("trial number: ", trial.number)

    classification_criterion = nn.CrossEntropyLoss()
    BATCHSIZE = trial.suggest_int("batch_size", 16, 64)
    best_acc = 0
    classifier = DinoVisionTransformerClassifier(
        dino_model=backbone, embed_dim=embedding_size, num_classes=10,freeze_feature_extractor=False
    )
            # Freeze layers based on trial suggestion
    freeze_or_not = trial.suggest_categorical("freeze", [True, False])
    if freeze_or_not:
        frozen_layers = []
        total_layers = len(list(classifier.named_children()))
        freeze_percentage = trial.suggest_float("freeze_percentage", 0.1, 0.9)
        print(f"freeze_percentage: {freeze_percentage}")
        num_layers_to_freeze = int(total_layers * freeze_percentage)
        named_layers = list(classifier.named_children())
        for name, layer in named_layers[:num_layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
                frozen_layers.append(name)
        trial.set_user_attr("frozen_layers", frozen_layers)

    # Suggest optimizer and learning rate
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(classifier.parameters(), lr=lr)

    # Suggest scheduler
    scheduler_name = trial.suggest_categorical(
        "scheduler", ["ReduceLROnPlateau", "CosineAnnealingLR"]
    )
    if scheduler_name == "ReduceLROnPlateau":
        factor = trial.suggest_float("factor", 0.1, 0.9)
        patience = trial.suggest_int("patience", 0, 50)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=patience, factor=factor
        )
    elif scheduler_name == "CosineAnnealingLR":
        T_max = trial.suggest_int("T_max", 50, 1000)
        eta_min = trial.suggest_float("eta_min", 0, 0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )

    # Suggest image input size
    # multiples_of_14 = [i for i in range(100, 225) if i % 14 == 0]
    # image_input = trial.suggest_categorical("image_input", multiples_of_14)

    accurary_lst = []

    # Load data
    dataset = MNISTFashionDataLoader(
        data_dir="../dataset", batch_size=BATCHSIZE, train_split=0.8   )
    train_loader, valid_loader= dataset.get_train_loader(), dataset.get_val_loader()
    num_epochs = 10
    # Training loop
    for epoch in tqdm.tqdm(range(num_epochs)):
        print
        result_metrics= train_validation(
            classifier,
            train_loader,
            valid_loader,
            optimizer,
            classification_criterion,
            epoch,
            DEVICE,
        )
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(result_metrics["val_loss"])
        else:
            scheduler.step()
        trial.report(result_metrics["val_acc"].cpu().numpy(), epoch)
        accurary_lst.append(result_metrics["val_acc"].cpu().numpy())

        if trial.should_prune():
            print("pruning")
            raise optuna.TrialPruned()
        if best_acc < result_metrics["val_acc"]:
            best_acc = result_metrics["val_acc"]

    return np.average(accurary_lst)


if __name__ == "__main__":

    # Create a study and optimize the objective function
    study_name = f"optuna-exp__{args.model_name}_{args.experiment_name}"
    storage_name = f"sqlite:///{args.model_name}_{args.experiment_name}.db"

    pruner = PatientPruner(MedianPruner(), patience=5)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )
    completed_trials = len(
        study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    )
    remaining_trials = max(0, 70 - completed_trials)
    print(f"completed_trials: {completed_trials}")
    print(f"remaining_trials: {remaining_trials}")
    study.optimize(objective, n_trials=remaining_trials)
    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

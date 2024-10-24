import os
import sys
import torch
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard SummaryWriter
from data.data_loading import MNISTFashionDataLoader
from models.classifier import DinoVisionTransformerClassifier
from models.train_validation_test import train_validation, test_loop_mix
from utils.Utils import EarlyStopping, get_metrics, dino_backbones


def main(resume_training: bool = True, train: bool = True):
    # Device initialization and Tensorboard writer
    Device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataloader
    Dataloader = MNISTFashionDataLoader(data_dir="../dataset", batch_size=32, train_split=0.8)
    train_loader, val_loader, test_loader = (
        Dataloader.get_train_loader(),
        Dataloader.get_val_loader(),
        Dataloader.get_test_loader(),
    )
    # train_loader,test_loader = Dataloader.get_train_loader(),Dataloader.get_test_loader()

    # Model initialization
    Static_path = "facebookresearch/dinov2"
    model_name = dino_backbones["dinov2_s"]["name"]
    embedding_size = dino_backbones["dinov2_l"]["embedding_size"]
    backbone = torch.hub.load(Static_path, model_name)
    backbone.to(Device)
    writer = SummaryWriter(f"model_logs_{model_name}")
    # Classifier
    classifier = DinoVisionTransformerClassifier(
        dino_model=backbone,
        embed_dim=embedding_size,
        num_classes=10,
        freeze_feature_extractor=False,
    )
    classifier.to(Device)
    if resume_training:
        checkpoint = torch.load(
            "model_logs/checkpoint/checkpoint.pt", map_location=Device, weights_only=True
        )
        classifier.load_state_dict(checkpoint)
    # Optimizer and Loss
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    # Early Stopping
    checkpoint_path = f"model_logs{model_name}/checkpoint"
    os.makedirs(checkpoint_path, exist_ok=True)
    early_stopping = EarlyStopping(
        patience=7, delta=0.0, verbose=True, path=f"{checkpoint_path}/checkpoint.pt"
    )

    # Training
    if train:
        num_epochs = 100
        for epoch in range(num_epochs):
            (
                results_metrics  #  model, train_loader, valid_loader, optimizer, loss_fn, epoch, device
            ) = train_validation(
                classifier,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                epoch,
                Device,
            )
            if results_metrics["val_acc"] is None:
                scheduler.step(results_metrics["train_loss"])
                early_stopping(results_metrics["train_acc"], classifier)
            else:
                scheduler.step(results_metrics["val_loss"])
                early_stopping(results_metrics["val_loss"], classifier)

            for key, value in results_metrics.items():
                writer.add_scalar(key, value, epoch)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        writer.close()
    else:
        metrics = get_metrics(task="multiclass", num_class=10, device=Device)
        checkpoint = torch.load(
            f"model_logs{model_name}/checkpoint/checkpoint.pt",
            map_location=Device,
            weights_only=True,
        )
        classifier.load_state_dict(checkpoint)
        (
            avg_test_loss,
            test_acc,
            test_prec,
            test_rec,
            test_f1,
        ) = test_loop_mix(classifier, test_loader, criterion, Device, metrics)
        print(
            f"Test Loss: {avg_test_loss}, Test Accuracy: {test_acc}, Test Precision: {test_prec}, Test Recall: {test_rec}, Test F1: {test_f1}"
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--resume_training", type=bool, default=False)
    argparser.add_argument("--train", type=bool, default=True)
    args = argparser.parse_args()

    main(args.resume_training, args.train)

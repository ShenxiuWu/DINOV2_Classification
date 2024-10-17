import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard SummaryWriter
from data.data_loading import MNISTFashionDataLoader
from models.classifier import DinoVisionTransformerClassifier
from models.train_validation_test import train_validation, test_loop_mix
from utils.Utils import EarlyStopping

dino_backbones = {
    "dinov2_s": {
        "name": "dinov2_vits14",
        "embedding_size": 384,
    },
    "dinov2_b": {
        "name": "dinov2_vitb14",
        "embedding_size": 768,
    },
    "dinov2_l": {
        "name": "dinov2_vitl14",
        "embedding_size": 1024,
    },
    "dinov2_g": {
        "name": "dinov2_vitg14",
        "embedding_size": 1536,
    },
}


def main():
    # Device initialization and Tensorboard writer
    Device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter("model_logs")

    # Dataloader
    Dataloader = MNISTFashionDataLoader(data_dir="../dataset", batch_size=32, train_split=0.8)
    train_loader, val_loader, test_loader = (
        Dataloader.get_train_loader(),
        Dataloader.get_val_loader(),
        Dataloader.get_test_loader(),
    )

    # Model initialization
    Static_path = "facebookresearch/dinov2"
    model_name = dino_backbones["dinov2_l"]["name"]
    embedding_size = dino_backbones["dinov2_l"]["embedding_size"]
    backbone = torch.hub.load(Static_path, model_name)
    backbone.to(Device)

    # Classifier
    classifier = DinoVisionTransformerClassifier(
        dino_model=backbone, embed_dim=embedding_size, num_classes=10
    )
    classifier.to(Device)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    # Early Stopping
    checkpoint_path = "model_logs/checkpoint"
    os.makedirs(checkpoint_path, exist_ok=True)
    early_stopping = EarlyStopping(
        patience=5, delta=0.0, verbose=True, path=f"{checkpoint_path}/checkpoint.pt"
    )

    # Training
    num_epochs = 50
    for epoch in range(num_epochs):
        (
            avg_val_loss,
            val_acc,
            val_prec,
            val_rec,
            val_f1,
            train_acc,
            avg_train_loss,
            train_prec,
            train_rec,
            train_f1,    #  model, train_loader, valid_loader, optimizer, loss_fn, epoch, device
        ) = train_validation(
            classifier, train_loader, val_loader, optimizer, criterion,epoch, Device,
        )
        scheduler.step(avg_val_loss)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Precision/train", train_prec, epoch)
        writer.add_scalar("Precision/val", val_prec, epoch)
        writer.add_scalar("Recall/train", train_rec, epoch)
        writer.add_scalar("Recall/val", val_rec, epoch)
        writer.add_scalar("F1/train", train_f1, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)

        early_stopping(avg_val_loss, classifier)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    writer.close()

    checkpoint = torch.load("model_logs/checkpoint/checkpoint.pt")
    classifier.load_state_dict(checkpoint["model_state_dict"])
    avg_test_loss, test_acc, test_acc_top5, test_prec, test_rec, test_f1 = test_loop_mix(
        classifier, test_loader, criterion, Device
    )
    print(
        f"Test Loss: {avg_test_loss}, Test Accuracy: {test_acc}, Test Accuracy Top5: {test_acc_top5}, Test Precision: {test_prec}, Test Recall: {test_rec}, Test F1: {test_f1}"
    )


if __name__ == "__main__":
    main()

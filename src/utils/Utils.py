
import torch
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy
import numpy as np
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
    ):
        """
        Args:
            patience : How long to wait after last time validation loss improved.
                            Default: 7
            verbose  : If True, prints a message for each validation loss improvement.
                            Default: False
            delta    : Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path     : Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func : trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )

        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        


def get_metrics(task="multiclass", num_class=10, device="cuda"):
    """
    Get the metrics for the model
        Args:
            task: The task to perform multiclass or multilabel
            num_class: The number of classes in the dataset in case of multiclass
            device: str: The device to use for the computation
        Returns:
            accuracy: The accuracy metric
            precision: The precision metric
            recall: The recall metric
            f1_score: The f1_score metric
    """
    if not isinstance(num_class, int):
        raise ValueError(f"`num_class` is expected to be `int` but `{type(num_class)}` was passed.")
    
    accuracy_top1 = MulticlassAccuracy(num_classes=num_class,top_k=1,average='macro').to(device)
   
    accuracy_top5 = MulticlassAccuracy(num_classes=num_class,top_k=5,average='macro').to(device)
    precision = torchmetrics.Precision(
        task=task, average="macro", num_classes=num_class
    ).to(device)
    recall = torchmetrics.Recall(
        task=task, average="macro", num_classes=num_class
    ).to(device)
    f1_score = torchmetrics.F1Score(
        task=task, average="macro", num_classes=num_class
    ).to(device)

    return accuracy_top1, accuracy_top5, precision, recall, f1_score
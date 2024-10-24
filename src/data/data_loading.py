import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class MNISTFashionDataLoader:
    def __init__(self, data_dir, batch_size=32, train_split=0.8):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_split = train_split
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self._load_data()

    def _load_data(self):
        dataset = datasets.FashionMNIST(
            self.data_dir, download=True, train=True, transform=self.transform
        )
        train_size = int(self.train_split * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        
        self.test_dataset = datasets.FashionMNIST(
            self.data_dir, download=True, train=False, transform=self.transform
        )

    def get_train_loader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8
        )

    def get_val_loader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

    def get_test_loader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

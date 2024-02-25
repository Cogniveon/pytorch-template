import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets


def create_dataloader(
    dataset=datasets.MNIST, batch_size=8, dataset_root="~/datasets/", **kwargs
):
    train_loader = DataLoader(
        dataset(
            dataset_root,
            train=True,
            download=True,
            transform=T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]),
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    val_loader = DataLoader(
        dataset(
            dataset_root,
            train=False,
            transform=T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]),
        ),
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )

    return train_loader, val_loader

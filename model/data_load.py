import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

transform = v2.Compose(
    [
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(mean=(0.1307,), std=(0.3081,)),
    ]
)

# mean and std are calculated for MNIST dataset

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)
train_dataset = torch.utils.data.Subset(train_dataset, range(50000))
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_dataset = torch.utils.data.Subset(test_dataset, range(10000))

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=3, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=3, pin_memory=True
)

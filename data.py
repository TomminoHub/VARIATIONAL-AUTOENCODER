import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_mnist_dataloaders(batch_size=128, data_dir='./data'):
    # Transform: convert to tensor and normalize to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Download and load the training data
    mnist_trainval = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # Split train into 50k train and 10k val
    train_size = 50000
    val_size = len(mnist_trainval) - train_size
    mnist_train, mnist_val = random_split(mnist_trainval, [train_size, val_size])

    # Dataloaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
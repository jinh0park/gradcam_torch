import torchvision
import torch
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

# Reference: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

def train_valid_loader(valid_ratio=0.9, data_path='data', batch_size=100):
    train_transform = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    ])

    valid_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    ])

    train_dataset = torchvision.datasets.STL10(data_path, split='train', download=True, transform=train_transform)
    valid_dataset = torchvision.datasets.STL10(data_path, split='train', download=True, transform=valid_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))

    split = int((1-valid_ratio) * num_train)

    train_idx, valid_idx = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
    )

    return (train_loader, valid_loader)

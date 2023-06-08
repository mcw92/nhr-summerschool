import torch
import torchvision
import numpy as np

def get_dataloaders_cifar10(batch_size, num_workers=0,
                            root='data',
                            validation_fraction=0.1,
                            train_transforms=None,
                            test_transforms=None):

    if train_transforms is None:
        train_transforms = torchvision.transforms.ToTensor()

    if test_transforms is None:
        test_transforms = torchvision.transforms.ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(root=root,
                                                 train=True,
                                                 transform=train_transforms,
                                                 download=True)

    valid_dataset = torchvision.datasets.CIFAR10(root=root,
                                                 train=True,
                                                 transform=test_transforms)

    test_dataset = torchvision.datasets.CIFAR10(root=root,
                                                train=False,
                                                transform=test_transforms)

    # Perform index-based train-validation split of original training data. 
    total = len(train_dataset) # Get overall number of samples in original training data.
    idx = list(range(total)) # Make index list.
    np.random.shuffle(idx) # Shuffle indices.
    vnum = int(validation_fraction * total) # Determine number of validation samples from validation split.
    train_indices, valid_indices = idx[vnum:], idx[0:vnum] # Extract train and validation indices.
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_indices)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               sampler=valid_sampler)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              shuffle=False)

    return train_loader, valid_loader, test_loader

def get_dataloaders_cifar10_ddp(batch_size, num_workers=0,
                                root='data',
                                validation_fraction=0.1,
                                train_transforms=None,
                                test_transforms=None):
    
    if train_transforms is None: train_transforms = torchvision.transforms.ToTensor()
    if test_transforms is None: test_transforms = torchvision.transforms.ToTensor()
        
    train_dataset = torchvision.datasets.CIFAR10(root=root,
                                                 train=True,
                                                 transform=train_transforms,
                                                 download=True)

    valid_dataset = torchvision.datasets.CIFAR10(root=root,
                                                 train=True,
                                                 transform=test_transforms)
    
    # Perform index-based train-validation split of original training data. 
    total = len(train_dataset) # Get overall number of samples in original training data.
    idx = list(range(total)) # Make index list.
    np.random.shuffle(idx) # Shuffle indices.
    vnum = int(validation_fraction * total) # Determine number of validation samples from validation split.
    train_indices, valid_indices = idx[vnum:], idx[0:vnum] # Extract train and validation indices.

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indices)

    # Sampler that restricts data loading to a subset of the dataset.
    # Especially useful in conjunction with torch.nn.parallel.DistributedDataParallel. 
    # Each process can pass a DistributedSampler instance as a DataLoader sampler, 
    # and load a subset of the original dataset that is exclusive to it.
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
                        train_dataset, 
                        num_replicas=torch.distributed.get_world_size(), 
                        rank=torch.distributed.get_rank(), 
                        shuffle=True, 
                        drop_last=True)
    
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
                        valid_dataset, 
                        num_replicas=torch.distributed.get_world_size(), 
                        rank=torch.distributed.get_rank(), 
                        shuffle=True, 
                        drop_last=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               drop_last=True,
                                               sampler=train_sampler)
    
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               drop_last=True,
                                               sampler=valid_sampler)

    return train_loader, valid_loader

def get_dataloaders_cifar10_ddp_min(batch_size,
                                    root='data',
                                    train_transforms=None):

    if train_transforms is None: train_transforms = torchvision.transforms.ToTensor()
        
    train_dataset = torchvision.datasets.CIFAR10(root=root,
                                                 train=True,
                                                 transform=train_transforms,
                                                 download=True)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
                        train_dataset,           # Dataset used for sampling.
                        num_replicas=torch.distributed.get_world_size(), # Number of processes in distributed training. 
                        rank=torch.distributed.get_rank(),               # Rank of current process within num_replicas. 
                        shuffle=True,            # Shuffle indices.
                        drop_last=True)          # Drop tail of data to make it evenly divisible across replicas.

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               drop_last=True,
                                               sampler=train_sampler)

    return train_loader

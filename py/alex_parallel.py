import os
import torch
import torchvision
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from model import AlexNet
from helper_dataset import get_dataloaders_cifar10_ddp
from helper_train import train_model_ddp, get_right_ddp, compute_accuracy_ddp

def main():
    world_size = int(os.getenv("SLURM_NPROCS")) # Get overall number of processes.
    rank = int(os.getenv("SLURM_PROCID"))       # Get individual process ID.
    slurm_job_gpus = os.getenv("SLURM_JOB_GPUS")
    slurm_localid = int(os.getenv("SLURM_LOCALID"))
    gpus_per_node = torch.cuda.device_count()
    gpu = rank % gpus_per_node
    assert gpu == slurm_localid
    device = f"cuda:{slurm_localid}"
    torch.cuda.set_device(device)
    
    # Initialize DDP.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method="env://")
    if dist.is_initialized(): 
        print(f"Rank {rank}/{world_size}: Process group initialized with torch rank {torch.distributed.get_rank()} and torch world size {torch.distributed.get_world_size()}.")
    
    b = 256 # Set batch size.
    e = 100 # Set number of epochs to be trained.
    
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),
        torchvision.transforms.RandomCrop((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),        
        torchvision.transforms.CenterCrop((64, 64)),            
        torchvision.transforms.ToTensor(),                
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Get distributed dataloaders for training and validation data on all ranks.
    train_loader, valid_loader = get_dataloaders_cifar10_ddp(
        batch_size=b, 
        root='/scratch/hpc-prf-nhrgs/mweiel/data', 
        train_transforms=train_transforms,
        test_transforms=test_transforms
    )
    
    # Get dataloader for test data. 
    # Final testing is only done on root.
    if dist.get_rank() == 0:
        test_dataset = torchvision.datasets.CIFAR10(
            root="/scratch/hpc-prf-nhrgs/mweiel/data",
            train=False,
            transform=test_transforms
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=b,
            shuffle=False
        )
    
    model = AlexNet(num_classes=10).to(device) # Create model and move it to GPU with id rank.
    ddp_model = DDP( # Wrap model with DDP.
        model, 
        device_ids=[slurm_localid], 
        output_device=slurm_localid
    )
    optimizer = torch.optim.SGD(
        ddp_model.parameters(), 
        momentum=0.9, 
        lr=0.1
    ) 
    
    # Train model.
    loss_history, train_acc_history, valid_acc_history = train_model_ddp(
        model=ddp_model,
        num_epochs=e,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer
    )
    
    # Test final model on root.
    if dist.get_rank() == 0:
        test_acc = compute_accuracy_ddp(ddp_model, test_loader) # Compute accuracy on test data.
        print(f'Test accuracy {test_acc :.2f}%')
        
    dist.destroy_process_group()

# MAIN STARTS HERE.    
if __name__ == '__main__':
    main()

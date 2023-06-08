import time
import torch
import os
import numpy as np
import random

def compute_accuracy_ddp(model, data_loader):
    """
    Compute accuracy of model predictions on given labeled data.
    
    Params
    ------
    model : torch.nn.Module
            Model.
    data_loader : torch.utils.data.Dataloader
                  Dataloader.
    device : torch.device
             device to use
    
    Returns
    -------
    float : The model's accuracy on the given dataset in percent.
    """
    with torch.no_grad():

        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):

            features = features.cuda()
            targets = targets.float().cuda()

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1) # Get class with highest score.

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


def get_right_ddp(model, data_loader):
    """
    Compute the number of correctly predicted samples and the overall number of samples in a given dataset.
    
    This function is needed to compute the accuracy over multiple processors in a distributed data-parallel setting.
    
    Params
    ------
    model : torch.nn.Module
            Model.
    data_loader : torch.utils.data.Dataloader
                  Dataloader.
    
    Returns
    -------
    int : The number of correctly predicted samples.
    int : The overall number of samples in the dataset.
    """
    with torch.no_grad():

        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):

            features = features.cuda()
            targets = targets.float().cuda()
            logits = model(features)
            _, predicted_labels = torch.max(logits, 1) # Get class with highest score.

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    num_examples = torch.Tensor([num_examples]).cuda()
    return correct_pred, num_examples

def train_model(
    model, 
    num_epochs, 
    train_loader,
    valid_loader, 
    test_loader, 
    optimizer,
    device, 
    logging_interval=50,
    scheduler=None
):
    """
    Train your model.
    
    Params
    ------
    model : torch.nn.Module
            model to train
    num_epochs : int
                 number of epochs to train
    train_loader : torch.utils.data.Dataloader
                   training dataloader
    valid_loader : torch.utils.data.Dataloader
                   validation dataloader
    test_loader : torch.utils.data.Dataloader
                  testing dataloader
    optimizer : torch.optim.Optimizer
                optimizer to use
    device : torch.device
             device to train on
    logging_interval : int
                       logging interval
    scheduler : torch.optim.lr_scheduler.<scheduler>
                optional learning rate scheduler
                
    Returns
    -------
    [float] : loss history
    [float] : training accuracy history
    [float] : validation accuracy history
    """
    start = time.perf_counter() # Measure training time.

    # Initialize history lists for loss, training accuracy, and validation accuracy.
    loss_history, train_acc_history, valid_acc_history = [], [], [] 

    for epoch in range(num_epochs): # Loop over epochs.

        model.train() # Set model to training mode.
        # Thus, layers like dropout which behave differently on train and test procedures 
        # know what is going on and can behave accordingly. model.train() sets the mode to 
        # train. One might expect this to train model but it does not do that.
        # Call either model.eval() or model.train(mode=False) to tell that you are testing. 
        
        for batch_idx, (features, targets) in enumerate(train_loader): # Loop over mini batches.

            features = features.to(device) # Move features to used device.
            targets = targets.to(device)   # Move targets to used device.

            # Forward and backward pass.
            logits = model(features) # Calculate logits as the model's output.
            loss = torch.nn.functional.cross_entropy(logits, targets) # Calculate cross-entropy loss.
            optimizer.zero_grad() # Zero out gradients.
            # Gradients are accumulated and not overwritten whenever .backward() is called.
            loss.backward() # Calculate gradients of loss w.r.t. model parameters in backward pass.
            optimizer.step() # Perform single optimization step to update model parameters.

            # Logging.
            loss_history.append(loss.item())
            if not batch_idx % logging_interval:
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                      f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                      f'| Loss: {loss:.4f}')

        model.eval() # Set model to evaluation mode.
        
        with torch.no_grad(): # Disable gradient calculateion to reduce memory consumption.
            train_acc = compute_accuracy(model, train_loader, device=device) # Compute accuracy on training data.
            valid_acc = compute_accuracy(model, valid_loader, device=device) # Compute accuracy on validation data.
            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                  f'| Train: {train_acc :.2f}% '
                  f'| Validation: {valid_acc :.2f}%')
            train_acc_history.append(train_acc.item())
            valid_acc_history.append(valid_acc.item())

        elapsed = (time.perf_counter() - start)/60 # Measure training time per epoch.
        print(f'Time elapsed: {elapsed:.2f} min')
        
        if scheduler is not None: 
            scheduler.step(valid_acc_history[-1])
        
    elapsed = (time.perf_counter() - start)/60 # Measure total training time.
    print(f'Total Training Time: {elapsed:.2f} min')

    test_acc = compute_accuracy(model, test_loader, device=device) # Compute accuracy on test data.
    print(f'Test accuracy {test_acc :.2f}%')

    return loss_history, train_acc_history, valid_acc_history


def train_model_ddp(
    model, 
    num_epochs, 
    train_loader, 
    valid_loader, 
    optimizer
):
    """
    Train model in distributed data-parallel fashion.

    Params
    ------
    model : torch.nn.Module
            model to train
    num_epochs : int
                 number of epochs to train
    train_loader : torch.utils.data.Dataloader
                   training dataloader
    valid_loader : torch.utils.data.Dataloader
                   validation dataloader
    optimizer : torch.optim.Optimizer
                optimizer to use
    """
    start = time.perf_counter() # Measure training time.
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    loss_history, train_acc_history, valid_acc_history = [], [], [] # Initialize history lists only on root.
        
    for epoch in range(num_epochs): # Loop over epochs.
        
        train_loader.sampler.set_epoch(epoch)   
        model.train() # Set model to training mode.
 
        for batch_idx, (features, targets) in enumerate(train_loader): # Loop over mini batches.

            features = features.cuda()
            targets = targets.cuda()
            
            # Forward and backward pass.
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # Perform single optimization step to update model parameters.

            # Logging.
            torch.distributed.all_reduce(loss) # Allreduce rank-local mini-batch losses.
            loss /= world_size # Average allreduced rank-local mini-batch losses over all ranks.
            loss_history.append(loss.item()) # Append globally averaged loss of this epoch to history list.
            
            if rank == 0: 
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                      f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                      f'| Averaged Loss: {loss:.4f}')

        model.eval() # Set model to evaluation mode.
        
        with torch.no_grad(): # Disable gradient calculation.
            # Get rank-local numbers of correctly classified and overall samples in training and validation set.
            right_train, num_train = get_right_ddp(model, train_loader) 
            right_valid, num_valid = get_right_ddp(model, valid_loader)

            torch.distributed.all_reduce(right_train)
            torch.distributed.all_reduce(right_valid)
            torch.distributed.all_reduce(num_train)
            torch.distributed.all_reduce(num_valid)
            train_acc = right_train.item() / num_train.item() * 100
            valid_acc = right_valid.item() / num_valid.item() * 100
            train_acc_history.append(train_acc)
            valid_acc_history.append(valid_acc)

            if rank == 0:
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                  f'| Train: {train_acc :.2f}% '
                  f'| Validation: {valid_acc :.2f}%')
                
        elapsed = (time.perf_counter() - start)/60 # Measure training time per epoch.
        Elapsed = torch.Tensor([elapsed]).cuda()
        torch.distributed.all_reduce(Elapsed)
        Elapsed /= world_size
        if rank == 0: 
            print('Time elapsed:', Elapsed.item(), 'min')
                
    elapsed = (time.perf_counter() - start)/60 # Measure total training time.
    Elapsed = torch.Tensor([elapsed]).cuda()
    torch.distributed.all_reduce(Elapsed)
    Elapsed /= world_size
    
    if rank == 0: 
        print('Total Training Time:', Elapsed.item(), 'min')
        torch.save(loss_history, f'loss_{world_size}_gpu.pt')
        torch.save(train_acc_history, f'train_acc_{world_size}_gpu.pt')
        torch.save(valid_acc_history, f'valid_acc_{world_size}_gpu.pt')
        
    return loss_history, train_acc_history, valid_acc_history

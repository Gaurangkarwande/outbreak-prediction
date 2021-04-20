import numpy as np
import torch
import torch.nn as nn


def Solver(device, model, train_loader, optim, criterion, epoch=10, lr=1e-1, print_every=10):
    '''
    The solver function for training your model

    model: your designed model
    train_loader: data loader for training data
    optim: your optimizer
    criterion: criterion for calculating loss, i.e. nn.CrosssEntropyLoss
    epoch: number of training epochs, an epoch means looping through all the data in the datasets
    lr: training learning rate
    print_every: number of epochs to print out loss and accuracies
    '''

    # Send model to GPU for training.
    model.to(device)

    for e in range(epoch):
        loss_epoch = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optim.zero_grad()
            y_pred = model(x)
            # if torch.any(y_pred.isnan()): break
            loss = criterion(y_pred, y)
            # print(loss)
            loss.backward()
            optim.step()
            loss_epoch += loss.item()
        if print_every <= epoch and e % print_every == 0:
            print(f'Epoch {e}: {loss_epoch}')
    return model

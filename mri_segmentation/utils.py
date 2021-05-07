import os
import copy
import random

import numpy as np
import torch
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Running on {device}')


def set_seed(seed):
    """Set seed rng for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def dice_coefficient(outputs, targets, eps=1.0):
    """Calculates the Dice coefficient between the predicted and target masks.

    More information:
        - https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient

    Args:
        outputs (torch.tensor): Outputs of the model (N x 1 x H x W).
        targets (torch.tensor): Manual segmentations (N x H x W).
        eps (float, optional): Smooth parameter (avoid zero divisions). Defaults to 1.

    Returns:
        (float): Dice coefficient.
    """
    outputs = outputs.contiguous().view(-1)
    targets = targets.contiguous().view(-1)

    intersection = (outputs * targets).sum()
    union = outputs.sum() + targets.sum()

    return ((2 * intersection + eps) / (union + eps))


def train(model, trainloader, validloader, optimizer, num_epochs):
    """Trains the model for a given number of epochs. After training, the model with better loss is retrieved.

    Args:
        model (torch.nn.modules.module.Module): Neural network (UNet).
        trainloader (torch.utils.data.dataloader.DataLoader): Training dataset split in batches.
        validloader (torch.utils.data.dataloader.DataLoader): Validation dataset split in batches.
        optimizer (torch.optim): Optimizer to update parameters.
        num_epochs (int): Number of epochs to train the network.
    """
    print(f'{model.__class__.__name__}')

    model.to(device)

    best_loss = float('inf')
    best_model = None
    best_epoch = None

    for epoch in range(num_epochs):
        # Train
        bar_train = tqdm(enumerate(trainloader, 1), total=len(trainloader),
                         desc=f'Epoch {epoch:>2} (Train)')  # Progressbar to show current epoch, loss and accuracy on train
        total_correct_train = 0
        total_loss_train = 0
        total_inputs_train = 0
        model.train()
        for batch, (inputs, labels) in bar_train:
            # Move tensors to gpu
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Initialize gradient
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            dc = dice_coefficient(outputs, labels)

            # Backward pass and optimize
            loss = (1 - dc)
            loss.backward()
            optimizer.step()

            # Show mean loss and accuracy in progressbar
            total_loss_train += loss.item()
            total_correct_train += dc.item()
            total_inputs_train += len(labels)
            bar_train.set_postfix_str(f'loss_train={total_loss_train/total_inputs_train:.4f}, '
                                      f'acc_train={total_correct_train/total_inputs_train:.4f}')

        # Sanity check (all training images were used)
        assert(total_inputs_train == len(trainloader.sampler))

        # Validation
        bar_valid = tqdm(enumerate(validloader, 1), total=len(validloader),
                         desc=f'Epoch {epoch:>2} (Valid)')
        total_correct_valid = 0
        total_loss_valid = 0
        total_inputs_valid = 0
        model.eval()  # Test mode
        with torch.no_grad():
            for batch, (inputs, labels) in bar_valid:
                # Move tensors to gpu
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                dc = dice_coefficient(outputs, labels)

                # Compute loss (no backprop)
                loss = (1 - dc)

                # Show mean loss and accuracy in progressbar
                total_loss_valid += loss.item()
                total_correct_valid += dc.item()
                total_inputs_valid += len(labels)
                bar_valid.set_postfix_str(f'loss_valid={total_loss_valid/total_inputs_valid:.4f}, '
                                          f'acc_valid={total_correct_valid/total_inputs_valid:.4f}')

        # Sanity check (all validation images were used)
        assert(total_inputs_valid == len(validloader.sampler))

        if total_loss_valid / batch < best_loss:
            best_loss = total_loss_valid / batch
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    model.load_state_dict(best_model)
    print(f'Best Loss in Validation: {best_loss:.4f} (Epoch {best_epoch})')

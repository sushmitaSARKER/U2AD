import torch
import numpy as np
import os


def to_flattened_numpy(x):
    
    """Flatten a torch tensor and convert it to numpy.
    Args:
        x: Input torch tensor.   
    Returns:
        np.ndarray: Flattened numpy array.
    """
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))

def get_lr(optimizer):
    """Get the current learning rate from the optimizer.
    Args:
        optimizer: PyTorch optimizer.
    Returns:
        float: Current learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def my_kl_loss(p, q):
    """Compute KL divergence loss between two probability distributions.
    Args:
        p: First probability distribution.
        q: Second probability distribution.
    Returns:
        torch.Tensor: KL divergence loss.
    """
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    """Adjust learning rate based on epoch using exponential decay.
    Args:
        optimizer: PyTorch optimizer.
        epoch: Current epoch number.
        base_lr: Base learning rate.
    """
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    """Early stopping utility to prevent overfitting. Monitors validation loss and stops training when it stops improving."""
    
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(
            path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


def get_data_inverse_scaler(data_centered):
    """Inverse data normalizer."""
    if data_centered:
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x

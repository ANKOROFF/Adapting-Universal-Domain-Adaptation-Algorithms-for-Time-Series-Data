"""
Data augmentation techniques for time series data.
Implements various augmentation methods including jittering, permutation, and scaling.
"""

import torch
import numpy as np


def jitter(x, sigma=0.8):
    """
    Add Gaussian noise to the input time series.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length)
        sigma (float): Standard deviation of the Gaussian noise
        
    Returns:
        torch.Tensor: Augmented time series with added noise
        
    Reference:
        https://arxiv.org/pdf/1706.00527.pdf
    """
    return x + torch.randn_like(x) * sigma


def permutation(x, max_segments=5, seg_mode="random"):
    """
    Randomly permute segments of the time series.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length)
        max_segments (int): Maximum number of segments to split the series into
        seg_mode (str): Segmentation mode - "random" for random split points, "uniform" for equal segments
        
    Returns:
        torch.Tensor: Time series with permuted segments
    """
    orig_steps = np.arange(x.shape[2])
    num_segs = np.random.randint(1, max_segments, (x.shape[0],))

    ret = torch.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                # Randomly select split points
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                # Uniform segmentation
                splits = np.array_split(orig_steps, num_segs[i])
            # Randomly permute segments
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat

    return ret


def scaling(x, sigma=0.1):
    """
    Apply random scaling to the time series.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length)
        sigma (float): Standard deviation of the scaling factor distribution
        
    Returns:
        torch.Tensor: Scaled time series
    """
    # Generate random scaling factors with mean 2 and standard deviation sigma
    scale = torch.randn(x.shape[0], 1, x.shape[2], device=x.device) * sigma + 2
    return x * scale
"""
Visualization utilities for Sinkhorn optimal transport.
Includes functions for visualizing assignments and transport plans between source and target domains.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch 
import sys
sys.path.append('D:\\Transfer-Learning-Time-Series\\Transfer-Learning-Time-Series')
from dataloader.dataloader import data_generator
sns.set_style("whitegrid")
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class

def show_assignments(a, b, P=None, ax=None): 
    """
    Visualize assignments between source and target points using arrows.
    
    Args:
        a (torch.Tensor): Source points coordinates
        b (torch.Tensor): Target points coordinates
        P (torch.Tensor, optional): Transport plan matrix
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on
    """
    if P is not None:
        norm_P = P/P.max()
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                ax.arrow(a[i, 0], a[i, 1], b[j, 0] - a[i, 0], b[j, 1] - a[i, 1],
                         alpha=norm_P[i,j].item(), color="k")

    ax = plt if ax is None else ax
    
    # ax.scatter(*a.t(), color="red")
    # ax.scatter(*b.t(), color="blue")

    
data_path = 'D:\\Transfer-Learning-Time-Series\\Transfer-Learning-Time-Series\\data\\WISDM'
hparams = {"batch_size":32, \
           'learning_rate': 0.01,    'src_cls_loss_wt': 1,   'coral_wt': 1}

def compl_mul1d(input, weights):
    """
    Perform complex 1D convolution operation.
    
    Args:
        input (torch.Tensor): Input tensor of shape (batch, in_channel, x)
        weights (torch.Tensor): Weight tensor of shape (in_channel, out_channel, x)
        
    Returns:
        torch.Tensor: Output tensor of shape (batch, out_channel, x)
    """
    return torch.einsum("bix,iox->box", input, weights)
    
def get_configs():
    """
    Get dataset and hyperparameter configurations for WISDM dataset.
    
    Returns:
        tuple: (dataset_configs, hparams_class) containing configuration objects
    """
    dataset_class = get_dataset_class('WISDM')
    hparams_class = get_hparams_class('WISDM')
    return dataset_class(), hparams_class()

# Load configurations and data
dataset_configs, hparams_class = get_configs()
dataset_configs.final_out_channels = dataset_configs.final_out_channels

# Generate data loaders for source and target domains
src_train_dl, src_test_dl = data_generator(data_path, '7' ,dataset_configs,hparams)
trg_train_dl, trg_test_dl = data_generator(data_path, '18', dataset_configs,hparams)

# Create figure for visualization
f = plt.figure(figsize=(9, 6))
import matplotlib.cm as cm

# Process source and target data
n = 32
xs,ys = next(iter(src_train_dl))
xt,yt = next(iter(trg_train_dl))

# Compute FFT for source and target data
xs_ft = torch.fft.rfft(xs, norm='ortho')
xt_ft = torch.fft.rfft(xt, norm='ortho')

# Process source domain features
outs_ft = torch.zeros(32, 3, xs.size(-1)//2 + 1,  device=xs.device, dtype=torch.cfloat)
weights = torch.rand(3,3,32,dtype=torch.cfloat)
outs_ft[:, :, :32] = compl_mul1d(xs_ft[:, :, :32], weights) 
rs = outs_ft[:, :, :32].abs().mean([1,2])
ps = outs_ft[:, :, :32].angle().mean([1,2])+torch.acos(torch.zeros(1)).item() * 2

# Process target domain features
outt_ft = torch.zeros(32, 3, xt.size(-1)//2 + 1,  device=xt.device, dtype=torch.cfloat)
outt_ft[:, :, :32] = compl_mul1d(xt_ft[:, :, :32], weights) 
rt = outt_ft[:,:,:32].abs().mean([1,2])
pt = outt_ft[:,:,:32].angle().mean([1,2])+torch.acos(torch.zeros(1)).item() * 2

# Convert to polar coordinates
polar_s = torch.vstack([rs,ps]).t()
polar_t = torch.vstack([rt,pt]).t()
show_assignments(polar_s, polar_t)

# Compute optimal transport plan
from sinkhorn import SinkhornSolver
epsilon = 10**(-(2*2))
solver = SinkhornSolver(epsilon=epsilon, iterations=10000)
cost, pi = solver.forward(polar_s, polar_t)

# Create polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.scatter(pt, rt,color="red")
ax.scatter(ps, rs,color="blue")
# show_assignments(polar_s, polar_t, pi, ax)
# ax.set_rmax(2)
ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)

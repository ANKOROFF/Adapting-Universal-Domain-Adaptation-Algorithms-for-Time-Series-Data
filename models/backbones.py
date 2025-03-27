"""
Neural network backbone architectures for time series classification.
Includes CNN, TCN, and attention-based variants optimized for domain adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Convolutional Neural Network backbone for time series classification.
    Uses three convolutional blocks with batch normalization and max pooling.
    
    Args:
        configs: Configuration object containing model parameters
    """
    def __init__(self, configs):
        super(CNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv1d(configs.input_channels, 32, kernel_size=8, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, padding=3)
        self.bn3 = nn.BatchNorm1d(configs.final_out_channels)
        
        # Common layers
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.first_forward = True
        
    def forward(self, x):
        """
        Forward pass of the CNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length)
            
        Returns:
            torch.Tensor: Flattened feature vector
        """
        if hasattr(self, 'verbose') and self.verbose and self.first_forward:
            print("\nNetwork architecture:")
            print(f"Input shape: {x.shape}")
            
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        if hasattr(self, 'verbose') and self.verbose and self.first_forward:
            print(f"After conv1: {x.shape}")
            
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        if hasattr(self, 'verbose') and self.verbose and self.first_forward:
            print(f"After conv2: {x.shape}")
            
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        if hasattr(self, 'verbose') and self.verbose and self.first_forward:
            print(f"After conv3: {x.shape}")
            
        # Flatten and dropout
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        if hasattr(self, 'verbose') and self.verbose and self.first_forward:
            print(f"Final output shape: {x.shape}\n")
            self.first_forward = False
            
        return x

class CNNUniOT(nn.Module):
    """
    CNN variant optimized for UniOT domain adaptation.
    Similar to CNN but uses non-inplace ReLU for gradient computation.
    
    Args:
        configs: Configuration object containing model parameters
    """
    def __init__(self, configs):
        super(CNNUniOT, self).__init__()
        # Three convolutional blocks without batch normalization
        self.conv1 = nn.Conv1d(configs.input_channels, 32, kernel_size=8, stride=1, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=3)
        self.conv3 = nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, padding=3)
        
        # Common layers
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.first_forward = True
        
    def forward(self, x):
        """
        Forward pass of the CNNUniOT.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length)
            
        Returns:
            torch.Tensor: Flattened feature vector
        """
        if hasattr(self, 'verbose') and self.verbose and self.first_forward:
            print("\nNetwork architecture:")
            print(f"Input shape: {x.shape}")
            
        # First block with non-inplace ReLU
        x = self.conv1(x.clone())
        x = F.relu(x, inplace=False)
        x = self.pool(x)
        
        if hasattr(self, 'verbose') and self.verbose and self.first_forward:
            print(f"After conv1: {x.shape}")
            
        # Second block
        x = self.conv2(x.clone())
        x = F.relu(x, inplace=False)
        x = self.pool(x)
        
        if hasattr(self, 'verbose') and self.verbose and self.first_forward:
            print(f"After conv2: {x.shape}")
            
        # Third block
        x = self.conv3(x.clone())
        x = F.relu(x, inplace=False)
        x = self.pool(x)
        
        if hasattr(self, 'verbose') and self.verbose and self.first_forward:
            print(f"After conv3: {x.shape}")
            
        # Flatten and dropout
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        
        if hasattr(self, 'verbose') and self.verbose and self.first_forward:
            print(f"Final output shape: {x.shape}\n")
            self.first_forward = False
            
        return x

class TCN(nn.Module):
    """
    Temporal Convolutional Network backbone.
    Uses dilated convolutions to capture long-range dependencies.
    
    Args:
        configs: Configuration object containing model parameters
    """
    def __init__(self, configs):
        super(TCN, self).__init__()
        # TCN layers with increasing dilation rates
        self.tcn1 = nn.Conv1d(configs.input_channels, 32, kernel_size=8, stride=1, padding=3, dilation=1)
        self.tcn2 = nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=6, dilation=2)
        self.tcn3 = nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, padding=12, dilation=4)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass of the TCN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length)
            
        Returns:
            torch.Tensor: Flattened feature vector
        """
        # Three TCN blocks with increasing receptive field
        x = self.tcn1(x)
        x = F.relu(x, inplace=False)
        
        x = self.tcn2(x)
        x = F.relu(x, inplace=False)
        
        x = self.tcn3(x)
        x = F.relu(x, inplace=False)
        
        # Flatten and dropout
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        
        return x

class TCNUniOT(nn.Module):
    """
    TCN variant optimized for UniOT domain adaptation.
    Similar to TCN but uses non-inplace ReLU for gradient computation.
    
    Args:
        configs: Configuration object containing model parameters
    """
    def __init__(self, configs):
        super(TCNUniOT, self).__init__()
        # TCN layers with increasing dilation rates
        self.tcn1 = nn.Conv1d(configs.input_channels, 32, kernel_size=8, stride=1, padding=3, dilation=1)
        self.tcn2 = nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=6, dilation=2)
        self.tcn3 = nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, padding=12, dilation=4)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass of the TCNUniOT.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length)
            
        Returns:
            torch.Tensor: Flattened feature vector
        """
        # Three TCN blocks with non-inplace ReLU
        x = self.tcn1(x.clone())
        x = F.relu(x, inplace=False)
        
        x = self.tcn2(x.clone())
        x = F.relu(x, inplace=False)
        
        x = self.tcn3(x.clone())
        x = F.relu(x, inplace=False)
        
        # Flatten and dropout
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        
        return x

class TCNAttention(nn.Module):
    """
    Temporal Convolutional Network with self-attention mechanism.
    Combines TCN's temporal modeling with transformer-style attention.
    
    Args:
        configs: Configuration object containing model parameters
    """
    def __init__(self, configs):
        super(TCNAttention, self).__init__()
        # TCN layers with increasing dilation
        self.tcn1 = nn.Conv1d(configs.input_channels, 32, kernel_size=8, stride=1, padding=3, dilation=1)
        self.tcn2 = nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=6, dilation=2)
        self.tcn3 = nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, padding=12, dilation=4)
        
        # Self-attention layers for each TCN block
        self.attention1 = nn.MultiheadAttention(32, num_heads=4, dropout=0.1)
        self.attention2 = nn.MultiheadAttention(64, num_heads=4, dropout=0.1)
        self.attention3 = nn.MultiheadAttention(configs.final_out_channels, num_heads=4, dropout=0.1)
        
        # Layer normalization for residual connections
        self.norm1 = nn.LayerNorm(32)
        self.norm2 = nn.LayerNorm(64)
        self.norm3 = nn.LayerNorm(configs.final_out_channels)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass of the TCNAttention network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length)
            
        Returns:
            torch.Tensor: Flattened feature vector
        """
        # First TCN block with attention
        x = self.tcn1(x)
        x = F.relu(x, inplace=False)
        # Reshape for attention: (batch, channels, seq_len) -> (seq_len, batch, channels)
        x_att = x.permute(2, 0, 1)
        x_att, _ = self.attention1(x_att, x_att, x_att)
        x_att = x_att.permute(1, 2, 0)  # Back to (batch, channels, seq_len)
        x = x + x_att
        x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Second TCN block with attention
        x = self.tcn2(x)
        x = F.relu(x, inplace=False)
        x_att = x.permute(2, 0, 1)
        x_att, _ = self.attention2(x_att, x_att, x_att)
        x_att = x_att.permute(1, 2, 0)
        x = x + x_att
        x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Third TCN block with attention
        x = self.tcn3(x)
        x = F.relu(x, inplace=False)
        x_att = x.permute(2, 0, 1)
        x_att, _ = self.attention3(x_att, x_att, x_att)
        x_att = x_att.permute(1, 2, 0)
        x = x + x_att
        x = self.norm3(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Flatten and dropout
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        
        return x 
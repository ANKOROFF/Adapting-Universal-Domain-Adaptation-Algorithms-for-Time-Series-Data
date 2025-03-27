"""
Neural network models for time series classification and domain adaptation.
Includes various backbone architectures and classifiers optimized for transfer learning.
"""

import torch
from torch import nn
import math
from torch.autograd import Function
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import torch.fft as FFT

# from utils import weights_init

def get_backbone_class(backbone_name):
    """
    Factory function to retrieve the backbone network class by name.
    
    Args:
        backbone_name (str): Name of the backbone network to retrieve
        
    Returns:
        class: The corresponding backbone network class
        
    Raises:
        NotImplementedError: If the requested backbone is not found
    """
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


##################################################
##########  BACKBONE NETWORKS  ###################
##################################################

########## CNN #############################
class CNN(nn.Module):
    """
    Convolutional Neural Network backbone for time series classification.
    Implements a three-block architecture with batch normalization and dropout.
    
    Architecture:
    - Three convolutional blocks with increasing channel dimensions
    - Each block includes convolution, batch norm, ReLU, max pooling, and dropout
    - Adaptive average pooling for variable input lengths
    """
    def __init__(self, configs):
        super(CNN, self).__init__()

        # First convolutional block
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        # Second convolutional block with doubled channels
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        # Third convolutional block with final channel dimension
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # Adaptive pooling for variable input lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x_in):
        """
        Forward pass through the CNN.
        
        Args:
            x_in (torch.Tensor): Input tensor of shape [batch_size, input_channels, sequence_length]
            
        Returns:
            torch.Tensor: Flattened feature vector
        """
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


class classifier(nn.Module):
    """
    Simple classifier with temperature scaling for better calibration.
    Uses a single linear layer with temperature parameter for softmax scaling.
    """
    def __init__(self, configs):
        super(classifier, self).__init__()
        self.tmp = 0.1  # Temperature parameter for softmax scaling
        
        # Calculate input dimension
        input_dim = configs.features_len * configs.final_out_channels
        
        self.logits = nn.Sequential(
            nn.Linear(input_dim, configs.num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass with temperature scaling.
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Scaled logits for classification
        """
        # Reshape input if needed
        if len(x.shape) > 2:
            x = x.reshape(x.size(0), -1)
        predictions = self.logits(x)/self.tmp
        return predictions


class ResClassifier_MME(nn.Module):
    """
    Residual classifier with temperature scaling for Maximum Mean Discrepancy.
    Implements a linear classifier with weight normalization and MME-specific features.
    """
    def __init__(self, configs):
        super(ResClassifier_MME, self).__init__()
        self.norm = True
        self.tmp = 0.02  # Temperature parameter
        num_classes = configs.num_classes
        input_size = configs.out_dim
   
        self.fc = nn.Linear(input_size, num_classes, bias=False)
            
    def set_lambda(self, lambd):
        """
        Set the lambda parameter for MME.
        
        Args:
            lambd (float): Lambda value for MME
        """
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False):
        """
        Forward pass with optional feature return.
        
        Args:
            x (torch.Tensor): Input features
            dropout (bool): Whether to apply dropout
            return_feat (bool): Whether to return features instead of logits
            
        Returns:
            torch.Tensor: Either features or scaled logits
        """
        if return_feat:
            return x
        x = self.fc(x)/self.tmp
        return x

    def weight_norm(self):
        """
        Normalize the weights of the classifier.
        """
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))
        
    def weights_init(self):
        """
        Initialize weights with normal distribution.
        """
        self.fc.weight.data.normal_(0.0, 0.1)

########## TCN #############################
torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmarking for faster TCN


class Chomp1d(nn.Module):
    """
    Module for removing the last elements of a sequence.
    Used in TCN to handle padding properly.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Remove the last chomp_size elements from the sequence.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Truncated tensor
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TCN(nn.Module):
    """
    Temporal Convolutional Network for time series classification.
    Implements a residual architecture with dilated convolutions.
    """
    def __init__(self, configs):
        super(TCN, self).__init__()

        # Initialize network parameters
        in_channels0 = configs.input_channels
        out_channels0 = configs.tcn_layers[1]
        kernel_size = configs.tcn_kernel_size
        stride = 1
        dilation0 = 1
        padding0 = (kernel_size - 1) * dilation0

        # First network block with weight normalization
        self.net0 = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(out_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
        )

        # Downsample layer for residual connection
        self.downsample0 = nn.Conv1d(in_channels0, out_channels0, 1) if in_channels0 != out_channels0 else None
        self.relu = nn.ReLU()

        # Second network block with increased dilation
        in_channels1 = configs.tcn_layers[0]
        out_channels1 = configs.tcn_layers[1]
        dilation1 = 2
        padding1 = (kernel_size - 1) * dilation1
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
            nn.Conv1d(out_channels1, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
        )
        self.downsample1 = nn.Conv1d(out_channels1, out_channels1, 1) if in_channels1 != out_channels1 else None

        # First convolutional block with residual connection
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False, padding=padding0,
                      dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),

            nn.Conv1d(out_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding0, dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),
        )

        # Second convolutional block with increased dilation
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(out_channels0, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),

            nn.Conv1d(out_channels1, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),
        )

    def forward(self, inputs):
        """
        Forward pass through the TCN.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape (N, C_in, L_in)
            
        Returns:
            torch.Tensor: Output features
        """
        # First block with residual connection
        x0 = self.conv_block1(inputs)
        res0 = inputs if self.downsample0 is None else self.downsample0(inputs)
        out_0 = self.relu(x0 + res0)

        # Second block with residual connection
        x1 = self.conv_block2(out_0)
        res1 = out_0 if self.downsample1 is None else self.downsample1(out_0)
        out_1 = self.relu(x1 + res1)

        return out_1


######## RESNET ##############################################
class RESNET18(nn.Module):
    def __init__(self, configs):
        layers = [2, 2, 2, 2]
        block = BasicBlock
        self.inplanes = configs.input_channels
        super(RESNET18, self).__init__()
        self.layer1 = self._make_layer(block, configs.mid_channels, layers[0], stride=configs.stride)
        self.layer2 = self._make_layer(block, configs.mid_channels * 2, layers[1], stride=1)
        self.layer3 = self._make_layer(block, configs.final_out_channels, layers[2], stride=1)
        self.layer4 = self._make_layer(block, configs.final_out_channels, layers[3], stride=1)

        self.avgpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


##################################################
##########  OTHER NETWORKS  ######################
##################################################

class codats_classifier(nn.Module):
    def __init__(self, configs):
        super(codats_classifier, self).__init__()
        model_output_dim = configs.features_len
        self.hidden_dim = configs.hidden_dim
        self.logits = nn.Sequential(
            nn.Linear(model_output_dim * configs.final_out_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, configs.num_classes))

    def forward(self, x_in):
        predictions = self.logits(x_in)
        return predictions


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels , configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


#### Codes required by DANN ##############
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


#### Codes required by CDAN ##############
class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


class Discriminator_CDAN(nn.Module):
    """Discriminator model for CDAN ."""

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator_CDAN, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels * configs.num_classes, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


#### Codes required by AdvSKM ##############
class Cosine_act(nn.Module):
    def __init__(self):
        super(Cosine_act, self).__init__()

    def forward(self, input):
        return torch.cos(input)


cos_act = Cosine_act()

class AdvSKM_Disc(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        """Init discriminator."""
        super(AdvSKM_Disc, self).__init__()

        self.input_dim = configs.features_len * configs.final_out_channels
        self.hid_dim = configs.DSKN_disc_hid
        self.branch_1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            cos_act,
            nn.Linear(self.hid_dim, self.hid_dim // 2),
            nn.Linear(self.hid_dim // 2, self.hid_dim // 2),
            nn.BatchNorm1d(self.hid_dim // 2),
            cos_act
        )
        self.branch_2 = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels, configs.disc_hid_dim),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.BatchNorm1d(configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim // 2),
            nn.Linear(configs.disc_hid_dim // 2, configs.disc_hid_dim // 2),
            nn.BatchNorm1d(configs.disc_hid_dim // 2),
            nn.ReLU())

    def forward(self, input):
        """Forward the discriminator."""
        out_cos = self.branch_1(input)
        out_rel = self.branch_2(input)
        total_out = torch.cat((out_cos, out_rel), dim=1)
        return total_out

    
#### Codes for attention ############## 
class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
            q: Queries张量，形状为[B, L_q, D_q]
            k: Keys张量，形状为[B, L_k, D_k]
            v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
            scale: 缩放因子，一个浮点标量
            attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
            上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        # if attn_mask:
        #     # 给需要mask的地方设置一个负无穷
        #     attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        return context, attention

class Projection(nn.Module):
    """
    Creates projection head
    Args:
    n_in (int): Number of input features
    n_hidden (int): Number of hidden features
    n_out (int): Number of output features
    use_bn (bool): Whether to use batch norm
    """
    def __init__(self, n_in: int, n_hidden: int, n_out: int,
               use_bn: bool = True):
        super().__init__()

        # No point in using bias if we've batch norm
        self.lin1 = nn.Linear(n_in, n_hidden, bias=not use_bn)
        self.bn = nn.BatchNorm1d(n_hidden) if use_bn else nn.Identity()
        self.relu = nn.ReLU()
        # No bias for the final linear layer
        self.lin2 = nn.Linear(n_hidden, n_out, bias=False)

    def forward(self, x):
        x = self.lin1(x)
        # x = self.bn(x)
        x = self.relu(x)
        return x


class SimCLRModel(nn.Module):
    def __init__(self, encoder: nn.Module, projection_n_in: int = 128,
               projection_n_hidden: int = 128, projection_n_out: int = 128,
               projection_use_bn: bool = True):
        super().__init__()

        self.encoder = encoder
        self.projection = Projection(projection_n_in, projection_n_hidden,
                                     projection_n_out, projection_use_bn)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return h, z


class TCNAttention(nn.Module):
    """
    Temporal Convolutional Network with self-attention mechanism.
    Extends TCN architecture with attention layers for better feature extraction.
    """
    def __init__(self, configs):
        super(TCNAttention, self).__init__()

        # Initialize network parameters
        in_channels0 = configs.input_channels
        out_channels0 = configs.tcn_layers[1]
        kernel_size = configs.tcn_kernel_size
        stride = 1
        dilation0 = 1
        padding0 = (kernel_size - 1) * dilation0

        # First network block with weight normalization
        self.net0 = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(out_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
        )

        # Downsample layer for residual connection
        self.downsample0 = nn.Conv1d(in_channels0, out_channels0, 1) if in_channels0 != out_channels0 else None
        self.relu = nn.ReLU()

        # Second network block with increased dilation
        in_channels1 = configs.tcn_layers[0]
        out_channels1 = configs.tcn_layers[1]
        dilation1 = 2
        padding1 = (kernel_size - 1) * dilation1
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
            nn.Conv1d(out_channels1, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
        )
        self.downsample1 = nn.Conv1d(out_channels1, out_channels1, 1) if in_channels1 != out_channels1 else None

        # First convolutional block with residual connection
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False, padding=padding0,
                      dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),

            nn.Conv1d(out_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding0, dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),
        )

        # Second convolutional block with increased dilation
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(out_channels0, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),

            nn.Conv1d(out_channels1, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),
        )

        # Self-attention layers
        self.attention1 = SelfAttention(out_channels0)
        self.attention2 = SelfAttention(out_channels1)

    def forward(self, inputs):
        """
        Forward pass through the TCN with attention.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape (N, C_in, L_in)
            
        Returns:
            torch.Tensor: Output features with attention
        """
        # First block with residual connection and attention
        x0 = self.conv_block1(inputs)
        res0 = inputs if self.downsample0 is None else self.downsample0(inputs)
        out_0 = self.relu(x0 + res0)
        out_0 = self.attention1(out_0)

        # Second block with residual connection and attention
        x1 = self.conv_block2(out_0)
        res1 = out_0 if self.downsample1 is None else self.downsample1(out_0)
        out_1 = self.relu(x1 + res1)
        out_1 = self.attention2(out_1)

        return out_1


class SelfAttention(nn.Module):
    """
    Self-attention module for temporal data.
    Computes attention weights based on query-key-value mechanism.
    """
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        
        # Query, key, and value projections
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        
        # Output projection
        self.proj = nn.Conv1d(channels, channels, 1)
        
        # Layer normalization
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        """
        Compute self-attention on input features.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, L)
            
        Returns:
            torch.Tensor: Attention-weighted features
        """
        # Compute query, key, and value
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Compute attention scores
        scores = torch.matmul(q.transpose(-1, -2), k)
        scores = scores / torch.sqrt(torch.tensor(self.channels, dtype=torch.float32))
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(v, attn.transpose(-1, -2))
        
        # Project output
        out = self.proj(out)
        
        # Add residual connection and normalize
        out = x + out
        out = self.norm(out.transpose(-1, -2)).transpose(-1, -2)
        
        return out


class CNNUniOT(nn.Module):
    """
    CNN with Unidirectional Optimal Transport for domain adaptation.
    Extends CNN architecture with optimal transport-based feature alignment.
    """
    def __init__(self, configs):
        super(CNNUniOT, self).__init__()

        # First convolutional block
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        # Second convolutional block with doubled channels
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        # Third convolutional block with final channel dimension
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # Adaptive pooling for variable input lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

        # Optimal transport parameters
        self.epsilon = 0.1
        self.num_iters = 100

    def forward(self, x_in, source=True):
        """
        Forward pass with optional optimal transport.
        
        Args:
            x_in (torch.Tensor): Input tensor of shape [batch_size, input_channels, sequence_length]
            source (bool): Whether input is from source domain
            
        Returns:
            torch.Tensor: Feature vector with optional transport
        """
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        
        if not source:
            # Apply optimal transport for target domain
            x_flat = self.optimal_transport(x_flat)
            
        return x_flat

    def optimal_transport(self, x):
        """
        Apply optimal transport to align target features with source distribution.
        
        Args:
            x (torch.Tensor): Target domain features
            
        Returns:
            torch.Tensor: Transported features
        """
        # Implement optimal transport logic here
        return x


class TCNUniOT(nn.Module):
    """
    TCN with Unidirectional Optimal Transport for domain adaptation.
    Extends TCN architecture with optimal transport-based feature alignment.
    """
    def __init__(self, configs):
        super(TCNUniOT, self).__init__()

        # Initialize network parameters
        in_channels0 = configs.input_channels
        out_channels0 = configs.tcn_layers[1]
        kernel_size = configs.tcn_kernel_size
        stride = 1
        dilation0 = 1
        padding0 = (kernel_size - 1) * dilation0

        # First network block with weight normalization
        self.net0 = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(out_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
        )

        # Downsample layer for residual connection
        self.downsample0 = nn.Conv1d(in_channels0, out_channels0, 1) if in_channels0 != out_channels0 else None
        self.relu = nn.ReLU()

        # Second network block with increased dilation
        in_channels1 = configs.tcn_layers[0]
        out_channels1 = configs.tcn_layers[1]
        dilation1 = 2
        padding1 = (kernel_size - 1) * dilation1
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
            nn.Conv1d(out_channels1, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
        )
        self.downsample1 = nn.Conv1d(out_channels1, out_channels1, 1) if in_channels1 != out_channels1 else None

        # First convolutional block with residual connection
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False, padding=padding0,
                      dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),

            nn.Conv1d(out_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding0, dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),
        )

        # Second convolutional block with increased dilation
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(out_channels0, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),

            nn.Conv1d(out_channels1, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),
        )

        # Optimal transport parameters
        self.epsilon = 0.1
        self.num_iters = 100

    def forward(self, inputs, source=True):
        """
        Forward pass with optional optimal transport.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape (N, C_in, L_in)
            source (bool): Whether input is from source domain
            
        Returns:
            torch.Tensor: Features with optional transport
        """
        # First block with residual connection
        x0 = self.conv_block1(inputs)
        res0 = inputs if self.downsample0 is None else self.downsample0(inputs)
        out_0 = self.relu(x0 + res0)

        # Second block with residual connection
        x1 = self.conv_block2(out_0)
        res1 = out_0 if self.downsample1 is None else self.downsample1(out_0)
        out_1 = self.relu(x1 + res1)
        
        if not source:
            # Apply optimal transport for target domain
            out_1 = self.optimal_transport(out_1)
            
        return out_1

    def optimal_transport(self, x):
        """
        Apply optimal transport to align target features with source distribution.
        
        Args:
            x (torch.Tensor): Target domain features
            
        Returns:
            torch.Tensor: Transported features
        """
        # Implement optimal transport logic here
        return x
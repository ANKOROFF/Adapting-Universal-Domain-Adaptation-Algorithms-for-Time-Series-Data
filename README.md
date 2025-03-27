
# Adapting-Universal-Domain-Adaptation-Algorithms-for-Time-Series-Data

## Overview

RAINCOAT (Robust Adaptation for Inertial Networks with Cross-domain Optimal Alignment and Transfer) is a comprehensive framework for domain adaptation in Human Activity Recognition (HAR) based on Inertial Measurement Units (IMU) sensor data.

The project implements several state-of-the-art domain adaptation methods:
- RAINCOAT - our innovative three-stage approach (Align-Correct-Inference)
- UniOT - unidirectional optimal transport-based adaptation method
- DANCE - neighborhood consistency and entropy minimization method
- UAN - universal adaptation network
- OVANet - one-versus-all adaptation approach

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Domain Adaptation Methods](#domain-adaptation-methods)
- [Neural Network Architectures](#neural-network-architectures)
- [Running Experiments](#running-experiments)
- [Evaluation Methodology](#evaluation-methodology)
- [Results](#results)
- [Visualization](#visualization)
- [Hyperparameter Configuration](#hyperparameter-configuration)
- [Extension](#extension)
- [Device Compatibility](#device-compatibility)
- [Debugging and Testing](#debugging-and-testing)
- [References](#references)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU acceleration)
- Additional dependencies in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ANKOROFF/Adapting-Universal-Domain-Adaptation-Algorithms-for-Time-Series-Data
cd Raincoat
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Prepare the data:
```bash
python check_data.py
```

## Repository Structure

```
Raincoat/
├── algorithms/            # Domain adaptation method implementations
│   ├── base.py            # Base class for all algorithms
│   ├── RAINCOAT.py        # RAINCOAT method implementation
│   ├── UniOT.py           # UniOT method implementation
│   ├── DANCE.py           # DANCE method implementation
│   ├── UAN.py             # UAN method implementation
│   ├── OVANet.py          # OVANet method implementation
│   ├── algorithm_factory.py # Factory for creating algorithm instances
│   └── utils.py           # Utility functions for algorithms
├── configs/               # Configuration files
│   ├── HHAR_SA.py         # Configuration for HHAR_SA dataset
│   ├── WISDM.py           # Configuration for WISDM dataset
│   ├── HAR.py             # General configurations for HAR
│   ├── data_model_configs.py # Model and data configurations
│   ├── hparams.py         # Hyperparameters for algorithms
│   └── config_factory.py  # Factory for loading configurations
├── dataloader/            # Data loaders
│   ├── data_loader.py     # Loaders for different datasets
│   └── data_preprocessing.py # Data preprocessing
├── models/                # Neural network architectures
│   ├── augmentations.py   # Data augmentation methods
│   ├── backbones.py       # Backbone architectures (CNN, TCN, TCNAttention)
│   ├── loss.py            # Loss functions
│   ├── model_factory.py   # Factory for creating models
│   ├── models.py          # Main models
│   └── sinkhorn_vis.py    # Optimal transport visualization
├── trainers/              # Trainer implementations
│   └── trainer.py         # Main class for training models
├── utils/                 # Utility functions
│   ├── logger.py          # Logging
│   └── visualization.py   # Visualization
├── data/                  # Directory for datasets
├── results/               # Directory for saving results
├── main.py                # Main script for running experiments
├── results_aggregation.py # Script for running experiments and aggregating results
├── check_data.py          # Data checking and analysis
├── requirements.txt       # Project dependencies
└── README.md              # Current file
```

## Datasets

### HHAR_SA (Heterogeneity Human Activity Recognition, Samsung)

Dataset contains human activity data collected from Samsung smartphone accelerometers.

- **Activities**: walking, walking upstairs, walking downstairs, sitting, standing, lying
- **Domains**: different users (9 domains total)
- **Sensors**: tri-axial accelerometer (x, y, z)
- **Sampling rate**: 50 Hz
- **Features**: high variability between users

### WISDM (Wireless Sensor Data Mining)

Dataset contains human activity data collected from various devices.

- **Activities**: similar to HHAR_SA
- **Domains**: different users and devices (36 domains total)
- **Sensors**: tri-axial accelerometer (x, y, z)
- **Sampling rate**: 20 Hz
- **Features**: high variability between devices and users

## Domain Adaptation Methods

### RAINCOAT

RAINCOAT implements a three-stage approach for domain adaptation:

1. **Stage 1 - Align**: alignment of source and target distributions
   - Uses Sinkhorn algorithm for optimal transport
   - Applies regularization with data reconstruction
   - Trains classifier on source domain

2. **Stage 2 - Correct**: correction of alignment errors
   - Focuses on minimizing reconstruction error on target domain
   - Preserves prototype drift for each class

3. **Stage 3 - Inference**: intelligent classification
   - Applies DIP-test for multi-modality detection
   - Uses K-means for separating common and private modalities
   - Adapts prediction strategy based on modalities

Key advantages:
- Processing of multi-modal distributions
- Robustness to private domain characteristics
- Effective alignment with information preservation

**Mathematical Formulation**:

The RAINCOAT optimization objective can be formulated as:

$\mathcal{L}_{\text{RAINCOAT}} = \mathcal{L}_{\text{cls}} + \lambda_{\text{rec}} \mathcal{L}_{\text{rec}} + \lambda_{\text{sink}} \mathcal{L}_{\text{sink}} + \lambda_{\text{contrast}} \mathcal{L}_{\text{contrast}}$

where:
- $\mathcal{L}_{\text{cls}}$ is the cross-entropy loss on source domain: $\mathcal{L}_{\text{cls}} = -\sum_{i=1}^{n_s} \sum_{c=1}^{C} y_i^c \log(\hat{y}_i^c)$
- $\mathcal{L}_{\text{rec}}$ is the reconstruction loss: $\mathcal{L}_{\text{rec}} = \frac{1}{n_s} \sum_{i=1}^{n_s} \|x_i - \hat{x}_i\|^2$
- $\mathcal{L}_{\text{sink}}$ is the Sinkhorn loss: $\mathcal{L}_{\text{sink}} = \langle \mathbf{P}, \mathbf{C} \rangle$, where $\mathbf{P}$ is the optimal transport plan computed via Sinkhorn algorithm
- $\mathcal{L}_{\text{contrast}}$ is the contrastive loss for semantic alignment

The DIP-test for multi-modality detection is defined as:
$$\text{DIP}(x) = \sup_{t \in [0,1]} |F_n(t) - t|$$
where $F_n$ is the empirical CDF of the data.

### UniOT

UniOT (Unidirectional Optimal Transport) implements adaptation based on unidirectional optimal transport:

- Computes transport plan from source to target using Sinkhorn algorithm
- Uses feature normalization for stability
- Applies unbalanced optimal transport to account for different domain sizes
- Implements adaptive filling for class balance

**Mathematical Formulation**:

The UniOT optimization objective:

$$\mathcal{L}_{\text{UniOT}} = \mathcal{L}_{\text{cls}} + \lambda_{\text{ot}} \mathcal{L}_{\text{ot}}$$

The optimal transport loss $\mathcal{L}_{\text{ot}}$ is computed as:

$$\mathcal{L}_{\text{ot}} = \min_{\mathbf{P} \in \Pi(\mu_s, \mu_t)} \langle \mathbf{P}, \mathbf{C} \rangle - \epsilon H(\mathbf{P})$$

where:
- $\Pi(\mu_s, \mu_t)$ is the set of transport plans
- $\mathbf{C}$ is the cost matrix
- $\epsilon$ is the regularization parameter
- $H(\mathbf{P})$ is the entropy of the transport plan

The Sinkhorn algorithm solves this by iterative updates:

$$\mathbf{u}^{l+1} = \frac{\mathbf{a}}{\mathbf{K}\mathbf{v}^l}, \quad \mathbf{v}^{l+1} = \frac{\mathbf{b}}{\mathbf{K}^T\mathbf{u}^{l+1}}$$

where $\mathbf{K} = e^{-\mathbf{C}/\epsilon}$.

### DANCE

DANCE (Domain Adaptation with Neighborhood Consistency and Entropy minimization):

- Ensures neighborhood consistency between domains
- Minimizes entropy of predictions on target domain
- Uses margin loss for improved class separation
- Includes mechanisms to protect against negative transfer

**Mathematical Formulation**:

The DANCE optimization objective:

$$\mathcal{L}_{\text{DANCE}} = \mathcal{L}_{\text{cls}} + \lambda_{\text{nc}} \mathcal{L}_{\text{nc}} + \lambda_{\text{ent}} \mathcal{L}_{\text{ent}}$$

The neighborhood consistency loss $\mathcal{L}_{\text{nc}}$ is:

$$\mathcal{L}_{\text{nc}} = \frac{1}{n_t} \sum_{i=1}^{n_t} \sum_{j \in \mathcal{N}_k(i)} \|\hat{y}_i - \hat{y}_j\|^2 w_{ij}$$

where $\mathcal{N}_k(i)$ is the set of k-nearest neighbors of sample $i$, and $w_{ij}$ is the similarity weight.

The entropy minimization loss $\mathcal{L}_{\text{ent}}$ is:

$$\mathcal{L}_{\text{ent}} = -\frac{1}{n_t} \sum_{i=1}^{n_t} \sum_{c=1}^{C} \hat{y}_i^c \log(\hat{y}_i^c)$$

### UAN

UAN (Universal Adaptation Network):

- Uses domain discriminator to determine sample relevance
- Implements weighted transfer based on entropy
- Includes self-training mechanism on target domain
- Adaptively adjusts weight of each sample

**Mathematical Formulation**:

The UAN optimization objective:

$$\mathcal{L}_{\text{UAN}} = \mathcal{L}_{\text{cls}} + \lambda_{\text{d}} \mathcal{L}_{\text{d}} - \lambda_{\text{ent}} \mathcal{L}_{\text{ent}}$$

The domain discriminator loss $\mathcal{L}_{\text{d}}$ is:

$$\mathcal{L}_{\text{d}} = -\frac{1}{n_s} \sum_{i=1}^{n_s} \log(D(f(x_i^s))) - \frac{1}{n_t} \sum_{i=1}^{n_t} \log(1 - D(f(x_i^t)))$$

where $D$ is the domain discriminator.

The weight assignment based on entropy is:

$$w_i = \begin{cases}
1, & \text{if } H(\hat{y}_i) < \eta \\
0, & \text{otherwise}
\end{cases}$$

where $\eta$ is the entropy threshold.

### OVANet

OVANet (One vs All Network):

- Implements binary classifiers for each class
- Uses contrastive metrics between classes
- Includes regularization for improved generalization
- Applies ensembling for final prediction

**Mathematical Formulation**:

The OVANet optimization objective:

$$\mathcal{L}_{\text{OVANet}} = \mathcal{L}_{\text{ova}} + \lambda_{\text{d}} \mathcal{L}_{\text{d}}$$

The one-vs-all loss $\mathcal{L}_{\text{ova}}$ is:

$$\mathcal{L}_{\text{ova}} = -\frac{1}{n_s} \sum_{i=1}^{n_s} \sum_{c=1}^{C} \left[ y_i^c \log(\sigma(f_c(x_i))) + (1-y_i^c) \log(1-\sigma(f_c(x_i))) \right]$$

where $f_c$ is the binary classifier for class $c$ and $\sigma$ is the sigmoid function.

## Neural Network Architectures

### CNN (Convolutional Neural Network)

Basic architecture for time series processing:

- 3 convolutional blocks with increasing number of filters (32 → 64 → 128)
- Each block: Conv1d → BatchNorm → ReLU → MaxPool → Dropout
- Final layer: AdaptiveAvgPool → Linear

Features:
- Simplicity and training efficiency
- Good performance with local patterns
- Fewer parameters

**Mathematical Formulation**:

For 1D convolution:

$$(\text{Conv1D}(x))_t = \sum_{i=0}^{k-1} w_i \cdot x_{t+i}$$

where $k$ is the kernel size and $w$ are the weights.

### TCN (Temporal Convolutional Network)

Specialized architecture for time series:

- Dilated convolutions with exponentially growing receptive field
- Residual blocks for improved gradient flow
- Preservation of temporal dimension

Features:
- Improved perception of long-term dependencies
- Efficiency for long sequences
- Training stability thanks to residual connections

**Mathematical Formulation**:

For dilated convolution with dilation factor $d$:

$$(\text{DilatedConv1D}(x))_t = \sum_{i=0}^{k-1} w_i \cdot x_{t+d \cdot i}$$

The residual connection is defined as:

$$y = \mathcal{F}(x) + x$$

where $\mathcal{F}$ is the function to be learned.

### TCNAttention

Enhanced version of TCN with attention mechanism:

- Basic TCN structure
- Addition of Self-Attention mechanism
- Dynamic focus on important parts of the sequence

Features:
- Most flexible architecture
- Better adaptability to different domain differences
- Improved performance with heterogeneous data

**Mathematical Formulation**:

The self-attention mechanism is defined as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q$, $K$, and $V$ are query, key, and value matrices, and $d_k$ is the dimension of keys.

## Running Experiments

### Single Experiment

To run a single experiment, use:

```bash
python main.py --da_method [METHOD] --backbone [BACKBONE] --source_dataset [SRC_DATASET] --target_dataset [TRG_DATASET] --source_domain [SRC_DOMAIN] --target_domain [TRG_DOMAIN] --device [DEVICE] --batch_size [BATCH_SIZE] --num_epochs [NUM_EPOCHS] --verbose
```

Parameters:
- `METHOD`: adaptation method (RAINCOAT, UniOT, DANCE, UAN, OVANet)
- `BACKBONE`: architecture (CNN, TCN, TCNAttention)
- `SRC_DATASET`: source dataset (HHAR_SA, WISDM)
- `TRG_DATASET`: target dataset (HHAR_SA, WISDM)
- `SRC_DOMAIN`: source domain index
- `TRG_DOMAIN`: target domain index
- `DEVICE`: device (cuda, cpu)
- `BATCH_SIZE`: batch size (default 32)
- `NUM_EPOCHS`: number of epochs (default 100)

Example:
```bash
python main.py --da_method RAINCOAT --backbone TCNAttention --source_dataset HHAR_SA --target_dataset WISDM --source_domain 0 --target_domain 4 --device cuda --batch_size 32 --num_epochs 50 --verbose
```

### Batch Experiments

To run multiple experiments and aggregate results:

```bash
python results_aggregation.py
```

The script will automatically run all combinations of methods, architectures, and scenarios, save results to a CSV file, and compute aggregated metrics.

## Evaluation Methodology

The following metrics are used to evaluate domain adaptation methods:

1. **Accuracy** - classification accuracy on target domain
2. **F1-score** - F1-measure on target domain (average across classes)
3. **H-score** - harmonic mean of accuracies on source and target domains:
   
   H-score = 2 * (Acc_src * Acc_trg) / (Acc_src + Acc_trg)

Experiments are conducted on two types of scenarios:
- **W→H**: WISDM → HHAR_SA (9 different domain combinations)
- **H→W**: HHAR_SA → WISDM (9 different domain combinations)

### Feature Visualization

- t-SNE projection of features before and after adaptation
- Visualization of class prototypes
- Display of feature drift during correction process

### Optimal Transport Visualization

- Sinkhorn coupling matrices
- Transport plan visualization
- Distribution changes visualization

### Multi-modality Diagnostics

- Drift histograms for modality detection
- DIP-test results
- K-means clustering

## Hyperparameter Configuration

Each domain adaptation method has its own set of hyperparameters that can be configured in `configs/hparams.py`:

### RAINCOAT

```python
'RAINCOAT': {
    'learning_rate': 1e-3,
    'lambda_cls': 1.0,        # Classification loss weight
    'lambda_rec': 0.1,        # Reconstruction loss weight
    'lambda_sinkhorn': 0.1,   # Sinkhorn loss weight
    'lambda_contrast': 0.1,   # Contrastive loss weight
    'temperature': 0.1,       # Temperature for softmax
    'projection_dim': 128,    # Projection dimension
    'mid_channels': 64,       # Intermediate channels
    'entropy_threshold': 0.5, # Entropy threshold
    'dbscan_eps': 0.5,        # DBSCAN parameter
    'dbscan_min_samples': 5   # DBSCAN minimum samples
}
```

### UniOT

```python
'UniOT': {
    'learning_rate': 1e-4,
    'lambda_ot': 1.0,        # OT loss weight
    'epsilon': 0.1,          # Sinkhorn regularization parameter
    'num_iters': 100,        # Number of Sinkhorn iterations
    'temperature': 0.1,      # Temperature for softmax
    'projection_dim': 512,   # Projection dimension
    'mid_channels': 1536     # Intermediate channels
}
```

## Extension

The system is designed with extensibility in mind. To add new methods:

1. Create a new file in `algorithms/` with a class inheriting from `BaseAlgorithm`
2. Implement the `update`, `evaluate`, and other necessary methods
3. Register the method in `algorithms/algorithm_factory.py`
4. Add hyperparameters in `configs/hparams.py`

To support new datasets:

1. Create a new configuration class in `configs/`
2. Implement a data loader in `dataloader/data_loader.py`
3. Add preprocessing in `dataloader/data_preprocessing.py`
4. Register the dataset in the configuration factory

## Device Compatibility

The project supports both CPU and GPU (CUDA) operation. For GPU, we recommend:
- CUDA 11.0+
- cuDNN 8.0+
- GPU with at least 8 GB memory for optimal performance

## Debugging and Testing

To check correct data operation:
```bash
python check_data.py
```

To test individual components:
```bash
python -m unittest discover
```

## References

1. Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M., & Lempitsky, V. (2016). Domain-Adversarial Training of Neural Networks. Journal of Machine Learning Research, 17(59), 1-35.

2. Wang, J., Chen, Y., Hu, L., Peng, X., & Yu, P. S. (2018). Stratified Transfer Learning for Cross-domain Activity Recognition. In IEEE International Conference on Pervasive Computing and Communications (PerCom).

3. Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. In Advances in Neural Information Processing Systems (NIPS).

4. Saito, K., Kim, D., Sclaroff, S., Darrell, T., & Saenko, K. (2019). Semi-supervised Domain Adaptation via Minimax Entropy. In IEEE International Conference on Computer Vision (ICCV).

5. Tang, H., Chen, K., & Jia, K. (2020). Unsupervised Domain Adaptation via Structurally Regularized Deep Clustering. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

6. Lea, C., Flynn, M. D., Vidal, R., Reiter, A., & Hager, G. D. (2017). Temporal Convolutional Networks for Action Segmentation and Detection. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

7. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. In Advances in Neural Information Processing Systems (NIPS).

8. You, K., Long, M., Cao, Z., Wang, J., & Jordan, M. I. (2019). Universal Domain Adaptation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

9. Hartigan, J. A., & Hartigan, P. M. (1985). The dip test of unimodality. The Annals of Statistics, 13(1), 70-84.

10. Stisen, A., Blunck, H., Bhattacharya, S., Prentow, T. S., Kjærgaard, M. B., Dey, A., Sonne, T., & Jensen, M. M. (2015). Smart Devices are Different: Assessing and Mitigating Mobile Sensing Heterogeneities for Activity Recognition. In Proceedings of the 13th ACM Conference on Embedded Networked Sensor Systems (SenSys).

11. Kwapisz, J. R., Weiss, G. M., & Moore, S. A. (2011). Activity Recognition using Cell Phone Accelerometers. ACM SIGKDD Explorations Newsletter, 12(2), 74-82.

12. Flamary, R., Courty, N., Gramfort, A., Alaya, M. Z., Boisbunon, A., Chambon, S., Chapel, L., Corenflos, A., Fatras, K., Fournier, N., Gautheron, L., Gayraud, N. T. H., Janati, H., Rakotomamonjy, A., Redko, I., Rolet, A., Schutz, A., Seguy, V., Sutherland, D. J., Tavenard, R., Tong, A., & Vayer, T. (2021). POT: Python Optimal Transport. Journal of Machine Learning Research, 22(78), 1-8.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use this code in your research, please cite our work:

```bibtex
@article{raincoat2023,
  title={RAINCOAT: Robust Domain Adaptation for Human Activity Recognition with Cross-domain Optimal Transport},
  author={Author1 and Author2 and Author3},
  journal={ArXiv},
  year={2023}
}
```

## Contact

For questions and suggestions for code improvement:
- Email: ankoroff57@gmail.com
- GitHub Issues: https://github.com/yourusername/Raincoat/issues

---

© 2025 ML_ELEPHANTS Team

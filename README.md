
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

L_RAINCOAT = L_cls + λ_rec * L_rec + λ_sink * L_sink + λ_contrast * L_contrast

where:
- L_cls = -Σ_{i=1}^{n_s} Σ_{c=1}^{C} y_i^c * log(ŷ_i^c)  # Classification loss
- L_rec = (1/n_s) * Σ_{i=1}^{n_s} ||x_i - x̂_i||^2       # Reconstruction loss
- L_sink = <P, C>                                       # Sinkhorn loss (P is transport plan)
- L_contrast is the contrastive loss for semantic alignment

The DIP-test for multi-modality detection is defined as:
DIP(x) = sup_{t ∈ [0,1]} |F_n(t) - t|

where F_n is the empirical CDF of the data

### UniOT

UniOT (Unidirectional Optimal Transport) implements adaptation based on unidirectional optimal transport:

- Computes transport plan from source to target using Sinkhorn algorithm
- Uses feature normalization for stability
- Applies unbalanced optimal transport to account for different domain sizes
- Implements adaptive filling for class balance

**Mathematical Formulation**:

The UniOT optimization objective:
L_UniOT = L_cls + λ_ot * L_ot

The optimal transport loss L_ot is computed as:
L_ot = min_{P ∈ Π(μ_s, μ_t)} <P, C> - ε*H(P)

where:
- Π(μ_s, μ_t) is the set of transport plans
- C is the cost matrix
- ε is the regularization parameter
- H(P) is the entropy of the transport pla

The Sinkhorn algorithm solves this by iterative updates:

$$\mathbf{u}^{l+1} = \frac{\mathbf{a}}{\mathbf{K}\mathbf{v}^l}, \quad \mathbf{v}^{l+1} = \frac{\mathbf{b}}{\mathbf{K}^T\mathbf{u}^{l+1}}$$

where $$\mathbf{K} = e^{-\mathbf{C}/\epsilon}$$.

### DANCE

DANCE (Domain Adaptation with Neighborhood Consistency and Entropy minimization):

- Ensures neighborhood consistency between domains
- Minimizes entropy of predictions on target domain
- Uses margin loss for improved class separation
- Includes mechanisms to protect against negative transfer

**Mathematical Formulation**:

The DANCE optimization objective:
L_DANCE = L_cls + λ_nc * L_nc + λ_ent * L_ent

The neighborhood consistency loss L_nc is:
L_nc = (1/n_t) * Σ_{i=1}^{n_t} Σ_{j ∈ N_k(i)} ||ŷ_i - ŷ_j||^2 * w_ij

The entropy minimization loss L_ent is:
L_ent = -(1/n_t) * Σ_{i=1}^{n_t} Σ_{c=1}^{C} ŷ_i^c * log(ŷ_i^c)

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
- **W→W**: WISDM → WISDM (9 different domain combinations)

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

## Contact

For questions and suggestions for code improvement:
- Email: ankoroff57@gmail.com
- GitHub Issues: https://github.com/yourusername/Raincoat/issues

---

© 2025 ML_ELEPHANTS Team

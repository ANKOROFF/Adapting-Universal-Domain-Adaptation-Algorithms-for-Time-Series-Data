"""
Hyperparameter configurations for different datasets and algorithms.
Note: The current hyperparameter values are not necessarily optimal for specific scenarios.
"""

def get_hparams_class(dataset_name):
    """
    Factory function to retrieve the hyperparameter configuration class for a specific dataset.
    
    Args:
        dataset_name (str): Name of the dataset to configure
        
    Returns:
        class: The corresponding hyperparameter configuration class
        
    Raises:
        NotImplementedError: If the specified dataset is not supported
    """
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class HAR():
    """
    Hyperparameter configuration for the Human Activity Recognition (HAR) dataset.
    Contains training parameters and algorithm-specific hyperparameters for various domain adaptation methods.
    """
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            'num_epochs': 3,
            'batch_size': 32,
            'weight_decay': 1e-4,
        }
        self.alg_hparams = {
            'RAINCOAT': {'learning_rate': 5e-4, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5},
            'DANN': {'learning_rate': 1e-2, 'src_cls_loss_wt': 9.74, 'domain_loss_wt': 5.43},
            'Deep_Coral': {'learning_rate': 5e-3, 'src_cls_loss_wt': 8.67, 'coral_wt': 0.44},
            'DDC': {'learning_rate': 5e-3, 'src_cls_loss_wt': 6.24, 'domain_loss_wt': 6.36},
            'HoMM': {'learning_rate': 1e-3, 'src_cls_loss_wt': 2.15, 'domain_loss_wt': 9.13},
            'CoDATS': {'learning_rate': 1e-3, 'src_cls_loss_wt': 6.21, 'domain_loss_wt': 1.72},
            'DSAN': {'learning_rate': 5e-4, 'src_cls_loss_wt': 1.76, 'domain_loss_wt': 1.59},
            'AdvSKM': {'learning_rate': 5e-3, 'src_cls_loss_wt': 3.05, 'domain_loss_wt': 2.876},
            'MMDA': {'learning_rate': 1e-3, 'src_cls_loss_wt': 6.13, 'mmd_wt': 2.37, 'coral_wt': 8.63, 'cond_ent_wt': 7.16},
            'CDAN': {'learning_rate': 1e-2, 'src_cls_loss_wt': 5.19, 'domain_loss_wt': 2.91, 'cond_ent_wt': 1.73},
            "AdaMatch": {'learning_rate': 3e-3, 'tau': 0.9, 'max_segments': 5, 'jitter_scale_ratio': 0.01, 'jitter_ratio': 0.2},
            'DIRT': {'learning_rate': 5e-4, 'src_cls_loss_wt': 7.00, 'domain_loss_wt': 4.51, 'cond_ent_wt': 0.79, 'vat_loss_wt': 9.31}
        }


class EEG():
    """
    Hyperparameter configuration for the EEG dataset.
    Contains training parameters and algorithm-specific hyperparameters for various domain adaptation methods.
    """
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
            'num_epochs': 3,
            'batch_size': 128,
            'weight_decay': 1e-4,
        }
        self.alg_hparams = {
            'RAINCOAT': {'learning_rate': 0.002, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5},
            "AdaMatch": {'learning_rate': 0.0005, 'tau': 0.9, 'max_segments': 5, 'jitter_scale_ratio': 0.01, 'jitter_ratio': 0.2},
            'DANN': {'learning_rate': 0.0005, 'src_cls_loss_wt': 8.30, 'domain_loss_wt': 0.324},
            'Deep_Coral': {'learning_rate': 0.0005, 'src_cls_loss_wt': 9.39, 'coral_wt': 0.19},
            'DDC': {'learning_rate': 0.0005, 'src_cls_loss_wt': 2.951, 'domain_loss_wt': 8.923},
            'HoMM': {'learning_rate': 0.0005, 'src_cls_loss_wt': 0.197, 'domain_loss_wt': 1.102},
            'CoDATS': {'learning_rate': 0.01, 'src_cls_loss_wt': 9.239, 'domain_loss_wt': 1.342},
            'DSAN': {'learning_rate': 0.001, 'src_cls_loss_wt': 6.713, 'domain_loss_wt': 6.708},
            'AdvSKM': {'learning_rate': 0.0005, 'src_cls_loss_wt': 2.50, 'domain_loss_wt': 2.50},
            'MMDA': {'learning_rate': 0.0005, 'src_cls_loss_wt': 4.48, 'mmd_wt': 5.951, 'coral_wt': 3.36, 'cond_ent_wt': 6.13},
            'CDAN': {'learning_rate': 0.001, 'src_cls_loss_wt': 6.803, 'domain_loss_wt': 4.726, 'cond_ent_wt': 1.307},
            'DIRT': {'learning_rate': 0.005, 'src_cls_loss_wt': 9.183, 'domain_loss_wt': 7.411, 'cond_ent_wt': 2.564, 'vat_loss_wt': 3.583},
        }


class WISDM():
    """
    Hyperparameter configuration for the WISDM dataset.
    Contains training parameters and algorithm-specific hyperparameters for various domain adaptation methods.
    """
    def __init__(self):
        super(WISDM, self).__init__()
        self.train_params = {
            'num_epochs': 3,
            'batch_size': 16,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'momentum': 0.9,
            'num_workers': 2,
            'shuffle': True,
            'pin_memory': True,
            'drop_last': True
        }
        self.alg_hparams = {
            'DANCE': {
                'learning_rate': 1e-3,
                'src_cls_loss_wt': 0.5,
                'domain_loss_wt': 0.5,
                'weight_decay': 1e-4,
                'temperature': 0.1,
                'projection_dim': 64,
                'mid_channels': 32,
                'num_classes': 6
            },
            'OVANet': {
                'learning_rate': 1e-3,
                'src_cls_loss_wt': 0.5,
                'domain_loss_wt': 0.5,
                'weight_decay': 1e-4,
                'temperature': 0.1,
                'projection_dim': 64,
                'mid_channels': 32,
                'num_classes': 6
            },
            'UniOT': {
                'learning_rate': 1e-3,
                'src_cls_loss_wt': 0.5,
                'domain_loss_wt': 0.5,
                'weight_decay': 1e-4,
                'temperature': 0.1,
                'projection_dim': 64,
                'mid_channels': 32,
                'num_classes': 6
            },
            'RAINCOAT': {
                'learning_rate': 1e-3,
                'lambda_cls': 1.0,
                'lambda_rec': 0.1,
                'lambda_sinkhorn': 0.1,
                'lambda_contrast': 0.1,
                'weight_decay': 1e-4,
                'temperature': 0.1,
                'projection_dim': 128,
                'mid_channels': 64,
                'num_classes': 6,
                'entropy_threshold': 0.5,
                'dbscan_eps': 0.5,
                'dbscan_min_samples': 5,
                'dropout': 0.2,
                'fourier_modes': 32,
                'sequence_len': 200,
                'features_len': 64,
                'kernel_size': 3,
                'stride': 1,
                'input_channels': 3,
                'final_out_channels': 64
            },
            'DANN': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1.613, 'domain_loss_wt': 1.857},
            'Deep_Coral': {'learning_rate': 0.005, 'src_cls_loss_wt': 8.876, 'coral_wt': 5.56},
            'DDC': {'learning_rate': 1e-3, 'src_cls_loss_wt': 7.01, 'domain_loss_wt': 7.595},
            'HoMM': {'learning_rate': 1e-3, 'src_cls_loss_wt': 0.1913, 'domain_loss_wt': 4.239},
            'CoDATS': {'learning_rate': 1e-3, 'src_cls_loss_wt': 7.187, 'domain_loss_wt': 6.439},
            'DSAN': {'learning_rate': 1e-3, 'src_cls_loss_wt': 0.1, 'domain_loss_wt': 0.1},
            'AdvSKM': {'learning_rate': 3e-4, 'src_cls_loss_wt': 3.05, 'domain_loss_wt': 2.876},
            'MMDA': {'learning_rate': 1e-3, 'src_cls_loss_wt': 0.1, 'mmd_wt': 0.1, 'coral_wt': 0.1, 'cond_ent_wt': 0.4753},
            "AdaMatch": {'learning_rate': 3e-3, 'tau': 0.9, 'max_segments': 5, 'jitter_scale_ratio': 0.01, 'jitter_ratio': 0.2},
            'CDAN': {'learning_rate': 1e-3, 'src_cls_loss_wt': 9.54, 'domain_loss_wt': 3.283, 'cond_ent_wt': 0.1},
            'DIRT': {'learning_rate': 1e-3, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5, 'cond_ent_wt': 0.1, 'vat_loss_wt': 0.1},
            "DANCE": {'learning_rate': 1e-3, 'momentum': 0.5, 'eta': 0.05, "thr": 1.71, "margin": 0.5, 'temp': 0.05},
            'UAN': {'learning_rate': 2e-2, 'src_cls_loss_wt': 1.0, 'domain_loss_wt': 2.0, 'entropy_threshold': 0.3, 'sgd_momentum': 0.9},
            'UniOT': {
                'learning_rate': 1e-4,
                'lambda_ot': 1.0,
                'epsilon': 0.1,
                'num_iters': 100
            },
        }


class HHAR_SA():
    """
    Hyperparameter configuration for the HHAR_SA dataset.
    Contains training parameters and algorithm-specific hyperparameters for various domain adaptation methods.
    """
    def __init__(self):
        super(HHAR_SA, self).__init__()
        self.train_params = {
            'num_epochs': 3,
            'batch_size': 128,
            'lr': 1e-2,
            'weight_decay': 1e-4,
            'early_stopping_patience': 10,
        }
        self.alg_hparams = {
            'RAINCOAT': {
                'learning_rate': 1e-3,
                'lambda_cls': 1.0,
                'lambda_rec': 0.1,
                'lambda_sinkhorn': 0.1,
                'lambda_contrast': 0.1,
                'weight_decay': 1e-4,
                'temperature': 0.1,
                'projection_dim': 128,
                'mid_channels': 64,
                'num_classes': 6,
                'entropy_threshold': 0.5,
                'dbscan_eps': 0.5,
                'dbscan_min_samples': 5,
                'dropout': 0.2,
                'fourier_modes': 32,
                'sequence_len': 200,
                'features_len': 64,
                'kernel_size': 3,
                'stride': 1,
                'input_channels': 3,
                'final_out_channels': 64
            },
            'DANCE': {
                'learning_rate': 1e-3,
                'momentum': 0.5,
                'eta': 0.05,
                'thr': 1.71,
                'margin': 0.5,
                'temp': 0.05,
            },
            'UAN': {
                'learning_rate': 2e-2,
                'src_cls_loss_wt': 1.0,
                'domain_loss_wt': 2.0,
                'entropy_threshold': 0.3,
                'sgd_momentum': 0.9
            },
            'UniOT': {
                'learning_rate': 1e-4,
                'lambda_ot': 1.0,
                'epsilon': 0.1,
                'num_iters': 100,
                'temperature': 0.1,
                'projection_dim': 512,
                'mid_channels': 1536
            },
            'OVANet': {
                'learning_rate': 1e-3,
                'src_cls_loss_wt': 0.5,
                'domain_loss_wt': 0.5,
                'weight_decay': 1e-4,
                'temperature': 0.1,
                'projection_dim': 64,
                'mid_channels': 32,
                'num_classes': 6
            }
        }


class Boiler():
    """
    Hyperparameter configuration for the Boiler dataset.
    Contains training parameters and algorithm-specific hyperparameters for various domain adaptation methods.
    """
    def __init__(self):
        super(Boiler, self).__init__()
        self.train_params = {
            'num_epochs': 3,
            'batch_size': 32,
            'weight_decay': 1e-4,
        }
        self.alg_hparams = {
            'RAINCOAT': {'learning_rate': 0.0005, 'src_cls_loss_wt': 0.9603, 'domain_loss_wt': 0.9238},
            'DANN': {'learning_rate': 0.0005, 'src_cls_loss_wt': 0.9603, 'domain_loss_wt': 0.9238},
            'Deep_Coral': {'learning_rate': 0.0005, 'src_cls_loss_wt': 0.05931, 'coral_wt': 8.452},
            'DDC': {'learning_rate': 0.01, 'src_cls_loss_wt': 0.1593, 'domain_loss_wt': 0.2048},
            'HoMM': {'learning_rate': 0.001, 'src_cls_loss_wt': 0.2429, 'domain_loss_wt': 0.9824},
            'CoDATS': {'learning_rate': 0.0005, 'src_cls_loss_wt': 0.5416, 'domain_loss_wt': 0.5582},
            'DSAN': {'learning_rate': 0.005, 'src_cls_loss_wt': 0.4133, 'domain_loss_wt': 0.16},
            'AdvSKM': {'learning_rate': 0.001, 'src_cls_loss_wt': 0.4637, 'domain_loss_wt': 0.1511},
            'MMDA': {'learning_rate': 0.001, 'src_cls_loss_wt': 0.9505, 'mmd_wt': 0.5476, 'cond_ent_wt': 0.5167, 'coral_wt': 0.5838},
            'CDAN': {'learning_rate': 0.001, 'src_cls_loss_wt': 0.6636, 'domain_loss_wt': 0.1954, 'cond_ent_wt': 0.0124},
            'DIRT': {'learning_rate': 0.001, 'src_cls_loss_wt': 0.9752, 'domain_loss_wt': 0.3892, 'cond_ent_wt': 0.09228, 'vat_loss_wt': 0.1947}
        }

# Default training parameters for general use
train_params = {
    'num_epochs': 100,
    'batch_size': 8,
    'num_workers': 0,
    'learning_rate': 0.001,
}

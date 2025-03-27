"""
Configuration class for the WISDM dataset.
Contains model architecture parameters, training configurations, and domain adaptation settings.
"""

class WISDM_Configs:
    """
    Configuration class for the WISDM dataset, containing all necessary parameters
    for data processing, model architecture, training, and domain adaptation.
    """
    def __init__(self):
        # Dataset identification
        self.dataset_name = "WISDM"
        
        # Model architecture parameters
        self.input_channels = 3  # Number of accelerometer axes
        self.sequence_len = 200  # Sequence length for temporal data
        self.num_classes = 6  # Number of activity classes
        self.fourier_modes = 32  # Number of Fourier modes for spectral analysis
        self.features_len = 64  # Feature vector dimension
        self.final_out_channels = 64  # Output channels for feature extraction
        self.mid_channels = 64  # Intermediate channel dimensions
        self.projection_dim = 128  # Projection space dimension
        
        # Data processing parameters
        self.sampling_rate = 20  # WISDM dataset sampling rate (Hz)
        self.need_resampling = False  # Whether resampling is required
        self.kernel_size = 3  # Convolution kernel size
        self.stride = 1  # Convolution stride
        self.dropout = 0.2  # Dropout probability
        
        # Training parameters
        self.learning_rate = 0.001  # Initial learning rate
        self.weight_decay = 0.0005  # L2 regularization factor
        self.batch_size = 16  # Batch size for training
        self.num_epochs = 100  # Maximum number of training epochs
        self.early_stopping_patience = 10  # Patience for early stopping
        self.min_delta = 0.001  # Minimum improvement threshold
        
        # RAINCOAT algorithm parameters
        self.lambda_cls = 1.0  # Classification loss weight
        self.lambda_rec = 0.1  # Reconstruction loss weight
        self.lambda_sinkhorn = 0.1  # Sinkhorn distance weight
        self.lambda_contrast = 0.1  # Contrastive loss weight
        self.temperature = 0.1  # Temperature parameter for contrastive learning
        self.entropy_threshold = 0.5  # Entropy threshold for bimodal testing
        self.dbscan_eps = 0.5  # DBSCAN epsilon parameter
        self.dbscan_min_samples = 5  # Minimum samples for DBSCAN clustering
        
        # Cross-domain adaptation scenarios
        self.scenarios = [
            ('1', '2'),
            ('1', '3'),
            ('2', '1'),
            ('2', '3'),
            ('3', '1'),
            ('3', '2')
        ] 
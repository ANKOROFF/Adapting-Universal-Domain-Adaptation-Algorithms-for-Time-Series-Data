"""
Configuration class for the HHAR dataset using Samsung device.
Contains model architecture parameters, training configurations, and domain adaptation settings.
"""

class HHAR_SA_Configs:
    """
    Configuration class for the HHAR dataset with Samsung device, containing all necessary parameters
    for data processing, model architecture, training, and domain adaptation.
    """
    def __init__(self):
        # Dataset identification
        self.dataset_name = "HHAR_SA"
        
        # Model architecture parameters
        self.input_channels = 3  # Accelerometer channels (x, y, z)
        self.sequence_len = 64  # Sequence length after resampling
        self.num_classes = 6  # Number of activity classes
        self.final_out_channels = 64  # Output channels for feature extraction
        
        # Data processing parameters
        self.use_accelerometer_only = True  # Use only accelerometer data
        self.need_resampling = True  # Whether to perform resampling
        self.sampling_rate = 200  # Original sampling rate (Hz)
        self.target_sampling_rate = 50  # Target sampling rate (matching WISDM)
        
        # Training parameters
        self.learning_rate = 0.001  # Initial learning rate
        self.weight_decay = 0.0005  # L2 regularization factor
        self.batch_size = 64  # Batch size as specified in the paper
        self.num_epochs = 50  # Number of epochs as specified in the paper
        self.early_stopping_patience = 10  # Patience for early stopping
        self.min_delta = 0.001  # Minimum improvement threshold
        
        # Domain adaptation parameters
        self.entropy_threshold = 1.0  # Threshold for UAN algorithm
        self.epsilon = 0.1  # Regularization parameter for UniOT
        self.num_iterations = 10  # Number of iterations for UniOT Sinkhorn-Knopp algorithm
        
        # Cross-domain adaptation scenarios
        self.scenarios = [
            ('0', '1'),
            ('0', '2'),
            ('1', '0'),
            ('1', '2'),
            ('2', '0'),
            ('2', '1')
        ] 
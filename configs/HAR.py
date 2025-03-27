"""
Configuration class for the Human Activity Recognition (HAR) dataset.
Contains model architecture parameters, training configurations, and domain adaptation settings.
"""

class HAR_Configs:
    """
    Configuration class for the Human Activity Recognition (HAR) dataset, containing all necessary parameters
    for data processing, model architecture, training, and domain adaptation.
    """
    def __init__(self):
        # Dataset identification
        self.dataset_name = "HAR"
        
        # Model architecture parameters
        self.input_channels = 9  # Number of sensor channels (accelerometer, gyroscope)
        self.sequence_length = 128  # Length of each time series segment
        self.num_classes = 6  # Number of activity classes
        self.final_out_channels = 64  # Output channels for feature extraction
        
        # Training parameters
        self.learning_rate = 0.001  # Initial learning rate
        self.weight_decay = 0.0005  # L2 regularization factor
        self.batch_size = 32  # Batch size for training
        self.num_epochs = 100  # Maximum number of training epochs
        self.early_stopping_patience = 10  # Patience for early stopping
        self.min_delta = 0.001  # Minimum improvement threshold
        
        # Domain adaptation parameters
        self.entropy_threshold = 1.0  # Threshold for UAN algorithm
        self.epsilon = 0.1  # Regularization parameter for UniOT
        self.num_iterations = 10  # Number of iterations for UniOT Sinkhorn-Knopp algorithm
        
        # Cross-domain adaptation scenarios
        self.scenarios = [
            ('1', '2'),
            ('1', '3'),
            ('2', '1'),
            ('2', '3'),
            ('3', '1'),
            ('3', '2')
        ] 
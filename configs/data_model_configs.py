"""
Configuration classes for different datasets and their corresponding model architectures.
Each class contains data loading parameters and model-specific configurations.
"""

def get_dataset_class(dataset_name):
    """
    Factory function to retrieve the dataset configuration class for a specific dataset.
    
    Args:
        dataset_name (str): Name of the dataset to configure
        
    Returns:
        class: The corresponding dataset configuration class
        
    Raises:
        NotImplementedError: If the specified dataset is not supported
    """
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class HAR():
    """
    Configuration class for the Human Activity Recognition (HAR) dataset.
    Contains data loading parameters and model architecture configurations.
    """
    def __init__(self):
        super(HAR, self)
        # Data loading parameters
        self.scenarios = [("2", "11"), ("6", "23"),("7", "13"),("9", "18"),("12", "16"),\
            ("13", "19"),  ("18", "21"), ("20", "6"),("23", "13"),("24", "12")]
        # self.scenarios = [("24", "12")]
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6
        self.fourier_modes = 64
        self.out_dim = 256
        # CNN and RESNET features
        
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128


class EEG():
    """
    Configuration class for the EEG dataset.
    Contains data loading parameters and model architecture configurations.
    """
    def __init__(self):
        super(EEG, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        self.scenarios = [("0", "11"), ("2", "5"), ("12", "5"), ("7", "18"), ("16", "1"), ("9", "14"),\
            ("4", "12"),("10", "7"),("6", "3"),("8", "10")]
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 128
        self.features_len = 1
        self.fourier_modes = 300
        self.out_dim = 256
        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 15 # 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100


class WISDM(object):
    """
    Configuration class for the WISDM dataset.
    Contains data loading parameters and model architecture configurations.
    """
    def __init__(self):
        super(WISDM, self).__init__()
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.sequence_len = 128
        # Closed Set DA
        self.scenarios = [("2", "32"), ("4", "15"),("7", "30"),('12','7'), ('12','19'),('18','20'),\
                          ('20','30'), ("21", "31"),("25", "29"), ('26','2')]
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6
        self.width = 64  # for FNN
        self.fourier_modes = 64
        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.out_dim = 256
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75,150,300]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500




class HHAR_SA(object):  ## HHAR dataset, SAMSUNG device.
    def __init__(self):
        super(HHAR_SA, self).__init__()
        self.sequence_len = 128
        # self.scenarios = [("0", "2")]
        self.scenarios = [("0", "2"), ("1", "6"),("2", "4"),("4", "0"),("4", "5"),\
            ("5", "1"),("5", "2"),("7", "2"),("7", "5"),("8", "4")]
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.fourier_modes = 32
        
        # Веса для различных компонент функции потерь
        self.lambda_recon = 1.0
        self.lambda_sink = 0.1
        self.lambda_cls = 1.0
        self.lambda_contrast = 0.1
        
        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.lr = 0.001  # Добавляем параметр скорости обучения

        # features
        self.mid_channels = 64 * 2
        self.final_out_channels = 128
        self.hidden_channels = 64  # Добавляем атрибут hidden_channels
        self.features_len = 1
        self.out_dim = 128

        # TCN features
        self.tcn_layers = [75,150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

class Boiler(object):
    def __init__(self):
        super(Boiler, self).__init__()
        self.class_names = ['0','1']
        self.sequence_len = 6
        self.scenarios = [("1", "2"),("1", "3"),("2", "3")]
        self.num_classes = 2
        self.sequence_len = 6
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 20
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 64
        self.features_len = 1

        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 15# 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100
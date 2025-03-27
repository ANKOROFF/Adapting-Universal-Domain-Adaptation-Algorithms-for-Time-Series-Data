import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

class BaseAlgorithm(nn.Module):
    def __init__(self, backbone_class, configs, hparams, device):
        super(BaseAlgorithm, self).__init__()
        self.configs = configs
        self.hparams = hparams
        self.device = device
        
        # Initialize feature extractor with verbose flag
        self.feature_extractor = backbone_class(configs).to(device)
        self.feature_extractor.verbose = hparams.get('verbose', False)
        
        # Initialize classifier
        feature_dim = self._get_feature_dim()
        print(f"Feature dimension: {feature_dim}")
        print(f"Number of classes: {configs.num_classes}")
        
        self.classifier = nn.Linear(feature_dim, configs.num_classes).to(device)
        
    def _get_feature_dim(self):
        """Calculate the dimension of features after feature extraction"""
        x = torch.randn(1, self.configs.input_channels, self.configs.sequence_len).to(self.device)
        x = self.feature_extractor(x)
        if isinstance(x, tuple):
            x = x[0]
        print(f"Feature shape: {x.shape}")
        return x.shape[1]
        
    def forward(self, x):
        features = self.feature_extractor(x)
        if isinstance(features, tuple):
            features = features[0]
        predictions = self.classifier(features)
        return predictions
        
    def update(self, src_x, src_y, trg_x):
        """
        Update the model parameters
        Args:
            src_x: source domain input data
            src_y: source domain labels
            trg_x: target domain input data
        Returns:
            dict of losses
        """
        raise NotImplementedError
        
    def get_parameters(self):
        """Returns parameters that will be updated during training"""
        return [
            {"params": self.feature_extractor.parameters()},
            {"params": self.classifier.parameters()}
        ] 

    def evaluate(self, src_pred_labels, src_true_labels, trg_pred_labels, trg_true_labels):
        """
        Evaluate the model on source and target domains
        Args:
            src_pred_labels: predicted labels for source domain
            src_true_labels: true labels for source domain
            trg_pred_labels: predicted labels for target domain
            trg_true_labels: true labels for target domain
        Returns:
            tuple of (target accuracy, target F1 score, H-score)
        """
        # Calculate accuracies
        src_acc = accuracy_score(src_true_labels, src_pred_labels)
        trg_acc = accuracy_score(trg_true_labels, trg_pred_labels)
        
        # Calculate F1 score for target domain
        trg_f1 = f1_score(trg_true_labels, trg_pred_labels, average='weighted')
        
        # Calculate H-score
        h_score = 2 * (src_acc * trg_acc) / (src_acc + trg_acc) * 100
        
        return trg_acc, trg_f1, h_score 
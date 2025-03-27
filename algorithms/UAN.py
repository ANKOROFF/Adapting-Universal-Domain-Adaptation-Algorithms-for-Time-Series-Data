"""
UAN algorithm implementation with adaptive feature projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from algorithms.base import BaseAlgorithm
from algorithms.utils import entropy, calc_coeff

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class UAN(BaseAlgorithm):
    def __init__(self, backbone_class, configs, hparams, device):
        # Call parent class initialization first, but skip classifier initialization
        super(BaseAlgorithm, self).__init__()
        self.configs = configs
        self.hparams = hparams
        self.device = device
        
        # Initialize feature extractor with verbose flag
        self.feature_extractor = backbone_class(configs).to(device)
        self.feature_extractor.verbose = hparams.get('verbose', False)
        
        # Get feature dimension for source dataset
        self.source_dataset = hparams.get('source_dataset', 'WISDM')
        self.target_dataset = hparams.get('target_dataset', 'WISDM')
        
        # Define dataset-specific sequence lengths
        self.dataset_seq_lengths = {
            'WISDM': 200,
            'HHAR_SA': 64
        }
        
        # Get feature dimensions for both datasets
        self.source_feature_dim = self._get_feature_dim(self.source_dataset)
        self.target_feature_dim = self._get_feature_dim(self.target_dataset)
        
        print(f"Source feature dimension ({self.source_dataset}): {self.source_feature_dim}")
        print(f"Target feature dimension ({self.target_dataset}): {self.target_feature_dim}")
        
        # Add projection layers with intermediate dimensions
        projection_dim = 512
        source_intermediate_dim = min(self.source_feature_dim, 1024)
        target_intermediate_dim = min(self.target_feature_dim, 1024)
        
        print(f"Source intermediate dimension: {source_intermediate_dim}")
        print(f"Target intermediate dimension: {target_intermediate_dim}")
        print(f"Final projection dimension: {projection_dim}")
        
        # Create separate projection layers for source and target
        self.source_projection = nn.Sequential(
            nn.Linear(self.source_feature_dim, source_intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(source_intermediate_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        ).to(device)
        
        self.target_projection = nn.Sequential(
            nn.Linear(self.target_feature_dim, target_intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(target_intermediate_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        ).to(device)
        
        # Initialize classifier with projected features
        print(f"Classifier input dimension: {projection_dim}")
        print(f"Classifier output dimension: {configs.num_classes}")
        
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(projection_dim // 2, configs.num_classes)
        ).to(device)
        
        # Non-adversarial domain discriminator D'
        print(f"Domain discriminator input dimension: {projection_dim}")
        
        self.domain_discriminator_nonadv = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(projection_dim, 1),
            nn.Sigmoid()
        ).to(device)
        
        # Adversarial domain discriminator D
        self.domain_discriminator_adv = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(projection_dim, 1),
            nn.Sigmoid()
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.get_parameters(),
            lr=hparams.get('learning_rate', 0.001),
            weight_decay=hparams.get('weight_decay', 0.0005)
        )
        
        self.lambda_ = hparams.get('lambda', 0.1)
        self.w0 = hparams.get('w0', 0.5)
        self.temperature = hparams.get('temperature', 10.0)
        
    def _get_feature_dim(self, dataset_name):
        """Get feature dimension for a specific dataset."""
        seq_len = self.dataset_seq_lengths.get(dataset_name, 200)
        x = torch.randn(1, self.configs.input_channels, seq_len).to(self.device)
        x = self.feature_extractor(x)
        if isinstance(x, tuple):
            x = x[0]
        if len(x.shape) == 3:
            x = x.reshape(x.size(0), -1)
        return x.shape[1]
        
    def get_parameters(self):
        return [
            {"params": self.feature_extractor.parameters()},
            {"params": self.source_projection.parameters()},
            {"params": self.target_projection.parameters()},
            {"params": self.classifier.parameters()},
            {"params": self.domain_discriminator_nonadv.parameters()},
            {"params": self.domain_discriminator_adv.parameters()}
        ]
        
    def compute_transferability(self, features, preds):
        probs = F.softmax(preds / self.temperature, dim=1)
        ent = entropy(probs) / torch.log(torch.tensor(self.configs.num_classes, dtype=torch.float32, device=self.device))
        domain_sim = self.domain_discriminator_nonadv(features).squeeze()
        
        w_s = ent - domain_sim
        w_t = domain_sim - ent
        
        w_s = (w_s - w_s.min()) / (w_s.max() - w_s.min() + 1e-10)
        w_t = (w_t - w_t.min()) / (w_t.max() - w_t.min() + 1e-10)
        
        return w_s, w_t
        
    def preprocess_input(self, x, is_source=True):
        """Preprocess input sequences to have the same length."""
        dataset = self.source_dataset if is_source else self.target_dataset
        target_length = self.dataset_seq_lengths.get(dataset, 200)
        batch_size, channels, seq_len = x.shape
        
        if seq_len < target_length:
            # Если последовательность короче целевой, дополняем нулями
            padding = target_length - seq_len
            x = F.pad(x, (0, padding), "constant", 0)
        elif seq_len > target_length:
            # Если последовательность длиннее целевой, обрезаем
            x = x[:, :, :target_length]
        
        return x

    def encode_features(self, x, is_source=True):
        """Extract and encode features from input data."""
        if self.hparams.get('verbose', False):
            print(f"\nInput shape: {x.shape}")
        
        # Предварительная обработка входных данных
        x = self.preprocess_input(x, is_source)
        if self.hparams.get('verbose', False):
            print(f"After preprocess: {x.shape}")
        
        # Извлечение признаков
        features = self.feature_extractor(x)
        if self.hparams.get('verbose', False):
            print(f"After feature_extractor: {features.shape}")
        
        # Решейп признаков в 2D тензор
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        if self.hparams.get('verbose', False):
            print(f"After reshape: {features.shape}")
        
        # Нормализация признаков
        features = F.normalize(features, p=2, dim=1)
        if self.hparams.get('verbose', False):
            print(f"After normalize: {features.shape}")
        
        # Проекция признаков с использованием соответствующего проектора
        projection = self.source_projection if is_source else self.target_projection
        features = projection(features)
        if self.hparams.get('verbose', False):
            print(f"After projection: {features.shape}")
        
        return features
        
    def update(self, src_x, src_y, trg_x, trg_indices=None, step=None, epoch=None, total_steps=None):
        self.train()
        self.optimizer.zero_grad()
        
        if self.hparams.get('verbose', False):
            print("\nProcessing source features:")
        src_features = self.encode_features(src_x, is_source=True)
        
        if self.hparams.get('verbose', False):
            print("\nProcessing target features:")
        trg_features = self.encode_features(trg_x, is_source=False)
        
        # Source classification loss
        src_preds = self.classifier(src_features)
        cls_loss = F.cross_entropy(src_preds, src_y)
        
        # Non-adversarial domain discrimination
        src_domain_preds_nonadv = self.domain_discriminator_nonadv(src_features)
        trg_domain_preds_nonadv = self.domain_discriminator_nonadv(trg_features)
        d_loss_nonadv = -torch.mean(torch.log(src_domain_preds_nonadv + 1e-10)) - \
                        torch.mean(torch.log(1 - trg_domain_preds_nonadv + 1e-10))
        
        # Compute transferability weights
        trg_preds = self.classifier(trg_features)
        w_s, w_t = self.compute_transferability(torch.cat([src_features, trg_features]),
                                              torch.cat([src_preds, trg_preds]))
        w_s = w_s[:src_x.size(0)]
        w_t = w_t[src_x.size(0):]
        
        # Adversarial domain discrimination with gradient reversal
        src_features_adv = GradientReversalLayer.apply(src_features, self.lambda_)
        trg_features_adv = GradientReversalLayer.apply(trg_features, self.lambda_)
        
        src_domain_preds_adv = self.domain_discriminator_adv(src_features_adv)
        trg_domain_preds_adv = self.domain_discriminator_adv(trg_features_adv)
        
        d_loss_adv = -torch.mean(w_s * torch.log(src_domain_preds_adv + 1e-10)) - \
                     torch.mean(w_t * torch.log(1 - trg_domain_preds_adv + 1e-10))
        
        # Total loss with adaptive weighting
        if step is not None and total_steps is not None:
            coeff = calc_coeff(step, max_iter=total_steps)
        else:
            coeff = 1.0
            
        total_loss = cls_loss + coeff * (d_loss_nonadv + d_loss_adv)
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'cls_loss': cls_loss.item(),
            'd_loss_nonadv': d_loss_nonadv.item(),
            'd_loss_adv': d_loss_adv.item()
        }
        
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            # При предсказании используем проектор целевого датасета
            features = self.encode_features(x, is_source=False)
            preds = self.classifier(features)
        return preds 
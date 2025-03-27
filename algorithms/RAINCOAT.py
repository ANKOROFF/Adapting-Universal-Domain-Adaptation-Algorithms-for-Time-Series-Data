import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.loss import SinkhornDistance
from pytorch_metric_learning import losses
from models.models import ResClassifier_MME, classifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import DBSCAN
import logging
from algorithms.base import BaseAlgorithm
from torch.nn import init
from torch.optim import Adam
from torch.nn import Parameter
import math
from models.backbones import TCN, CNN

def weights_init(m):
    """
    Initialize network weights using different initialization strategies
    based on the layer type.
    """
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, SpectralConv1d):
        scale = 1.0 / (m.in_channels * m.out_channels)
        nn.init.uniform_(m.weights1.data, -scale, scale)
        nn.init.uniform_(m.weights2.data, -scale, scale)

class Algorithm(torch.nn.Module):
    """
    Base class for domain adaptation algorithms.
    Subclasses must implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def update(self, *args, **kwargs):
        raise NotImplementedError
        
    def encode_features(self, x):
        """
        Encode input features using the feature extractor.
        Handles different backbone architectures and feature dimensions.
        
        Args:
            x: Input tensor
            
        Returns:
            Encoded features with unified dimensionality
        """
        print("\nFeature encoding:")
        print(f"Input shape: {x.shape}")
        
        # Extract features using the feature extractor
        features = self.feature_extractor(x)
        print(f"Features after feature_extractor: {features.shape}")
        
        # Process features based on backbone type
        if isinstance(self.feature_extractor, TCN):
            print("Processing TCN backbone")
            features = features.reshape(features.shape[0], -1)
            print(f"2D TCN features: {features.shape}")
        else:
            print("Processing CNN backbone")
            if isinstance(features, tuple):
                features = features[0]  # Take only features, ignore None
            features = features.reshape(features.shape[0], -1)
            print(f"2D CNN features: {features.shape}")
        
        # Check and adjust feature dimensions if needed
        if features.shape[1] != self.unified_feature_dim:
            print(f"Dimension mismatch with unified_feature_dim ({features.shape[1]} != {self.unified_feature_dim})")
            if not hasattr(self, 'dimension_adjust') or self.dimension_adjust is None:
                print(f"Creating dimension_adjust: {features.shape[1]} -> {self.unified_feature_dim}")
                self.dimension_adjust = nn.Sequential(
                    nn.Linear(features.shape[1], self.intermediate_dim),
                    nn.BatchNorm1d(self.intermediate_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(self.intermediate_dim, self.unified_feature_dim),
                    nn.BatchNorm1d(self.unified_feature_dim)
                ).to(self.device)
            features = self.dimension_adjust(features)
            print(f"After dimension_adjust: {features.shape}")
        else:
            print("Dimensions match unified_feature_dim")
            
        return features

class SpectralConv1d(nn.Module):
    """
    Spectral Convolution layer for 1D signals.
    Implements Fourier-based convolution operations.
    """
    def __init__(self, in_channels, out_channels, modes1, fl=64):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.fl = fl
        
        # Initialize weights with shape [in_channels, out_channels, modes1]
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        
        # Normalization layers for out_channels
        self.bn_r = nn.BatchNorm1d(out_channels)
        self.bn_p = nn.BatchNorm1d(out_channels)
        
        # Projection layers with out_channels channels
        self.projection_r = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        self.projection_p = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.Tanh()
        )
        
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        
    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Apply Hann window for reducing spectral leakage
        window = torch.hann_window(x.size(-1), device=x.device)
        x = x * window
        
        # Compute FFT
        x_ft = torch.fft.rfft(x, norm='ortho')
        
        # Apply two sets of weights for better frequency feature representation
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        out_ft[:, :, :self.modes1] += self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights2)
        
        # Extract amplitude and phase
        r = out_ft[:, :, :self.modes1].abs()
        p = out_ft[:, :, :self.modes1].angle()
        
        # Normalize and apply non-linearities
        r = self.bn_r(r)
        p = self.bn_p(p)
        
        # Project to desired dimensionality
        r = self.projection_r(r)
        p = self.projection_p(p)
        
        # Combine features through addition instead of concatenation
        freq_features = r + p  # [batch, out_channels, modes1]
        
        return freq_features, out_ft


class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()
        self.width = configs.input_channels
        self.channel = configs.input_channels
        self.fl = configs.input_channels
        
        # Improved input layer
        self.fc0 = nn.Sequential(
            nn.Linear(configs.input_channels, configs.mid_channels),
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        )
        
        # Improved first convolution block
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.Conv1d(configs.mid_channels, configs.mid_channels, kernel_size=3,
                      stride=1, bias=False, padding=1),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        # Improved second convolution block
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels, kernel_size=8, 
                      stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.Conv1d(configs.mid_channels, configs.mid_channels, kernel_size=5,
                      stride=1, bias=False, padding=2),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        # Improved third convolution block
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.final_out_channels, kernel_size=8, 
                      stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.Conv1d(configs.final_out_channels, configs.final_out_channels, kernel_size=5,
                      stride=1, bias=False, padding=2),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )
        
        # Adaptive pooling with fixed output dimension
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)
        
    def forward(self, x):
        # x: [B, C, L]
        batch_size = x.size(0)
        
        # Apply input layer
        x = x.transpose(1, 2)  # [B, L, C]
        x = self.fc0(x)  # [B, L, mid_channels]
        x = x.transpose(1, 2)  # [B, mid_channels, L]
        
        # Apply convolution blocks
        x = self.conv_block1(x)  # [B, mid_channels, L//2]
        x = self.conv_block2(x)  # [B, mid_channels, L//4]
        x = self.conv_block3(x)  # [B, final_out_channels, L//8]
        
        # Apply adaptive pooling
        x = self.adaptive_pool(x)  # [B, final_out_channels, features_len]
        
        # Return tuple for compatibility with tf_encoder
        return x, None  # [B, final_out_channels, features_len], None

class tf_encoder(nn.Module):
    def __init__(self, configs):
        super(tf_encoder, self).__init__()
        
        # Initialize parameters
        self.input_channels = configs.input_channels
        self.mid_channels = configs.mid_channels
        self.sequence_len = configs.sequence_len
        self.fourier_modes = min(configs.fourier_modes, configs.sequence_len // 2 + 1)
        self.projection_dim = configs.projection_dim
        
        # Time encoder
        self.time_encoder = nn.Sequential(
            nn.Conv1d(self.input_channels, self.mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.mid_channels),
            nn.ReLU(),
            nn.Conv1d(self.mid_channels, self.mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.mid_channels),
            nn.ReLU()
        )
        
        # Frequency encoder
        self.freq_encoder = SpectralConv1d(
            in_channels=self.input_channels,
            out_channels=self.mid_channels,
            modes1=self.fourier_modes,
            fl=self.sequence_len
        )
        
        # Projection layer for frequency features
        self.freq_projection = nn.Sequential(
            nn.Conv1d(self.mid_channels, self.mid_channels, kernel_size=1),
            nn.BatchNorm1d(self.mid_channels),
            nn.ReLU(),
            nn.Upsample(size=self.sequence_len, mode='linear', align_corners=True)
        )
        
        # Projection layer for feature fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(self.mid_channels * 2, self.projection_dim, kernel_size=1),
            nn.BatchNorm1d(self.projection_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x: [batch, channels, length]
        
        # Get time features
        time_features = self.time_encoder(x)  # [batch, mid_channels, length]
        
        # Get frequency features
        freq_features, freq_ft = self.freq_encoder(x)  # [batch, mid_channels, modes1]
        
        # Project frequency features to desired dimensionality
        freq_features = self.freq_projection(freq_features)  # [batch, mid_channels, length]
        
        # Combine features
        features = torch.cat([time_features, freq_features], dim=1)  # [batch, mid_channels*2, length]
        features = self.fusion(features)  # [batch, projection_dim, length]
        
        return features, freq_ft

class tf_decoder(nn.Module):
    def __init__(self, configs):
        super(tf_decoder, self).__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(configs.final_out_channels, configs.mid_channels, 1),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU()
        )
        
        self.reconstruction = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.input_channels, 1),
            nn.BatchNorm1d(configs.input_channels)
        )
        
    def forward(self, x):
        # x: [B, final_out_channels, L]
        x = self.projection(x)  # [B, mid_channels, L]
        x = self.reconstruction(x)  # [B, input_channels, L]
        return x  # [B, input_channels, L]

class SinkhornDistance(nn.Module):
    def __init__(self, eps=1e-4, max_iter=2000, reduction='sum'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # Compute cost matrix
        C = torch.cdist(x, y, p=2)
        
        # Sinkhorn iterations
        n, m = x.size(0), y.size(0)
        a = torch.ones(n, device=x.device) / n
        b = torch.ones(m, device=x.device) / m
        
        # Normalize cost matrix
        C = C / torch.mean(C)
        
        # Sinkhorn iterations
        K = torch.exp(-C / self.eps)
        
        # Initialize dual variables
        u = torch.zeros_like(a)
        v = torch.zeros_like(b)
        
        # Sinkhorn iterations
        for _ in range(self.max_iter):
            u = a / (K @ v + 1e-8)
            v = b / (K.t() @ u + 1e-8)
        
        # Transport plan
        P = torch.diag(u) @ K @ torch.diag(v)
        
        # Transport cost
        cost = torch.sum(P * C)
        
        if self.reduction == 'mean':
            cost = cost / (n * m)
            
        return cost, u, v

class BaseAlgorithm(torch.nn.Module):
    def __init__(self, backbone_class, configs, hparams, device):
        """Инициализация базового алгоритма."""
        super().__init__()
        self.configs = configs
        self.hparams = hparams
        self.device = device
        
        # Инициализация feature extractor
        if backbone_class is not None:
            self.feature_extractor = backbone_class(configs).to(device)
        else:
            self.feature_extractor = CNN(configs).to(device)
            
        # Создаем dummy input для определения размерности
        dummy_input = torch.randn(1, configs.input_channels, configs.sequence_len).to(device)
        features = self.feature_extractor(dummy_input)  # Получаем только features
        
        # Определяем размерность входного слоя проекции
        if isinstance(features, tuple):
            features = features[0]
        input_dim = features.view(1, -1).size(1)
        
        # Инициализация проекционного слоя
        self.projection = nn.Sequential(
            nn.Linear(input_dim, self.hparams['projection_dim']),
            nn.BatchNorm1d(self.hparams['projection_dim']),
            nn.ReLU(),
            nn.Dropout(self.hparams['dropout'])
        ).to(device)
        
        # Decoder
        self.decoder = tf_decoder(configs).to(device)
        
        # Classifier
        self.classifier = nn.Linear(self.hparams['projection_dim'], configs.num_classes).to(device)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.hparams['projection_dim'], self.hparams['projection_dim']),
            nn.ReLU(),
            nn.Linear(self.hparams['projection_dim'], self.hparams['projection_dim'])
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams['learning_rate'],
            weight_decay=self.hparams['weight_decay']
        )
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        
        # Sinkhorn distance parameters
        self.epsilon = 0.01
        self.num_iters = 100
        
        # Temperature for contrastive loss
        self.temperature = self.hparams['temperature']
        
        # Initialize feature extractor and classifier
        self.feature_extractor.apply(weights_init)
        self.classifier.apply(weights_init)
        self.decoder.apply(weights_init)
        self.projection_head.apply(weights_init)
        
        # Initialize lists for predictions
        self.src_pred_labels = []
        self.src_true_labels = []
        self.trg_pred_labels = []
        self.trg_true_labels = []

    def parameters(self):
        """Возвращает все параметры модели"""
        return list(self.feature_extractor.parameters()) + \
               list(self.projection.parameters()) + \
               list(self.decoder.parameters()) + \
               list(self.classifier.parameters()) + \
               list(self.projection_head.parameters())
        
    def encode_features(self, x):
        """Кодирование признаков с помощью feature_extractor"""
        print("\nКодирование признаков:")
        print(f"Входная форма: {x.shape}")
        
        # Получаем признаки из экстрактора
        features = self.feature_extractor(x)
        print(f"Признаки после feature_extractor: {features.shape}")
        
        # Определяем тип бэкбона и обрабатываем признаки соответственно
        if isinstance(self.feature_extractor, TCN):
            print("Обработка TCN бэкбона")
            features = features.reshape(features.shape[0], -1)
            print(f"2D признаки TCN: {features.shape}")
        else:
            print("Обработка CNN бэкбона")
            if isinstance(features, tuple):
                features = features[0]  # Берем только признаки, игнорируем None
            features = features.reshape(features.shape[0], -1)
            print(f"2D признаки CNN: {features.shape}")
        
        # Проверяем размерность признаков
        if features.shape[1] != self.unified_feature_dim:
            print(f"Размерность не соответствует unified_feature_dim ({features.shape[1]} != {self.unified_feature_dim})")
            if not hasattr(self, 'dimension_adjust') or self.dimension_adjust is None:
                print(f"Создание dimension_adjust: {features.shape[1]} -> {self.unified_feature_dim}")
                self.dimension_adjust = nn.Sequential(
                    nn.Linear(features.shape[1], self.intermediate_dim),
                    nn.BatchNorm1d(self.intermediate_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(self.intermediate_dim, self.unified_feature_dim),
                    nn.BatchNorm1d(self.unified_feature_dim)
                ).to(self.device)
            features = self.dimension_adjust(features)
            print(f"После dimension_adjust: {features.shape}")
        else:
            print("Размерность соответствует unified_feature_dim")
            
        return features
        
    def update(self, src_x, src_y, trg_x):
        """Обновление модели на одном батче."""
        print(f"\nОбновление модели:")
        print(f"Source batch shape: {src_x.shape}")
        print(f"Target batch shape: {trg_x.shape}")
        
        # Извлекаем признаки
        src_features = self.encode_features(src_x)
        trg_features = self.encode_features(trg_x)
        print(f"Source features shape: {src_features.shape}")
        print(f"Target features shape: {trg_features.shape}")
        
        # Проверяем размерности перед классификацией
        if src_features.size(1) != self.unified_feature_dim:
            print(f"Ошибка: размерность признаков ({src_features.size(1)}) не соответствует unified_feature_dim ({self.unified_feature_dim})")
            return None
        
        # Выравниваем признаки
        src_features_flat = src_features.view(src_features.size(0), -1)
        trg_features_flat = trg_features.view(trg_features.size(0), -1)
        print(f"Flattened source features shape: {src_features_flat.shape}")
        print(f"Flattened target features shape: {trg_features_flat.shape}")
        
        # Проецируем признаки
        src_features_proj = self.projection(src_features_flat)
        trg_features_proj = self.projection(trg_features_flat)
        print(f"Projected source features shape: {src_features_proj.shape}")
        print(f"Projected target features shape: {trg_features_proj.shape}")
        
        # Классификация
        src_logits = self.classifier(src_features_proj)
        print(f"Source logits shape: {src_logits.shape}")
        
        # Подготавливаем признаки для декодера (преобразуем в 3D)
        channels = self.configs.final_out_channels
        length = self.configs.features_len
        src_features_3d = src_features_flat.view(src_features_flat.size(0), channels, -1)
        trg_features_3d = trg_features_flat.view(trg_features_flat.size(0), channels, -1)
        print(f"3D source features shape: {src_features_3d.shape}")
        print(f"3D target features shape: {trg_features_3d.shape}")
        
        # Реконструкция
        src_recon = self.decoder(src_features_3d)
        trg_recon = self.decoder(trg_features_3d)
        print(f"Source reconstruction shape: {src_recon.shape}")
        print(f"Target reconstruction shape: {trg_recon.shape}")
        
        # Интерполяция для восстановления исходной длины
        if src_recon.size(-1) != src_x.size(-1):
            src_recon = F.interpolate(src_recon, size=src_x.size(-1), mode='linear', align_corners=False)
            trg_recon = F.interpolate(trg_recon, size=trg_x.size(-1), mode='linear', align_corners=False)
            print(f"Interpolated source reconstruction shape: {src_recon.shape}")
            print(f"Interpolated target reconstruction shape: {trg_recon.shape}")
        
        # Вычисляем потери
        cls_loss = F.cross_entropy(src_logits, src_y)
        recon_loss = F.mse_loss(src_recon, src_x) + F.mse_loss(trg_recon, trg_x)
        
        # Контрастная потеря
        src_features_proj_norm = F.normalize(src_features_proj, dim=1)
        trg_features_proj_norm = F.normalize(trg_features_proj, dim=1)
        contrast_loss = -torch.mean(torch.sum(src_features_proj_norm * trg_features_proj_norm, dim=1))
        
        # Sinkhorn Distance для выравнивания распределений
        sinkhorn = SinkhornDistance(eps=self.hparams['sinkhorn_eps'], 
                                  max_iter=self.hparams['sinkhorn_max_iter'])
        sinkhorn_loss, _, _ = sinkhorn(src_features_proj, trg_features_proj)
        
        # Общая потеря
        total_loss = (cls_loss + 
                     self.hparams['recon_weight'] * recon_loss + 
                     self.hparams['contrast_weight'] * contrast_loss +
                     self.hparams['sinkhorn_weight'] * sinkhorn_loss)
        
        print(f"\nПотери:")
        print(f"Classification loss: {cls_loss.item():.4f}")
        print(f"Reconstruction loss: {recon_loss.item():.4f}")
        print(f"Contrastive loss: {contrast_loss.item():.4f}")
        print(f"Sinkhorn loss: {sinkhorn_loss.item():.4f}")
        print(f"Total loss: {total_loss.item():.4f}")
        
        # Обновление параметров
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'cls_loss': cls_loss.item(),
            'recon_loss': recon_loss.item(),
            'contrast_loss': contrast_loss.item(),
            'sinkhorn_loss': sinkhorn_loss.item(),
            'total_loss': total_loss.item()
        }
        
    def correct(self, src_x, src_y, trg_x):
        """Метод коррекции с бимодальным тестом и кластеризацией."""
        # Извлекаем признаки
        src_features = self.encode_features(src_x)
        trg_features = self.encode_features(trg_x)
        
        # Проецируем признаки
        src_features = self.projection(src_features)
        trg_features = self.projection(trg_features)
        
        # Классифицируем исходные данные
        src_pred = self.classifier(src_features)
        cls_loss = self.criterion(src_pred, src_y)
        
        # Бимодальный тест для целевого домена
        is_common = self.bimodal_test(trg_features)
        
        # Кластеризация приватных образцов
        private_labels = self.cluster_private_samples(trg_features, ~is_common)
        
        # Коррекция предсказаний
        trg_pred = self.classifier(trg_features)
        corrected_pred = trg_pred.clone()
        
        if len(private_labels) > 0:
            private_indices = torch.where(~is_common)[0]
            valid_private = private_labels >= 0
            valid_indices = private_indices[valid_private]
            valid_labels = private_labels[valid_private]
            
            if len(valid_indices) > 0:
                # Обнуляем вероятности для приватных классов и устанавливаем их метки
                corrected_pred[valid_indices] = F.one_hot(valid_labels, num_classes=self.configs.num_classes + len(torch.unique(valid_labels))).float()
        
        # Контрастивная потеря
        contrastive_loss = self.contrastive_loss(src_features, trg_features)
        
        # Общая потеря
        loss = cls_loss + self.hparams.get('lambda_contrast', 1.0) * contrastive_loss
        
        return loss
        
    def evaluate_metrics(self, src_pred_labels, src_true_labels, trg_pred_labels, trg_true_labels):
        # Convert predictions and labels to numpy arrays
        src_pred = np.concatenate(src_pred_labels)
        src_true = np.concatenate(src_true_labels)
        trg_pred = np.concatenate(trg_pred_labels)
        trg_true = np.concatenate(trg_true_labels)
        
        # Calculate accuracies
        src_acc = accuracy_score(src_true, src_pred)
        trg_acc = accuracy_score(trg_true, trg_pred)
        
        # Calculate F1 scores
        src_f1 = f1_score(src_true, src_pred, average='weighted')
        trg_f1 = f1_score(trg_true, trg_pred, average='weighted')
        
        # Calculate H-score
        h_score = 2 * (src_acc * trg_acc) / (src_acc + trg_acc) * 100
        
        return {
            'src_acc': src_acc,
            'trg_acc': trg_acc,
            'src_f1': src_f1,
            'trg_f1': trg_f1,
            'h_score': h_score
        }

    def compute_sinkhorn_distance(self, src_features, trg_features):
        # Normalize features
        src_features = F.normalize(src_features, dim=1)
        trg_features = F.normalize(trg_features, dim=1)
        
        # Compute cost matrix
        C = torch.cdist(src_features, trg_features, p=2)
        
        # Sinkhorn iterations
        n, m = src_features.size(0), trg_features.size(0)
        a = torch.ones(n, device=src_features.device) / n
        b = torch.ones(m, device=trg_features.device) / m
        
        # Normalize cost matrix
        C = C / torch.mean(C)
        
        # Sinkhorn iterations
        K = torch.exp(-C / self.epsilon)
        
        # Initialize dual variables
        u = torch.zeros_like(a)
        v = torch.zeros_like(b)
        
        # Sinkhorn iterations
        for _ in range(self.num_iters):
            u = a / (K @ v + 1e-8)
            v = b / (K.t() @ u + 1e-8)
        
        # Transport plan
        P = torch.diag(u) @ K @ torch.diag(v)
        
        # Transport cost
        cost = torch.sum(P * C)
        
        return cost

    def compute_contrastive_loss(self, src_features, trg_features):
        # Project features
        src_proj = self.projection_head(src_features)
        trg_proj = self.projection_head(trg_features)
        
        # Normalize features
        src_proj = F.normalize(src_proj, dim=1)
        trg_proj = F.normalize(trg_proj, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(src_proj, trg_proj.t()) / self.temperature
        
        # Create labels for contrastive loss (diagonal is positive pairs)
        labels = torch.arange(src_proj.size(0)).to(src_proj.device)
        
        # Compute loss
        loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels)
        loss = loss / 2
        
        return loss

    def _get_feature_dim(self):
        """Получает размерность признаков из бэкбона."""
        print("\nПолучение размерности признаков:")
        with torch.no_grad():
            # Создаем тестовый входной тензор
            x = torch.randn(1, self.configs.input_channels, self.configs.sequence_len).to(self.device)
            print(f"Тестовый вход: {x.shape}")
            
            # Получаем признаки
            features = self.feature_extractor(x)
            if isinstance(features, tuple):
                features = features[0]
            print(f"Признаки после feature_extractor: {features.shape}")
            
            # Определяем размерность
            if len(features.shape) == 3:  # [B, C, L]
                dim = features.size(1) * features.size(2)
                print(f"3D тензор, размерность после уплощения: {dim}")
            elif len(features.shape) == 2:  # [B, D]
                dim = features.size(1)
                print(f"2D тензор, размерность: {dim}")
            else:  # [B]
                dim = 1
                print(f"1D тензор, размерность: {dim}")
            
            return dim

class RAINCOAT(BaseAlgorithm):
    def __init__(self, backbone_class, configs, hparams, device, trg_train_size):
        """Инициализация RAINCOAT."""
        print("\nИнициализация RAINCOAT:")
        print(f"Backbone: {backbone_class.__name__}")
        print(f"Device: {device}")
        print(f"Target train size: {trg_train_size}")
        
        # Инициализируем атрибуты до вызова родительского конструктора
        self.trg_train_size = trg_train_size
        self.feature_transform = None
        self.source_transform = None
        self.target_transform = None
        self.adaptive_pool = None
        self.unified_feature_dim = 1536
        self.intermediate_dim = 2048
        self.projection = None
        self.classifier = None
        self.decoder = None
        self.projection_head = None
        self.optimizer = None
        self.dimension_adjust = None
        self.device = device
        
        # Добавляем поддержку кросс-датасет обучения
        self.source_dataset = configs.dataset_name
        self.target_dataset = configs.target_dataset if hasattr(configs, 'target_dataset') else configs.dataset_name
        print(f"\nДатасеты:")
        print(f"Source dataset: {self.source_dataset}")
        print(f"Target dataset: {self.target_dataset}")
        
        # Определяем тип бэкбона
        self.backbone_type = "CNN" if "CNN" in backbone_class.__name__ else ("TCNAttention" if "TCNAttention" in backbone_class.__name__ else "TCN")
        print(f"\nТип бэкбона: {self.backbone_type}")
        
        # Определяем размерности признаков для разных датасетов
        self.dataset_dims = {
            'WISDM': {'TCN': 12352, 'CNN': 1536, 'TCNAttention': 12352},
            'HHAR_SA': {'TCN': 3648, 'CNN': 448, 'TCNAttention': 3648},
            'UCI': {'TCN': 5120, 'CNN': 512, 'TCNAttention': 5120},
            'OPPORTUNITY': {'TCN': 8192, 'CNN': 1024, 'TCNAttention': 8192},
            'PAMAP2': {'TCN': 6144, 'CNN': 768, 'TCNAttention': 6144},
            'USCHAD': {'TCN': 7168, 'CNN': 896, 'TCNAttention': 7168},
            'DSADS': {'TCN': 5120, 'CNN': 640, 'TCNAttention': 5120},
            'MHEALTH': {'TCN': 3072, 'CNN': 384, 'TCNAttention': 3072}
        }
        
        # Устанавливаем размерность признаков в зависимости от датасета и бэкбона
        if self.source_dataset in self.dataset_dims:
            self.unified_feature_dim = self.dataset_dims[self.source_dataset][self.backbone_type]
            print(f"Установлена размерность признаков для {self.source_dataset} ({self.backbone_type}): {self.unified_feature_dim}")
        
        if self.target_dataset in self.dataset_dims:
            self.target_feature_dim = self.dataset_dims[self.target_dataset][self.backbone_type]
            print(f"Установлена размерность признаков для {self.target_dataset} ({self.backbone_type}): {self.target_feature_dim}")
            
            # Создаем dimension_adjust для кросс-датасетного обучения
            if self.target_feature_dim != self.unified_feature_dim:
                print(f"Создание dimension_adjust для кросс-датасетного обучения: {self.target_feature_dim} -> {self.unified_feature_dim}")
                self.dimension_adjust = nn.Sequential(
                    nn.Linear(self.target_feature_dim, self.intermediate_dim),
                    nn.BatchNorm1d(self.intermediate_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(self.intermediate_dim, self.unified_feature_dim),
                    nn.BatchNorm1d(self.unified_feature_dim)
                ).to(device)
        
        # Добавляем недостающие параметры в конфигурацию
        if not hasattr(configs, 'final_out_channels'):
            configs.final_out_channels = 64
        if not hasattr(configs, 'mid_channels'):
            configs.mid_channels = 32
        if not hasattr(configs, 'features_len'):
            configs.features_len = 24
        if not hasattr(configs, 'sequence_len'):
            configs.sequence_len = 200
        
        print("\nКонфигурация:")
        print(f"Final out channels: {configs.final_out_channels}")
        print(f"Mid channels: {configs.mid_channels}")
        print(f"Features length: {configs.features_len}")
        print(f"Sequence length: {configs.sequence_len}")
        
        # Добавляем недостающие параметры в hparams
        if 'projection_dim' not in hparams:
            hparams['projection_dim'] = 128
        if 'recon_weight' not in hparams:
            hparams['recon_weight'] = 1.0
        if 'contrast_weight' not in hparams:
            hparams['contrast_weight'] = 0.1
        if 'temperature' not in hparams:
            hparams['temperature'] = 0.07
        if 'learning_rate' not in hparams:
            hparams['learning_rate'] = 1e-4
        if 'weight_decay' not in hparams:
            hparams['weight_decay'] = 1e-5
        if 'sinkhorn_weight' not in hparams:
            hparams['sinkhorn_weight'] = 0.1
        if 'sinkhorn_eps' not in hparams:
            hparams['sinkhorn_eps'] = 0.01
        if 'sinkhorn_max_iter' not in hparams:
            hparams['sinkhorn_max_iter'] = 100
        if 'entropy_threshold' not in hparams:
            hparams['entropy_threshold'] = 0.5
        if 'eps' not in hparams:
            hparams['eps'] = 0.5
        if 'min_samples' not in hparams:
            hparams['min_samples'] = 5
        
        print("\nГиперпараметры:")
        for k, v in hparams.items():
            print(f"{k}: {v}")
        
        # Сначала вызываем родительский конструктор
        super(RAINCOAT, self).__init__(backbone_class, configs, hparams, device)
        
        # Получаем размерность признаков из бэкбона
        feature_dim = self._get_feature_dim()
        print(f"\nРазмерность признаков: {feature_dim}")
        
        # Инициализируем проекционный слой
        self.projection = nn.Sequential(
            nn.Linear(self.unified_feature_dim, self.intermediate_dim),
            nn.ReLU(),
            nn.Linear(self.intermediate_dim, hparams['projection_dim'])
        ).to(device)
        
        # Инициализируем классификатор
        self.classifier = nn.Linear(hparams['projection_dim'], configs.num_classes).to(device)
        
        # Инициализируем декодер
        self.decoder = tf_decoder(configs).to(device)
        
        # Инициализируем проекционный слой для контрастного обучения
        self.projection_head = nn.Sequential(
            nn.Linear(self.unified_feature_dim, self.unified_feature_dim),
            nn.ReLU(),
            nn.Linear(self.unified_feature_dim, self.unified_feature_dim)
        ).to(device)
        
        # Инициализируем оптимизатор с различными скоростями обучения
        self.optimizer = torch.optim.Adam([
            {"params": self.feature_extractor.parameters(), "lr": hparams.get('learning_rate', 0.001)},
            {"params": self.classifier.parameters(), "lr": hparams.get('learning_rate', 0.001) * 10},
            {"params": self.decoder.parameters(), "lr": hparams.get('learning_rate', 0.001)},
            {"params": self.projection.parameters(), "lr": hparams.get('learning_rate', 0.001)},
            {"params": self.projection_head.parameters(), "lr": hparams.get('learning_rate', 0.001)}
        ], weight_decay=hparams.get('weight_decay', 0.0005))
        
        # Добавляем параметры dimension_adjust в оптимизатор, если он существует
        if self.dimension_adjust is not None:
            self.optimizer.add_param_group({
                "params": self.dimension_adjust.parameters(),
                "lr": hparams.get('learning_rate', 0.001) * 5
            })
        
        # Параметры для кластеризации
        self.entropy_threshold = hparams.get('entropy_threshold', 0.5)
        self.eps = hparams.get('eps', 0.5)
        self.min_samples = hparams.get('min_samples', 5)
        
        print("\nИнициализация завершена!")

    def bimodal_test(self, features):
        """Бимодальный тест для разделения общих и приватных классов."""
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        threshold = torch.quantile(entropy, self.entropy_threshold)
        is_common = entropy < threshold
        return is_common
        
    def cluster_private_samples(self, features, is_private):
        """Кластеризация приватных образцов."""
        private_features = features[is_private].detach().cpu().numpy()
        if len(private_features) > 0:
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(private_features)
            labels = clustering.labels_
            # Присваиваем метки приватным классам, начиная с num_classes
            private_labels = labels + self.configs.num_classes
            # Фильтруем шумовые точки
            private_labels[private_labels < self.configs.num_classes] = -1
            return torch.tensor(private_labels, device=self.device)
        return torch.tensor([], device=self.device)
        
    def update(self, src_x, src_y, trg_x):
        """Обновление модели на одном батче."""
        print(f"\nОбновление модели:")
        print(f"Source batch shape: {src_x.shape}")
        print(f"Target batch shape: {trg_x.shape}")
        
        # Извлекаем признаки
        src_features = self.encode_features(src_x)
        trg_features = self.encode_features(trg_x)
        print(f"Source features shape: {src_features.shape}")
        print(f"Target features shape: {trg_features.shape}")
        
        # Проверяем размерности перед классификацией
        if src_features.size(1) != self.unified_feature_dim:
            print(f"Ошибка: размерность признаков ({src_features.size(1)}) не соответствует unified_feature_dim ({self.unified_feature_dim})")
            return None
        
        # Выравниваем признаки
        src_features_flat = src_features.view(src_features.size(0), -1)
        trg_features_flat = trg_features.view(trg_features.size(0), -1)
        print(f"Flattened source features shape: {src_features_flat.shape}")
        print(f"Flattened target features shape: {trg_features_flat.shape}")
        
        # Проецируем признаки
        src_features_proj = self.projection(src_features_flat)
        trg_features_proj = self.projection(trg_features_flat)
        print(f"Projected source features shape: {src_features_proj.shape}")
        print(f"Projected target features shape: {trg_features_proj.shape}")
        
        # Классификация
        src_logits = self.classifier(src_features_proj)
        print(f"Source logits shape: {src_logits.shape}")
        
        # Подготавливаем признаки для декодера (преобразуем в 3D)
        channels = self.configs.final_out_channels
        length = self.configs.features_len
        src_features_3d = src_features_flat.view(src_features_flat.size(0), channels, -1)
        trg_features_3d = trg_features_flat.view(trg_features_flat.size(0), channels, -1)
        print(f"3D source features shape: {src_features_3d.shape}")
        print(f"3D target features shape: {trg_features_3d.shape}")
        
        # Реконструкция
        src_recon = self.decoder(src_features_3d)
        trg_recon = self.decoder(trg_features_3d)
        print(f"Source reconstruction shape: {src_recon.shape}")
        print(f"Target reconstruction shape: {trg_recon.shape}")
        
        # Интерполяция для восстановления исходной длины
        if src_recon.size(-1) != src_x.size(-1):
            src_recon = F.interpolate(src_recon, size=src_x.size(-1), mode='linear', align_corners=False)
            trg_recon = F.interpolate(trg_recon, size=trg_x.size(-1), mode='linear', align_corners=False)
            print(f"Interpolated source reconstruction shape: {src_recon.shape}")
            print(f"Interpolated target reconstruction shape: {trg_recon.shape}")
        
        # Вычисляем потери
        cls_loss = F.cross_entropy(src_logits, src_y)
        recon_loss = F.mse_loss(src_recon, src_x) + F.mse_loss(trg_recon, trg_x)
        
        # Контрастная потеря
        src_features_proj_norm = F.normalize(src_features_proj, dim=1)
        trg_features_proj_norm = F.normalize(trg_features_proj, dim=1)
        contrast_loss = -torch.mean(torch.sum(src_features_proj_norm * trg_features_proj_norm, dim=1))
        
        # Sinkhorn Distance для выравнивания распределений
        sinkhorn = SinkhornDistance(eps=self.hparams['sinkhorn_eps'], 
                                  max_iter=self.hparams['sinkhorn_max_iter'])
        sinkhorn_loss, _, _ = sinkhorn(src_features_proj, trg_features_proj)
        
        # Общая потеря
        total_loss = (cls_loss + 
                     self.hparams['recon_weight'] * recon_loss + 
                     self.hparams['contrast_weight'] * contrast_loss +
                     self.hparams['sinkhorn_weight'] * sinkhorn_loss)
        
        print(f"\nПотери:")
        print(f"Classification loss: {cls_loss.item():.4f}")
        print(f"Reconstruction loss: {recon_loss.item():.4f}")
        print(f"Contrastive loss: {contrast_loss.item():.4f}")
        print(f"Sinkhorn loss: {sinkhorn_loss.item():.4f}")
        print(f"Total loss: {total_loss.item():.4f}")
        
        # Обновление параметров
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'cls_loss': cls_loss.item(),
            'recon_loss': recon_loss.item(),
            'contrast_loss': contrast_loss.item(),
            'sinkhorn_loss': sinkhorn_loss.item(),
            'total_loss': total_loss.item()
        }
        
    def correct(self, src_x, src_y, trg_x):
        """Метод коррекции с бимодальным тестом и кластеризацией."""
        # Извлекаем признаки
        src_features = self.encode_features(src_x)
        trg_features = self.encode_features(trg_x)
        
        # Проецируем признаки
        src_features = self.projection(src_features)
        trg_features = self.projection(trg_features)
        
        # Классифицируем исходные данные
        src_pred = self.classifier(src_features)
        cls_loss = self.criterion(src_pred, src_y)
        
        # Бимодальный тест для целевого домена
        is_common = self.bimodal_test(trg_features)
        
        # Кластеризация приватных образцов
        private_labels = self.cluster_private_samples(trg_features, ~is_common)
        
        # Коррекция предсказаний
        trg_pred = self.classifier(trg_features)
        corrected_pred = trg_pred.clone()
        
        if len(private_labels) > 0:
            private_indices = torch.where(~is_common)[0]
            valid_private = private_labels >= 0
            valid_indices = private_indices[valid_private]
            valid_labels = private_labels[valid_private]
            
            if len(valid_indices) > 0:
                # Обнуляем вероятности для приватных классов и устанавливаем их метки
                corrected_pred[valid_indices] = F.one_hot(valid_labels, num_classes=self.configs.num_classes + len(torch.unique(valid_labels))).float()
        
        # Контрастивная потеря
        contrastive_loss = self.contrastive_loss(src_features, trg_features)
        
        # Общая потеря
        loss = cls_loss + self.hparams.get('lambda_contrast', 1.0) * contrastive_loss
        
        return loss

    def parameters(self):
        """Возвращает все параметры модели для оптимизации."""
        params = [
            {"params": self.feature_extractor.parameters()},
            {"params": self.classifier.parameters()},
            {"params": self.decoder.parameters()},
            {"params": self.projection.parameters()},
            {"params": self.projection_head.parameters()}
        ]
        
        if self.feature_transform is not None:
            params.append({"params": self.feature_transform.parameters()})
        if self.source_transform is not None:
            params.append({"params": self.source_transform.parameters()})
        if self.target_transform is not None:
            params.append({"params": self.target_transform.parameters()})
            
        return params

    def encode_features(self, x):
        """Кодирование признаков с помощью feature_extractor"""
        print("\nКодирование признаков:")
        print(f"Входная форма: {x.shape}")
        
        # Получаем признаки из экстрактора
        features = self.feature_extractor(x)
        print(f"Признаки после feature_extractor: {features.shape}")
        
        # Определяем тип бэкбона и обрабатываем признаки соответственно
        if isinstance(self.feature_extractor, TCN):
            print("Обработка TCN бэкбона")
            features = features.reshape(features.shape[0], -1)
            print(f"2D признаки TCN: {features.shape}")
        else:
            print("Обработка CNN бэкбона")
            if isinstance(features, tuple):
                features = features[0]  # Берем только признаки, игнорируем None
            features = features.reshape(features.shape[0], -1)
            print(f"2D признаки CNN: {features.shape}")
        
        # Проверяем размерность признаков
        if features.shape[1] != self.unified_feature_dim:
            print(f"Размерность не соответствует unified_feature_dim ({features.shape[1]} != {self.unified_feature_dim})")
            if not hasattr(self, 'dimension_adjust') or self.dimension_adjust is None:
                print(f"Создание dimension_adjust: {features.shape[1]} -> {self.unified_feature_dim}")
                self.dimension_adjust = nn.Sequential(
                    nn.Linear(features.shape[1], self.intermediate_dim),
                    nn.BatchNorm1d(self.intermediate_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(self.intermediate_dim, self.unified_feature_dim),
                    nn.BatchNorm1d(self.unified_feature_dim)
                ).to(self.device)
            features = self.dimension_adjust(features)
            print(f"После dimension_adjust: {features.shape}")
        else:
            print("Размерность соответствует unified_feature_dim")
            
        return features

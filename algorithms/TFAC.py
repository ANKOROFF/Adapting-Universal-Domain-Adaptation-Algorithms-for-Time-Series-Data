import torch
import torch.nn as nn
import torch.nn.functional as F
from models.loss import SinkhornDistance
from pytorch_metric_learning import losses
import numpy as np

class TFAC(nn.Module):
    def __init__(self, configs, hparams, device):
        super(TFAC, self).__init__()
        self.configs = configs
        self.hparams = hparams
        self.device = device
        
        # CNN слои для обработки временных рядов
        self.encoder = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        ).to(device)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, configs.num_classes)
        ).to(device)
        
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sink = SinkhornDistance(eps=1e-3, max_iter=1000, reduction='sum')
        
    def forward(self, x):
        features = self.encoder(x)
        return features
        
    def update(self, src_x, src_y, trg_x):
        self.optimizer.zero_grad()
        
        # Извлечение признаков
        src_features = self.encoder(src_x)
        trg_features = self.encoder(trg_x)
        
        # Классификация
        src_logits = self.classifier(src_features)
        cls_loss = self.cross_entropy(src_logits, src_y)
        
        # Выравнивание доменов
        dr, _, _ = self.sink(src_features, trg_features)
        sink_loss = dr
        
        # Общая функция потерь
        loss = cls_loss + sink_loss
        loss.backward()
        self.optimizer.step()
        
        return {'cls_loss': cls_loss.item(), 'sink_loss': sink_loss.item()}
        
    def correct(self, src_x, src_y, trg_x):
        self.optimizer.zero_grad()
        
        # Реконструкция для коррекции
        src_features = self.encoder(src_x)
        trg_features = self.encoder(trg_x)
        
        # Минимизация реконструкционной ошибки
        recon_loss = F.mse_loss(src_features, trg_features)
        recon_loss.backward()
        self.optimizer.step()
        
        return {'recon_loss': recon_loss.item()} 
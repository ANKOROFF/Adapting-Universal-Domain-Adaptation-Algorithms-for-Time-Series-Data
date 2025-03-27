import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.base import BaseAlgorithm
from algorithms.utils import entropy
from sklearn.metrics import accuracy_score, f1_score

class OVANet(BaseAlgorithm):
    """
    OVANet implementation
    """
    def __init__(self, backbone_class, configs, hparams, device):
        super(OVANet, self).__init__(backbone_class, configs, hparams, device)
        
        # Инициализация feature extractor
        self.feature_extractor = backbone_class(configs).to(device)
        
        # Получаем размерность признаков
        dummy_input = torch.randn(1, configs.input_channels, configs.sequence_len).to(device)
        features = self.feature_extractor(dummy_input)
        if len(features.shape) == 3:
            features = features.mean(dim=2)  # Агрегируем по временной оси
        feature_dim = features.shape[1]
        print(f"Initial feature dimension: {feature_dim}")
        
        # Определяем тип backbone
        self.backbone_type = 'TCN' if 'TCN' in str(backbone_class) else 'CNN'
        print(f"Backbone type: {self.backbone_type}")
        
        # Унифицированная размерность для всех backbone
        self.unified_dim = 1536
        
        # Добавляем слои для унификации размерности
        self.feature_transform_tcn_wisdm = nn.Sequential(
            nn.Linear(12352, self.unified_dim),  # Для TCN WISDM
            nn.BatchNorm1d(self.unified_dim),
            nn.ReLU()
        ).to(device)
        
        self.feature_transform_tcn_hhar = nn.Sequential(
            nn.Linear(3648, self.unified_dim),  # Для TCN HHAR_SA
            nn.BatchNorm1d(self.unified_dim),
            nn.ReLU()
        ).to(device)
        
        self.feature_transform_cnn = nn.Sequential(
            nn.Linear(448, self.unified_dim),  # Для CNN HHAR_SA
            nn.BatchNorm1d(self.unified_dim),
            nn.ReLU()
        ).to(device)
        
        # Проекционный слой для уменьшения размерности
        self.projection = nn.Sequential(
            nn.Linear(self.unified_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        ).to(device)
        
        # Основной классификатор
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, configs.num_classes)
        ).to(device)
        
        # Инициализация one-vs-all классификаторов
        self.ova_classifiers = nn.ModuleList([
            nn.Linear(512, 1).to(device) for _ in range(configs.num_classes)
        ])
        
        # Настройка гиперпараметров
        if 'temperature' not in hparams:
            hparams['temperature'] = 0.1
        if 'ova_weight' not in hparams:
            hparams['ova_weight'] = 1.0
        if 'entropy_weight' not in hparams:
            hparams['entropy_weight'] = 0.1
        if 'learning_rate' not in hparams:
            hparams['learning_rate'] = 1e-4
        if 'weight_decay' not in hparams:
            hparams['weight_decay'] = 1e-5
        if 'grad_clip' not in hparams:
            hparams['grad_clip'] = 1.0
        if 'class_balance' not in hparams:
            hparams['class_balance'] = True
        if 'feature_aggregation' not in hparams:
            hparams['feature_aggregation'] = 'mean'
        if 'entropy_scale' not in hparams:
            hparams['entropy_scale'] = 1.0
        if 'ova_entropy_weight' not in hparams:
            hparams['ova_entropy_weight'] = 0.05
        
        # Оптимизатор
        self.optimizer = torch.optim.Adam(
            self.get_parameters(),
            lr=hparams['learning_rate'],
            weight_decay=hparams['weight_decay']
        )
        
    def get_parameters(self):
        """Returns parameters that will be updated during training"""
        return [
            {"params": self.feature_extractor.parameters()},
            {"params": self.feature_transform_tcn_wisdm.parameters()},
            {"params": self.feature_transform_tcn_hhar.parameters()},
            {"params": self.feature_transform_cnn.parameters()},
            {"params": self.projection.parameters()},
            {"params": self.classifier.parameters()},
            {"params": self.ova_classifiers.parameters()}
        ]
        
    def _get_ova_labels(self, labels, num_classes):
        """Convert regular labels to one-vs-all format"""
        batch_size = labels.size(0)
        ova_labels = torch.zeros(batch_size, num_classes).to(self.device)
        ova_labels.scatter_(1, labels.unsqueeze(1), 1)
        return ova_labels
        
    def _get_class_weights(self, labels):
        """Вычисляет веса классов для балансировки потерь"""
        class_counts = torch.bincount(labels, minlength=self.configs.num_classes)
        total = len(labels)
        weights = total / (self.configs.num_classes * class_counts.float())
        return weights.to(self.device)
        
    def encode_features(self, x):
        """Извлекает признаки из входных данных."""
        print(f"\nencode_features:")
        print(f"Input shape: {x.shape}")
        
        # Получаем признаки из backbone
        features = self.feature_extractor(x)
        if isinstance(features, tuple):
            features = features[0]
        if len(features.shape) == 3:
            features = features.reshape(features.size(0), -1)
        print(f"After feature_extractor: {features.shape}")
        
        # Нормализуем признаки
        features = F.normalize(features, p=2, dim=1)
        print(f"After normalize: {features.shape}")
        
        # Обрабатываем NaN/Inf значения
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Применяем преобразование размерности в зависимости от размера признаков
        if features.size(1) == 12352:  # TCN WISDM
            features = self.feature_transform_tcn_wisdm(features)
        elif features.size(1) == 448:  # CNN HHAR_SA
            features = self.feature_transform_cnn(features)
        elif features.size(1) == 3648:  # TCN HHAR_SA
            features = self.feature_transform_tcn_hhar(features)
        print(f"After feature_transform: {features.shape}")
        
        # Применяем проекционный слой
        features = self.projection(features)
        print(f"After projection: {features.shape}")
        
        # Нормализуем спроецированные признаки
        features = F.normalize(features, p=2, dim=1)
        print(f"After final normalize: {features.shape}")
        
        return features
        
    def update(self, src_x, src_y, trg_x):
        """Обновление модели на одном батче."""
        # Извлекаем признаки
        src_features = self.encode_features(src_x)
        trg_features = self.encode_features(trg_x)
        
        # Основная классификация
        src_logits = self.classifier(src_features)
        trg_logits = self.classifier(trg_features)
        
        # Классификационная потеря с весами классов
        if self.hparams['class_balance']:
            class_weights = self._get_class_weights(src_y)
        else:
            class_weights = None
        cls_loss = F.cross_entropy(src_logits, src_y, weight=class_weights)
        
        # One-vs-all классификация
        ova_labels = self._get_ova_labels(src_y, len(self.ova_classifiers))
        ova_loss = 0
        for i, ova_classifier in enumerate(self.ova_classifiers):
            src_ova_logits = ova_classifier(src_features).squeeze(-1)  # [B, 1] -> [B]
            ova_loss += F.binary_cross_entropy_with_logits(src_ova_logits, ova_labels[:, i].float())
        ova_loss = ova_loss / len(self.ova_classifiers)
        
        # Энтропийная потеря для основного классификатора
        trg_probs = F.softmax(trg_logits / self.hparams['temperature'], dim=1)
        entropy_loss = -(trg_probs * torch.log(trg_probs + 1e-10)).sum(dim=1).mean()
        
        # Энтропийная потеря для one-vs-all классификаторов
        ova_entropy_loss = 0
        for ova_classifier in self.ova_classifiers:
            trg_ova_logits = ova_classifier(trg_features).squeeze(-1)  # [B, 1] -> [B]
            trg_ova_probs = torch.sigmoid(trg_ova_logits)
            ova_entropy_loss += -(trg_ova_probs * torch.log(trg_ova_probs + 1e-10) + 
                                (1 - trg_ova_probs) * torch.log(1 - trg_ova_probs + 1e-10)).mean()
        ova_entropy_loss = ova_entropy_loss / len(self.ova_classifiers)
        
        # Общая потеря
        total_loss = (cls_loss + 
                     self.hparams['ova_weight'] * ova_loss +
                     self.hparams['entropy_weight'] * entropy_loss * self.hparams['entropy_scale'] +
                     self.hparams['ova_entropy_weight'] * ova_entropy_loss)
        
        # Обновление параметров
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Клиппинг градиентов
        parameters = []
        for param_group in self.get_parameters():
            parameters.extend(param_group['params'])
        torch.nn.utils.clip_grad_norm_(parameters, self.hparams['grad_clip'])
        
        self.optimizer.step()
        
        return {
            'cls_loss': cls_loss.item(),
            'ova_loss': ova_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'ova_entropy_loss': ova_entropy_loss.item(),
            'total_loss': total_loss.item()
        } 

    def evaluate(self, src_dl, trg_dl):
        """Оценка модели на тестовых данных"""
        print("\nОценка модели:")
        
        # Оценка на исходном домене
        print("\nОценка на исходном домене:")
        src_features_list = []
        src_labels_list = []
        src_preds_list = []
        
        self.eval()
        with torch.no_grad():
            for src_x, src_y in src_dl:
                src_x = src_x.to(self.device)
                src_y = src_y.to(self.device)
                
                # Получаем признаки
                src_features = self.encode_features(src_x)
                
                # Получаем предсказания
                src_logits = self.classifier(src_features)
                src_preds = torch.argmax(src_logits, dim=1)
                
                src_features_list.append(src_features)
                src_labels_list.append(src_y)
                src_preds_list.append(src_preds)
        
        src_features = torch.cat(src_features_list, dim=0)
        src_labels = torch.cat(src_labels_list, dim=0)
        src_preds = torch.cat(src_preds_list, dim=0)
        
        # Оценка на целевом домене
        print("\nОценка на целевом домене:")
        trg_features_list = []
        trg_labels_list = []
        trg_preds_list = []
        
        with torch.no_grad():
            for trg_x, trg_y in trg_dl:
                trg_x = trg_x.to(self.device)
                trg_y = trg_y.to(self.device)
                
                # Получаем признаки
                trg_features = self.encode_features(trg_x)
                
                # Получаем предсказания
                trg_logits = self.classifier(trg_features)
                trg_preds = torch.argmax(trg_logits, dim=1)
                
                trg_features_list.append(trg_features)
                trg_labels_list.append(trg_y)
                trg_preds_list.append(trg_preds)
        
        trg_features = torch.cat(trg_features_list, dim=0)
        trg_labels = torch.cat(trg_labels_list, dim=0)
        trg_preds = torch.cat(trg_preds_list, dim=0)
        
        # Вычисляем метрики
        src_acc = accuracy_score(src_labels.cpu().numpy(), src_preds.cpu().numpy())
        trg_acc = accuracy_score(trg_labels.cpu().numpy(), trg_preds.cpu().numpy())
        trg_f1 = f1_score(trg_labels.cpu().numpy(), trg_preds.cpu().numpy(), average='weighted')
        h_score = 2 * (src_acc * trg_acc) / (src_acc + trg_acc)
        
        print(f"\nРезультаты:")
        print(f"Source accuracy: {src_acc:.4f}")
        print(f"Target accuracy: {trg_acc:.4f}")
        print(f"Target F1-score: {trg_f1:.4f}")
        print(f"H-score: {h_score:.4f}")
        
        return trg_acc, trg_f1, h_score 
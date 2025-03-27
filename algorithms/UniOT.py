import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.base import BaseAlgorithm
from algorithms.utils import entropy
from models.backbones import CNNUniOT, TCNUniOT

# Отключаем CUDNN для предотвращения ошибок с in-place операциями
torch.backends.cudnn.enabled = False

# Включаем обнаружение аномалий
torch.autograd.set_detect_anomaly(True)

class UniOT(BaseAlgorithm):
    def __init__(self, backbone_class, configs, hparams, device):
        # Инициализируем базовый класс
        super(UniOT, self).__init__(backbone_class, configs, hparams, device)
        
        # Получаем размерность признаков из backbone
        self.feature_dim = self._get_feature_dim()
        print(f"Initial feature dimension: {self.feature_dim}")
        
        # Определяем тип бэкбона
        self.backbone_type = "CNN" if "CNN" in str(backbone_class) else ("TCNAttention" if "TCNAttention" in str(backbone_class) else "TCN")
        print(f"Backbone type: {self.backbone_type}")
        
        # Унифицированная размерность для всех backbone
        self.unified_dim = 1536
        
        # Добавляем слои для унификации размерности
        self.feature_transform_tcn_wisdm = nn.Sequential(
            nn.Linear(12352, self.unified_dim),  # Для TCN WISDM
            nn.BatchNorm1d(self.unified_dim),
            nn.ReLU()
        )
        
        self.feature_transform_tcn_hhar = nn.Sequential(
            nn.Linear(3648, self.unified_dim),  # Для TCN HHAR_SA
            nn.BatchNorm1d(self.unified_dim),
            nn.ReLU()
        )
        
        self.feature_transform_cnn = nn.Sequential(
            nn.Linear(448, self.unified_dim),  # Для CNN HHAR_SA
            nn.BatchNorm1d(self.unified_dim),
            nn.ReLU()
        )
        
        # Проекционный слой для уменьшения размерности
        self.projection = nn.Sequential(
            nn.Linear(self.unified_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        # Классификатор
        self.classifier = nn.Linear(512, configs.num_classes)
        
        # Инициализация прототипов
        self.source_prototypes = nn.Parameter(torch.randn(configs.num_classes, 512))
        self.target_prototypes = nn.Parameter(torch.randn(hparams.get('num_target_prototypes', 100), 512))
        
        # Нормализация прототипов
        with torch.no_grad():
            self.source_prototypes.data = F.normalize(self.source_prototypes.data, p=2, dim=1)
            self.target_prototypes.data = F.normalize(self.target_prototypes.data, p=2, dim=1)
        
        # Очередь памяти для хранения признаков
        self.memory_size = hparams.get('memory_size', 4000)
        self.memory_queue = torch.zeros(self.memory_size, 512, device=device)
        self.memory_ptr = 0
        
        # Параметры
        self.num_source_classes = configs.num_classes
        self.num_target_prototypes = hparams.get('num_target_prototypes', 100)
        self.epsilon = hparams.get('epsilon', 0.05)
        self.kappa = hparams.get('kappa', 0.3)
        self.tau = hparams.get('tau', 0.05)
        self.gamma = hparams.get('gamma', 0.5)
        self.mu = hparams.get('mu', 0.9)
        self.lambda_weight = hparams.get('lambda_weight', 0.5)
        self.num_iterations = hparams.get('num_iterations', 20)
        self.min_epsilon = 1e-8
        
        # Оптимизатор
        self.optimizer = torch.optim.Adam([
            {'params': self.feature_extractor.parameters(), 'lr': hparams.get('learning_rate', 0.0001)},
            {'params': self.feature_transform_tcn_wisdm.parameters(), 'lr': hparams.get('learning_rate', 0.0001)},
            {'params': self.feature_transform_tcn_hhar.parameters(), 'lr': hparams.get('learning_rate', 0.0001)},
            {'params': self.feature_transform_cnn.parameters(), 'lr': hparams.get('learning_rate', 0.0001)},
            {'params': self.projection.parameters(), 'lr': hparams.get('learning_rate', 0.0001)},
            {'params': self.classifier.parameters(), 'lr': hparams.get('learning_rate', 0.0001)},
            {'params': [self.source_prototypes], 'lr': hparams.get('learning_rate', 0.0001) * 0.1},
            {'params': [self.target_prototypes], 'lr': hparams.get('learning_rate', 0.0001) * 0.1}
        ], lr=hparams.get('learning_rate', 0.0001), weight_decay=hparams.get('weight_decay', 0.0001))
        
        # Перемещаем все компоненты на нужное устройство
        self.to(device)
        
    def get_parameters(self):
        return [
            {"params": self.feature_extractor.parameters()},
            {"params": self.feature_transform_tcn_wisdm.parameters()},
            {"params": self.feature_transform_tcn_hhar.parameters()},
            {"params": self.feature_transform_cnn.parameters()},
            {"params": self.projection.parameters()},
            {"params": self.classifier.parameters()},
            {"params": self.source_prototypes},
            {"params": self.target_prototypes}
        ]
        
    def _stabilized_log(self, x):
        """Логарифм с числовой стабильностью"""
        return torch.log(torch.clamp(x, min=self.min_epsilon))
    
    def _uot_sinkhorn(self, similarity_matrix, a, b):
        """Unbalanced OT с использованием обобщенного Sinkhorn"""
        batch_size = similarity_matrix.size(0)
        num_prototypes = similarity_matrix.size(1)
        
        K = torch.exp(similarity_matrix / self.epsilon)
        K = torch.clamp(K, min=self.min_epsilon, max=1.0)
        
        u = torch.ones(batch_size, 1, device=self.device)
        v = torch.ones(num_prototypes, 1, device=self.device)
        
        for _ in range(self.num_iterations):
            # Обновление u
            Kv = torch.matmul(K, v)
            u = a * torch.exp(-self.kappa * torch.log(Kv + self.min_epsilon))
            u = torch.clamp(u, min=self.min_epsilon)
            
            # Обновление v
            Kut = torch.matmul(K.t(), u)
            v = b * torch.exp(-self.kappa * torch.log(Kut + self.min_epsilon))
            v = torch.clamp(v, min=self.min_epsilon)
        
        P = torch.matmul(torch.matmul(torch.diag(u.squeeze()), K), torch.diag(v.squeeze()))
        P = torch.clamp(P, min=self.min_epsilon)
        P = P / (torch.sum(P) + self.min_epsilon)
        return P
    
    def _ot_sinkhorn(self, similarity_matrix):
        """Стандартный OT для PCD"""
        batch_size = similarity_matrix.size(0) // 2  # Учитываем удвоенный размер
        num_prototypes = similarity_matrix.size(1)
        
        a = torch.ones(2 * batch_size, 1, device=self.device) / (2 * batch_size)
        b = torch.ones(num_prototypes, 1, device=self.device) / num_prototypes
        
        # Масштабирование матрицы сходства для численной стабильности
        similarity_matrix = similarity_matrix - similarity_matrix.max()
        K = torch.exp(similarity_matrix / self.epsilon)
        K = torch.clamp(K, min=self.min_epsilon, max=1.0)
        
        log_a = self._stabilized_log(a)
        log_b = self._stabilized_log(b)
        log_K = self._stabilized_log(K)
        
        for _ in range(self.num_iterations):
            log_Kv = torch.logsumexp(log_K + log_b.t(), dim=1, keepdim=True)
            log_u = log_a - log_Kv
            log_u = torch.clamp(log_u, min=-100, max=100)
            
            log_Ktu = torch.logsumexp(log_K + log_u, dim=0, keepdim=True).t()
            log_v = log_b - log_Ktu
            log_v = torch.clamp(log_v, min=-100, max=100)
            
            log_a = log_u.clone()
            log_b = log_v.clone()
        
        log_P = log_u + log_K + log_v.t()
        P = torch.exp(log_P)
        P = torch.clamp(P, min=self.min_epsilon)
        P = P / (torch.sum(P) + self.min_epsilon)
        return P
    
    def _adaptive_filling(self, trg_features, similarity_matrix):
        """Адаптивное заполнение для CCD"""
        batch_size = trg_features.size(0)
        positive_mask = similarity_matrix.max(dim=1)[0] > self.gamma
        num_positive = positive_mask.sum().item()
        num_negative = batch_size - num_positive
        
        if num_positive > batch_size // 2:
            fill_size = num_positive - num_negative
            neg_features = []
            for i in range(batch_size):
                if positive_mask[i]:
                    farthest_proto_idx = similarity_matrix[i].argmin()
                    neg_feature = 0.5 * (trg_features[i] + self.source_prototypes[farthest_proto_idx])
                    neg_features.append(neg_feature)
            if neg_features and fill_size > 0:
                neg_features = torch.stack(neg_features[:fill_size])
                trg_features = torch.cat([trg_features, neg_features], dim=0)
        elif num_negative > batch_size // 2:
            fill_size = num_negative - num_positive
            pos_features = trg_features[positive_mask]
            if pos_features.size(0) > 0 and fill_size > 0:
                pos_features = pos_features[:fill_size]
                trg_features = torch.cat([trg_features, pos_features], dim=0)
        
        return trg_features
    
    def _update_memory(self, trg_features):
        """Обновление очереди памяти"""
        with torch.no_grad():
            batch_size = trg_features.size(0)
            if self.memory_ptr + batch_size > self.memory_size:
                remaining = self.memory_size - self.memory_ptr
                self.memory_queue[self.memory_ptr:] = trg_features[:remaining].detach().clone()
                self.memory_ptr = 0
                if batch_size - remaining > 0:
                    self.memory_queue[self.memory_ptr:batch_size - remaining] = trg_features[remaining:].detach().clone()
                    self.memory_ptr = batch_size - remaining
            else:
                self.memory_queue[self.memory_ptr:self.memory_ptr + batch_size] = trg_features.detach().clone()
                self.memory_ptr = (self.memory_ptr + batch_size) % self.memory_size
    
    def _get_nearest_neighbors(self, trg_features):
        """Получение ближайших соседей из очереди памяти"""
        with torch.no_grad():
            if self.memory_ptr == 0:
                return trg_features.detach().clone()
            
            distances = torch.cdist(trg_features.detach(), self.memory_queue[:self.memory_ptr])
            nn_indices = distances.argmin(dim=1)
            return self.memory_queue[nn_indices].detach().clone()
    
    def update(self, src_inputs, src_labels, trg_inputs):
        """
        Обновление модели на одном батче
        """
        print("\nupdate:")
        print(f"Source inputs shape: {src_inputs.shape}")
        print(f"Target inputs shape: {trg_inputs.shape}")
        
        # Получаем признаки через проекционный слой
        src_features = self.encode_features(src_inputs)
        trg_features = self.encode_features(trg_inputs)
        
        print(f"\nFeatures shapes:")
        print(f"Source features: {src_features.shape}")
        print(f"Target features: {trg_features.shape}")
        print(f"Source prototypes: {self.source_prototypes.shape}")
        
        # Нормализуем признаки
        src_features = F.normalize(src_features, dim=1)
        trg_features = F.normalize(trg_features, dim=1)
        self.source_prototypes.data = F.normalize(self.source_prototypes.data, dim=1)
        
        # Вычисляем матрицу сходства
        print("\nComputing similarity matrix:")
        print(f"Target features shape: {trg_features.shape}")
        print(f"Source prototypes shape: {self.source_prototypes.shape}")
        S_st = torch.matmul(trg_features, self.source_prototypes.t())
        print(f"Similarity matrix shape: {S_st.shape}")
        
        # Вычисляем оптимальный транспорт
        P = self.compute_optimal_transport(S_st)
        print(f"Transport matrix shape: {P.shape}")
        
        # Вычисляем потери
        cls_loss = self.compute_classification_loss(src_features, src_labels)
        ot_loss = self.compute_ot_loss(P, S_st)
        
        # Общая потеря
        total_loss = cls_loss + self.hparams['lambda_ot'] * ot_loss
        
        # Обновляем веса
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'cls_loss': cls_loss.item(),
            'ot_loss': ot_loss.item(),
            'total_loss': total_loss.item()
        }
        
    def compute_classification_loss(self, features, labels):
        """
        Вычисление потери классификации
        """
        logits = self.classifier(features)
        return F.cross_entropy(logits, labels)
        
    def compute_ot_loss(self, P, S):
        """
        Вычисление потери оптимального транспорта
        """
        return -torch.sum(P * S)
        
    def compute_optimal_transport(self, S):
        """
        Вычисление оптимального транспорта
        """
        print("\ncompute_optimal_transport:")
        print(f"Input similarity matrix shape: {S.shape}")
        
        # Реализация алгоритма Sinkhorn
        epsilon = self.hparams['epsilon']
        num_iters = self.hparams['num_iters']
        
        # Инициализация
        batch_size = S.size(0)
        num_classes = S.size(1)
        
        # Нормализуем маргинальные распределения
        mu = torch.ones(batch_size, device=self.device) / batch_size
        nu = torch.ones(num_classes, device=self.device) / num_classes
        
        print(f"mu shape: {mu.shape}")
        print(f"nu shape: {nu.shape}")
        
        # Итерации Sinkhorn
        K = torch.exp(S / epsilon)
        u = torch.ones(batch_size, device=self.device)
        
        for _ in range(num_iters):
            v = nu / (torch.matmul(K.t(), u) + 1e-8)
            u = mu / (torch.matmul(K, v) + 1e-8)
        
        # Вычисление матрицы транспорта
        P = torch.diag(u) @ K @ torch.diag(v)
        P = F.normalize(P, p=1, dim=1)  # Нормализуем строки
        
        print(f"Output transport matrix shape: {P.shape}")
        return P
    
    def encode_features(self, x):
        """Извлечение признаков из входных данных"""
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.base import BaseAlgorithm
from algorithms.utils import entropy, calc_coeff, grl_hook

class DANCE(BaseAlgorithm):
    def __init__(self, backbone_class, configs, hparams, device, trg_train_size):
        super(DANCE, self).__init__(backbone_class, configs, hparams, device)
        self.trg_train_size = trg_train_size
        
        # Получаем размерность признаков из бэкбона
        feature_dim = self._get_feature_dim()
        print(f"Initial feature dimension: {feature_dim}")
        
        # Определяем тип бэкбона
        self.backbone_type = "CNN" if "CNN" in backbone_class.__name__ else ("TCNAttention" if "TCNAttention" in backbone_class.__name__ else "TCN")
        print(f"Backbone type: {self.backbone_type}")
        
        # Устанавливаем унифицированную размерность признаков
        self.unified_feature_dim = 1536
        self.intermediate_dim = 2048
        
        # Добавляем адаптивный пулинг для TCN и TCNAttention
        if self.backbone_type in ["TCN", "TCNAttention"]:
            self.adaptive_pool = nn.AdaptiveAvgPool1d(64)  # Унифицируем длину последовательности
            # Для TCN/TCNAttention входная размерность будет равна числу каналов * длину после пулинга
            tcn_input_dim = feature_dim  # Используем исходную размерность
            print(f"TCN/TCNAttention input dimension: {tcn_input_dim}")
            self.feature_transform = nn.Sequential(
                nn.Linear(tcn_input_dim, self.intermediate_dim),
                nn.BatchNorm1d(self.intermediate_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.intermediate_dim, self.unified_feature_dim),
                nn.BatchNorm1d(self.unified_feature_dim)
            ).to(device)
        else:
            self.adaptive_pool = None
            # Для CNN создаем динамический трансформатор, который будет адаптироваться под входную размерность
            self.feature_transform = None  # Будет создан в encode_features
            self.source_transform = None
            self.target_transform = None
            
        # Инициализируем классификатор с правильной размерностью входа
        self.classifier = nn.Linear(self.unified_feature_dim, configs.num_classes).to(device)
        
        # Инициализируем дискриминатор с улучшенной архитектурой
        self.discriminator = nn.Sequential(
            nn.Linear(self.unified_feature_dim, self.intermediate_dim),
            nn.BatchNorm1d(self.intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.intermediate_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        ).to(device)
        
        # Инициализируем оптимизатор с различными скоростями обучения
        self.optimizer = torch.optim.Adam([
            {"params": self.feature_extractor.parameters(), "lr": hparams.get('learning_rate', 0.001)},
            {"params": self.classifier.parameters(), "lr": hparams.get('learning_rate', 0.001) * 10},
            {"params": self.discriminator.parameters(), "lr": hparams.get('learning_rate', 0.001) * 10}
        ], weight_decay=hparams.get('weight_decay', 0.0005))
        
    def get_parameters(self):
        params = [
            {"params": self.feature_extractor.parameters()},
            {"params": self.classifier.parameters()},
            {"params": self.discriminator.parameters()}
        ]
        if self.feature_transform is not None:
            params.append({"params": self.feature_transform.parameters()})
        if self.source_transform is not None:
            params.append({"params": self.source_transform.parameters()})
        if self.target_transform is not None:
            params.append({"params": self.target_transform.parameters()})
        return params
        
    def update(self, src_x, src_y, trg_x, trg_indices=None, step=None, epoch=None, total_steps=None):
        self.train()
        self.optimizer.zero_grad()
        
        # Вычисляем адаптивный фактор
        if step is None:
            step = 0
        if total_steps is None:
            total_steps = 1000
        coeff = calc_coeff(step, max_iter=total_steps)
        
        # Извлекаем и трансформируем признаки
        src_features = self.encode_features(src_x)
        trg_features = self.encode_features(trg_x)
        
        # Нормализуем признаки
        src_features = F.normalize(src_features, p=2, dim=1)
        trg_features = F.normalize(trg_features, p=2, dim=1)
        
        # Потери классификации для исходного домена
        src_preds = self.classifier(src_features)
        cls_loss = F.cross_entropy(src_preds, src_y)
        
        # Потери дискриминации домена с градиентным реверсом
        src_features_grl = GradientReverseFunction.apply(src_features, coeff)
        trg_features_grl = GradientReverseFunction.apply(trg_features, coeff)
        
        src_domain_preds = self.discriminator(src_features_grl)
        trg_domain_preds = self.discriminator(trg_features_grl)
        
        src_domain_labels = torch.ones(src_x.size(0), 1).to(self.device)
        trg_domain_labels = torch.zeros(trg_x.size(0), 1).to(self.device)
        
        domain_loss = F.binary_cross_entropy_with_logits(
            src_domain_preds, src_domain_labels
        ) + F.binary_cross_entropy_with_logits(
            trg_domain_preds, trg_domain_labels
        )
        
        # Потери энтропии для целевого домена
        trg_preds = self.classifier(trg_features)
        entropy_loss = entropy(trg_preds).mean()
        
        # Общие потери с адаптивным взвешиванием
        loss = cls_loss + coeff * (domain_loss + 0.1 * entropy_loss)
        
        # Обновляем параметры с градиентным клиппингом
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'cls_loss': cls_loss.item(),
            'domain_loss': domain_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': loss.item()
        }

    def train(self, mode=True):
        """Переключает модель в режим обучения или оценки"""
        self.training = mode
        self.feature_extractor.train(mode)
        self.classifier.train(mode)
        self.discriminator.train(mode)
        if self.feature_transform is not None:
            self.feature_transform.train(mode)
        return self
        
    def eval(self):
        """Переключает модель в режим оценки"""
        return self.train(False)
        
    def encode_features(self, x):
        """Извлекает и трансформирует признаки из входных данных."""
        print("\n=== DEBUG: Размерности в encode_features ===")
        print(f"1. Входные данные x: {x.shape}")
        
        features = self.feature_extractor(x)
        if isinstance(features, tuple):
            features = features[0]
        print(f"2. Признаки после feature_extractor: {features.shape}")
        
        # Применяем адаптивный пулинг для TCN и TCNAttention
        if self.backbone_type in ["TCN", "TCNAttention"]:
            print(f"3. Тип бэкбона: {self.backbone_type}")
            # Преобразуем признаки в формат [batch, channels, time]
            batch_size = features.size(0)
            print(f"4. Размер батча: {batch_size}")
            
            if len(features.shape) == 2:
                print(f"5. Признаки 2D, преобразуем в 3D")
                features = features.unsqueeze(1)  # [batch, 1, time]
                print(f"6. Признаки после unsqueeze: {features.shape}")
            elif len(features.shape) == 3:
                print(f"5. Признаки 3D, меняем оси")
                features = features.transpose(1, 2)  # [batch, time, channels] -> [batch, channels, time]
                print(f"6. Признаки после transpose: {features.shape}")
            
            # Применяем адаптивный пулинг
            print(f"7. Применяем adaptive_pool")
            features = self.adaptive_pool(features)  # -> [batch, channels, 64]
            print(f"8. Признаки после adaptive_pool: {features.shape}")
            
            # Решейпим в 2D тензор
            print(f"9. Решейпим в 2D")
            features = features.reshape(batch_size, -1)  # -> [batch, channels * 64]
            print(f"10. Признаки после reshape: {features.shape}")
            
            # Проверяем и корректируем размерность перед feature_transform
            current_dim = features.size(1)
            expected_dim = self.feature_transform[0].in_features
            print(f"11. Текущая размерность: {current_dim}, Ожидаемая размерность: {expected_dim}")
            
            if current_dim != expected_dim:
                print(f"12. Корректируем размерность с {current_dim} на {expected_dim}")
                adjustment_layer = nn.Linear(current_dim, expected_dim).to(self.device)
                features = adjustment_layer(features)
                print(f"13. Признаки после корректировки: {features.shape}")
            
            # Применяем трансформацию признаков
            print(f"14. Применяем feature_transform")
            features = self.feature_transform(features)
            print(f"15. Признаки после feature_transform: {features.shape}")
        else:
            print(f"3. Тип бэкбона: CNN")
            if len(features.shape) == 3:
                print(f"4. Признаки 3D, решейпим в 2D")
                features = features.reshape(features.size(0), -1)
                print(f"5. Признаки после reshape: {features.shape}")
            
            # Создаем или обновляем трансформаторы для CNN
            input_dim = features.size(1)
            print(f"6. Входная размерность: {input_dim}")
            
            # Определяем, какой трансформатор использовать на основе размерности входа
            if input_dim == 1536:  # WISDM
                if self.source_transform is None:
                    print(f"7. Создаем source_transform")
                    self.source_transform = nn.Sequential(
                        nn.Linear(input_dim, self.intermediate_dim),
                        nn.BatchNorm1d(self.intermediate_dim),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(self.intermediate_dim, self.unified_feature_dim),
                        nn.BatchNorm1d(self.unified_feature_dim)
                    ).to(self.device)
                    self.optimizer.add_param_group({"params": self.source_transform.parameters(), 
                                                  "lr": self.hparams.get('learning_rate', 0.001) * 5})
                self.feature_transform = self.source_transform
            elif input_dim == 448:  # HHAR_SA
                if self.target_transform is None:
                    print(f"7. Создаем target_transform")
                    self.target_transform = nn.Sequential(
                        nn.Linear(input_dim, self.intermediate_dim),
                        nn.BatchNorm1d(self.intermediate_dim),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(self.intermediate_dim, self.unified_feature_dim),
                        nn.BatchNorm1d(self.unified_feature_dim)
                    ).to(self.device)
                    self.optimizer.add_param_group({"params": self.target_transform.parameters(), 
                                                  "lr": self.hparams.get('learning_rate', 0.001) * 5})
                self.feature_transform = self.target_transform
            else:
                # Для других размерностей создаем новый трансформатор
                print(f"7. Создаем новый трансформатор для размерности {input_dim}")
                self.feature_transform = nn.Sequential(
                    nn.Linear(input_dim, self.intermediate_dim),
                    nn.BatchNorm1d(self.intermediate_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(self.intermediate_dim, self.unified_feature_dim),
                    nn.BatchNorm1d(self.unified_feature_dim)
                ).to(self.device)
                self.optimizer.add_param_group({"params": self.feature_transform.parameters(), 
                                              "lr": self.hparams.get('learning_rate', 0.001) * 5})
            
            # Применяем трансформацию
            print(f"8. Применяем feature_transform")
            features = self.feature_transform(features)
            print(f"9. Признаки после feature_transform: {features.shape}")
        
        print("=== КОНЕЦ ОТЛАДКИ ===\n")
        return features

class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None 
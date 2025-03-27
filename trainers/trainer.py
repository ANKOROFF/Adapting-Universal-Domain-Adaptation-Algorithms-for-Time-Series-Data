import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
import collections
import logging
from sklearn.metrics import classification_report, accuracy_score
from dataloader.dataloader import data_generator, few_shot_data_generator, generator_percentage_of_data
from configs.config_factory import get_dataset_class
from configs.hparams import get_hparams_class
from algorithms.utils import fix_randomness, copy_Files, starting_logs, save_checkpoint, _calc_metrics
from algorithms.utils import calc_dev_risk, calculate_risk
from algorithms.algorithms import get_algorithm_class
from algorithms.RAINCOAT import RAINCOAT
from algorithms.OVANet import OVANet
from algorithms.DANCE import DANCE
from algorithms.UAN import UAN
from algorithms.UniOT import UniOT
from models.models import get_backbone_class
from algorithms.utils import AverageMeter
from sklearn.metrics import f1_score
from data.data_manager import DataManager
from models.model_factory import get_backbone_class
from algorithms.algorithm_factory import get_algorithm_class
from algorithms.utils import weights_init
from utils.logger import setup_logger
from torch import optim
torch.backends.cudnn.benchmark = True  
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)        

def weights_init(m):
    """
    Initialize network weights using different initialization strategies
    based on the layer type.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname != 'SpectralConv1d':
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname == 'SpectralConv1d':
        scale = 1.0 / (m.in_channels * m.out_channels)
        nn.init.uniform_(m.weights1, -scale, scale)
        nn.init.uniform_(m.weights2, -scale, scale)

class EarlyStopping:
    """
    Early stopping implementation to prevent overfitting and save the best model.
    """
    def __init__(self, patience=49, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_h_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.best_model_state = None
        self.best_optimizer_state = None
        
    def __call__(self, val_loss, model, optimizer, epoch, h_score, fpath):
        if self.best_h_score is None:
            self.best_h_score = h_score
            self.save_checkpoint(model, optimizer, epoch, fpath)
        elif h_score <= self.best_h_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_h_score = h_score
            self.save_checkpoint(model, optimizer, epoch, fpath)
            self.counter = 0
            if self.verbose:
                print(f'New best H-score: {self.best_h_score:.4f}')
        
    def save_checkpoint(self, model, optimizer, epoch, fpath):
        """
        Save the best model checkpoint with its optimizer state.
        """
        self.best_model_state = model.state_dict()
        self.best_optimizer_state = optimizer.state_dict()
        self.best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.best_model_state,
            'optimizer_state_dict': self.best_optimizer_state,
            'h_score': self.best_h_score,
        }, fpath)
        
    def load_best_model(self, model, optimizer):
        """
        Load the best model and optimizer states.
        """
        model.load_state_dict(self.best_model_state)
        optimizer.load_state_dict(self.best_optimizer_state)
        return self.best_epoch, self.best_h_score

class Trainer:
    """
    Main training class that handles model training, evaluation, and visualization.
    Implements various domain adaptation methods and training strategies.
    """
    def __init__(self, source_dataset, target_dataset, source_domain, target_domain, backbone, da_method, device='cpu', verbose=True, **kwargs):
        """
        Инициализация тренера
        """
        self.device = device  # Используем переданный device
        self.verbose = verbose
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.backbone = backbone
        self.da_method = da_method
        
        if self.verbose:
            print(f"Initializing trainer for {da_method} on {source_dataset}->{target_dataset} datasets")
            print(f"Using {backbone} backbone on device: {self.device}")
        
        # Set up logging
        self.logger = setup_logger(__name__, kwargs.get('save_dir', 'results'))
        
        # Load dataset configs
        self.source_configs = get_dataset_class(source_dataset)()
        self.target_configs = get_dataset_class(target_dataset)()
        
        # Load hyperparameters
        self.hparams = get_hparams_class(source_dataset)()
        
        if self.verbose:
            print(f"Dataset configs loaded:")
            print(f"Source: {self.source_configs.dataset_name}")
            print(f"Target: {self.target_configs.dataset_name}")
            print(f"Input channels: {self.source_configs.input_channels}")
            print(f"Sequence length: {self.source_configs.sequence_len}")
            print(f"Number of classes: {self.source_configs.num_classes}")
        
        # Training parameters
        self.num_epochs = kwargs.get('num_epochs', 100)
        self.batch_size = kwargs.get('batch_size', 32)
        self.log_interval = kwargs.get('log_interval', 10)
        self.save_dir = kwargs.get('save_dir', 'results')
        
        if self.verbose:
            print(f"\nTraining parameters:")
            print(f"Number of epochs: {self.num_epochs}")
            print(f"Batch size: {self.batch_size}")
            print(f"Save directory: {self.save_dir}")
        
        # Initialize paths
        os.makedirs(self.save_dir, exist_ok=True)
        self.fpath = os.path.join(self.save_dir, f'{da_method}_model.pth')
        
        # Initialize algorithm
        backbone_fe = get_backbone_class(backbone)
        algorithm_class = get_algorithm_class(da_method)
        
        if self.verbose:
            print(f"\nInitializing data loaders...")
        
        # Initialize data loaders for source and target domains with Windows-specific settings
        self.source_manager = DataManager(self.source_configs, self.batch_size)
        self.target_manager = DataManager(self.target_configs, self.batch_size)
        
        # Устанавливаем параметры для Windows
        self.source_manager.num_workers = 0  # Отключаем многопоточность
        self.target_manager.num_workers = 0  # Отключаем многопоточность
        
        self.src_train_dl, self.src_test_dl = self.source_manager.get_source_loaders(self.source_domain)
        self.trg_train_dl, self.trg_test_dl = self.target_manager.get_target_loaders(self.target_domain)
        
        if self.verbose:
            print(f"Source domain: {self.source_domain}")
            print(f"Target domain: {self.target_domain}")
            print(f"Source train samples: {len(self.src_train_dl.dataset)}")
            print(f"Source test samples: {len(self.src_test_dl.dataset)}")
            print(f"Target train samples: {len(self.trg_train_dl.dataset)}")
            print(f"Target test samples: {len(self.trg_test_dl.dataset)}")
            print(f"\nInitializing {da_method} algorithm...")
        
        # Инициализируем алгоритм
        algorithm_params = {
            'configs': self.source_configs,
            'hparams': self.hparams.alg_hparams[da_method.upper()],
            'device': self.device
        }
        
        if da_method.upper() in ['DANCE', 'RAINCOAT']:
            algorithm_params['trg_train_size'] = len(self.trg_train_dl.dataset)
            algorithm_params['backbone_class'] = backbone_fe
        else:
            algorithm_params['backbone_class'] = backbone_fe
            
        self.algorithm = algorithm_class(**algorithm_params)
            
        print("Initialization complete!")
        
        # Инициализируем оптимизатор
        self.optimizer = optim.Adam(
            self.algorithm.parameters(),
            lr=self.hparams.alg_hparams[da_method.upper()]['learning_rate']
        )

    def train(self):
        """
        Основной цикл обучения
        """
        best_h_score = 0
        early_stopping = EarlyStopping(patience=49, verbose=True)
        
        for epoch in range(self.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch [{epoch+1}/{self.num_epochs}]")
            print(f"{'='*50}")
            
            self.algorithm.train()
            total_loss = 0
            batch_losses = []
            
            # Загружаем все данные сразу
            print("\nЗагрузка данных...")
            src_data_list = []
            trg_data_list = []
            
            for src_data in self.src_train_dl:
                src_data_list.append((src_data[0].to(self.device), src_data[1].to(self.device)))
            for trg_data in self.trg_train_dl:
                trg_data_list.append((trg_data[0].to(self.device), trg_data[1].to(self.device)))
            
            num_batches = min(len(src_data_list), len(trg_data_list))
            print(f"Загружено {num_batches} батчей")
            
            # Обучаем на батчах
            print("\nНачало обучения...")
            for batch_idx in range(num_batches):
                src_inputs, src_labels = src_data_list[batch_idx]
                trg_inputs, _ = trg_data_list[batch_idx]
                
                # Обновляем модель
                losses = self.algorithm.update(src_inputs, src_labels, trg_inputs)
                
                # Суммируем потери
                batch_loss = sum(losses.values())
                total_loss += batch_loss
                batch_losses.append(batch_loss)
                
                # Логируем потери каждые 5 батчей
                if self.verbose and batch_idx % 5 == 0:
                    print(f"\nBatch [{batch_idx+1}/{num_batches}]")
                    print(f"Текущие потери:")
                    for k, v in losses.items():
                        print(f"  {k}: {v:.4f}")
                    print(f"Средняя потеря: {total_loss/(batch_idx+1):.4f}")
            
            # Оцениваем производительность
            print("\nОценка производительности...")
            trg_acc, trg_f1, h_score = self.evaluate(self.src_test_dl, self.trg_test_dl)
            
            # Выводим результаты эпохи
            print(f"\nРезультаты эпохи [{epoch+1}/{self.num_epochs}]:")
            print(f"Точность на целевом домене: {trg_acc:.4f}")
            print(f"F1-score на целевом домене: {trg_f1:.4f}")
            print(f"H-score: {h_score:.4f}")
            print(f"Средняя потеря: {total_loss/num_batches:.4f}")
            print(f"Минимальная потеря в батче: {min(batch_losses):.4f}")
            print(f"Максимальная потеря в батче: {max(batch_losses):.4f}")
            
            # Проверяем early stopping
            early_stopping(total_loss/num_batches, self.algorithm, self.optimizer, epoch, h_score, self.fpath)
            
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
            
            print(f"\n{'='*50}")
        
        # Загружаем лучшую модель
        best_epoch, best_h_score = early_stopping.load_best_model(self.algorithm, self.optimizer)
        print(f"\nЗагружена лучшая модель с эпохи {best_epoch}")
        print(f"Лучший H-score: {best_h_score:.4f}")
                    
        return best_h_score

    def visualize(self):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        feature_extractor.eval()
        self.trg_pred_labels = np.array([])

        self.trg_true_labels = np.array([])
        self.trg_all_features = []
        self.src_true_labels = np.array([])
        self.src_all_features = []
        with torch.no_grad():
            # for data, labels in self.trg_test_dl:
            for data, labels in self.trg_train_dl:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)
                features = feature_extractor(data)
                self.trg_all_features.append(features.cpu().numpy())
                self.trg_true_labels = np.append(self.trg_true_labels, labels.data.cpu().numpy())
        
            
            for data, labels in self.src_train_dl:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)
                features = feature_extractor(data)
                self.src_all_features.append(features.cpu().numpy())
                self.src_true_labels = np.append(self.src_true_labels, labels.data.cpu().numpy())
            self.src_all_features = np.vstack(self.src_all_features)
            self.trg_all_features = np.vstack(self.trg_all_features)
        
    
    def compute_h_score(self, src_true_labels, src_pred_labels, trg_true_labels, trg_pred_labels):
        """Вычисляет H-score для оценки производительности модели"""
        # Вычисляем точность на исходном домене
        src_acc = accuracy_score(src_true_labels, src_pred_labels)
        
        # Вычисляем точность на целевом домене
        trg_acc = accuracy_score(trg_true_labels, trg_pred_labels)
        
        # Вычисляем H-score
        if src_acc + trg_acc == 0:
            return 0.0
            
        h_score = 2 * (src_acc * trg_acc) / (src_acc + trg_acc)
        return h_score

    def evaluate(self, src_dl, trg_dl):
        """Оценка модели на исходном и целевом доменах"""
        self.algorithm.eval()
        
        # Если это OVANet, используем его собственный метод evaluate
        if isinstance(self.algorithm, OVANet):
            return self.algorithm.evaluate(src_dl, trg_dl)
            
        # Если это UniOT, используем специальную обработку
        if isinstance(self.algorithm, UniOT):
            src_features_list = []
            src_labels_list = []
            src_preds_list = []
            trg_features_list = []
            trg_labels_list = []
            trg_preds_list = []
            
            with torch.no_grad():
                # Оценка на исходном домене
                print("\nОценка на исходном домене:")
                for batch in src_dl:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    
                    # Получаем признаки и предсказания
                    features = self.algorithm.encode_features(x)
                    logits = self.algorithm.classifier(features)
                    preds = torch.argmax(logits, dim=1)
                    
                    src_features_list.append(features)
                    src_labels_list.append(y)
                    src_preds_list.append(preds)
                
                # Оценка на целевом домене
                print("\nОценка на целевом домене:")
                for batch in trg_dl:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    
                    # Получаем признаки и предсказания
                    features = self.algorithm.encode_features(x)
                    logits = self.algorithm.classifier(features)
                    preds = torch.argmax(logits, dim=1)
                    
                    trg_features_list.append(features)
                    trg_labels_list.append(y)
                    trg_preds_list.append(preds)
            
            # Объединяем все батчи
            src_features = torch.cat(src_features_list, dim=0)
            src_labels = torch.cat(src_labels_list, dim=0)
            src_preds = torch.cat(src_preds_list, dim=0)
            trg_features = torch.cat(trg_features_list, dim=0)
            trg_labels = torch.cat(trg_labels_list, dim=0)
            trg_preds = torch.cat(trg_preds_list, dim=0)
            
            # Вычисляем метрики
            src_acc = accuracy_score(src_labels.cpu().numpy(), src_preds.cpu().numpy())
            src_f1 = f1_score(src_labels.cpu().numpy(), src_preds.cpu().numpy(), average='weighted')
            trg_acc = accuracy_score(trg_labels.cpu().numpy(), trg_preds.cpu().numpy())
            trg_f1 = f1_score(trg_labels.cpu().numpy(), trg_preds.cpu().numpy(), average='weighted')
            
            # Вычисляем H-score
            h_score = 2 * (src_acc * trg_acc) / (src_acc + trg_acc)
            
            print(f"\nМетрики:")
            print(f"Исходный домен - Accuracy: {src_acc:.4f}, F1: {src_f1:.4f}")
            print(f"Целевой домен - Accuracy: {trg_acc:.4f}, F1: {trg_f1:.4f}")
            print(f"H-score: {h_score:.4f}")
            
            return trg_acc, trg_f1, h_score
            
        # Если это DANCE, UAN или RAINCOAT , используем специальную обработку
        if isinstance(self.algorithm, (DANCE, RAINCOAT, UAN)):
            src_features_list = []
            src_labels_list = []
            src_preds_list = []
            trg_features_list = []
            trg_labels_list = []
            trg_preds_list = []
            
            with torch.no_grad():
                # Оценка на исходном домене
                print("\nОценка на исходном домене:")
                for batch in src_dl:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    
                    # Получаем признаки и предсказания
                    features = self.algorithm.encode_features(x)
                    if isinstance(self.algorithm, RAINCOAT):
                        features = self.algorithm.projection(features)
                    logits = self.algorithm.classifier(features)
                    preds = torch.argmax(logits, dim=1)
                    
                    src_features_list.append(features)
                    src_labels_list.append(y)
                    src_preds_list.append(preds)
                
                # Оценка на целевом домене
                print("\nОценка на целевом домене:")
                for batch in trg_dl:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    
                    # Получаем признаки и предсказания
                    features = self.algorithm.encode_features(x)
                    if isinstance(self.algorithm, RAINCOAT):
                        features = self.algorithm.projection(features)
                    logits = self.algorithm.classifier(features)
                    preds = torch.argmax(logits, dim=1)
                    
                    trg_features_list.append(features)
                    trg_labels_list.append(y)
                    trg_preds_list.append(preds)
            
            # Объединяем все батчи
            src_features = torch.cat(src_features_list, dim=0)
            src_labels = torch.cat(src_labels_list, dim=0)
            src_preds = torch.cat(src_preds_list, dim=0)
            trg_features = torch.cat(trg_features_list, dim=0)
            trg_labels = torch.cat(trg_labels_list, dim=0)
            trg_preds = torch.cat(trg_preds_list, dim=0)
            
            # Вычисляем метрики
            src_acc = accuracy_score(src_labels.cpu().numpy(), src_preds.cpu().numpy())
            src_f1 = f1_score(src_labels.cpu().numpy(), src_preds.cpu().numpy(), average='weighted')
            trg_acc = accuracy_score(trg_labels.cpu().numpy(), trg_preds.cpu().numpy())
            trg_f1 = f1_score(trg_labels.cpu().numpy(), trg_preds.cpu().numpy(), average='weighted')
            
            # Вычисляем H-score
            h_score = 2 * (src_acc * trg_acc) / (src_acc + trg_acc)
            
            print(f"\nМетрики:")
            print(f"Исходный домен - Accuracy: {src_acc:.4f}, F1: {src_f1:.4f}")
            print(f"Целевой домен - Accuracy: {trg_acc:.4f}, F1: {trg_f1:.4f}")
            print(f"H-score: {h_score:.4f}")
            
            return trg_acc, trg_f1, h_score
        
        # Для остальных методов используем стандартную обработку
        src_features_list = []
        src_labels_list = []
        src_preds_list = []
        trg_features_list = []
        trg_labels_list = []
        trg_preds_list = []
        
        with torch.no_grad():
            # Оценка на исходном домене
            print("\nОценка на исходном домене:")
            for batch in src_dl:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                # Получаем признаки и предсказания
                features = self.algorithm.feature_extractor(x)
                if hasattr(self.algorithm, 'feature_transform'):
                    features = self.algorithm.feature_transform(features)
                if hasattr(self.algorithm, 'projection'):
                    features = self.algorithm.projection(features)
                
                logits = self.algorithm.classifier(features)
                preds = torch.argmax(logits, dim=1)
                
                src_features_list.append(features)
                src_labels_list.append(y)
                src_preds_list.append(preds)
            
            # Оценка на целевом домене
            print("\nОценка на целевом домене:")
            for batch in trg_dl:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                # Получаем признаки и предсказания
                features = self.algorithm.feature_extractor(x)
                if hasattr(self.algorithm, 'feature_transform'):
                    features = self.algorithm.feature_transform(features)
                if hasattr(self.algorithm, 'projection'):
                    features = self.algorithm.projection(features)
                
                logits = self.algorithm.classifier(features)
                preds = torch.argmax(logits, dim=1)
                
                trg_features_list.append(features)
                trg_labels_list.append(y)
                trg_preds_list.append(preds)
        
        # Объединяем все батчи
        src_features = torch.cat(src_features_list, dim=0)
        src_labels = torch.cat(src_labels_list, dim=0)
        src_preds = torch.cat(src_preds_list, dim=0)
        trg_features = torch.cat(trg_features_list, dim=0)
        trg_labels = torch.cat(trg_labels_list, dim=0)
        trg_preds = torch.cat(trg_preds_list, dim=0)
        
        # Вычисляем метрики
        src_acc = accuracy_score(src_labels.cpu().numpy(), src_preds.cpu().numpy())
        src_f1 = f1_score(src_labels.cpu().numpy(), src_preds.cpu().numpy(), average='weighted')
        trg_acc = accuracy_score(trg_labels.cpu().numpy(), trg_preds.cpu().numpy())
        trg_f1 = f1_score(trg_labels.cpu().numpy(), trg_preds.cpu().numpy(), average='weighted')
        
        # Вычисляем H-score
        h_score = 2 * (src_acc * trg_acc) / (src_acc + trg_acc)
        
        print(f"\nМетрики:")
        print(f"Исходный домен - Accuracy: {src_acc:.4f}, F1: {src_f1:.4f}")
        print(f"Целевой домен - Accuracy: {trg_acc:.4f}, F1: {trg_f1:.4f}")
        print(f"H-score: {h_score:.4f}")
        
        return trg_acc, trg_f1, h_score

    def get_configs(self):
        dataset_class = get_dataset_class(self.source_dataset)
        hparams_class = get_hparams_class(self.source_dataset)
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id):
        # Get data loaders
        data_manager = DataManager(self.source_configs, self.batch_size)
        self.src_train_dl, self.src_test_dl = data_manager.get_source_loaders(src_id)
        self.trg_train_dl, self.trg_test_dl = data_manager.get_target_loaders(trg_id)
        
        # Initialize DANCE algorithm if needed
        if self.da_method == 'DANCE' and self.algorithm is None:
            backbone_fe = get_backbone_class(self.backbone)
            algorithm_class = get_algorithm_class(self.da_method)
            trg_train_size = len(self.trg_train_dl.dataset)
            self.algorithm = algorithm_class(backbone_fe, self.source_configs, self.hparams.alg_hparams[self.da_method], self.device, trg_train_size)
            self.algorithm.apply(weights_init)

    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def avg_result(self, df, name):
        # Создаем таблицу с лучшими результатами по каждому сценарию
        best_results = df.groupby('scenario').agg({
            'accuracy': 'max',
            'f1': 'max',
            'H-score': 'max'
        }).round(2)
        
        # Создаем таблицу со средними значениями и стандартными отклонениями
        mean_results = df.groupby('scenario').agg({
            'accuracy': ['mean', 'std'],
            'f1': ['mean', 'std'],
            'H-score': ['mean', 'std']
        }).round(2)
        
        # Сохраняем таблицы в CSV
        best_path = os.path.join(self.exp_log_dir, 'best_results.csv')
        mean_path = os.path.join(self.exp_log_dir, 'mean_results.csv')
        
        best_results.to_csv(best_path)
        mean_results.to_csv(mean_path)
        
        # Выводим результаты в консоль
        print("\nЛучшие результаты по сценариям:")
        print(best_results)
        print("\nСредние значения и стандартные отклонения по сценариям:")
        print(mean_results)
        
        # Возвращаем общие средние значения для итоговой строки
        return (float(df['accuracy'].mean()), float(df['accuracy'].std()),
                float(df['f1'].mean()), float(df['f1'].std()),
                float(df['H-score'].mean()), float(df['H-score'].std()))

    
        
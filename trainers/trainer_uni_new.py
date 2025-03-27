import torch
import torch.nn.functional as F
import os
import copy
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats
from algorithms.utils import fix_randomness, copy_Files, starting_logs, save_checkpoint, load_checkpoint
from algorithms.algorithm_factory import get_algorithm_class
from models.models import get_backbone_class
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, f1_score
from dataloader.uni_dataloader import data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from algorithms.utils import _calc_metrics, calculate_risk
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
import torch.nn as nn
from algorithms.base import BaseAlgorithm
from algorithms.utils import entropy, calc_coeff, grl_hook

torch.backends.cudnn.benchmark = True  
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)        

class cross_domain_trainer(object):
    """
    This class contain the main training functions for our AdAtime
    """
    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device

        self.run_description = args.experiment_description
        self.experiment_description = args.experiment_description

        self.best_acc = 0
        # paths
        self.home_path = os.getcwd()
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.create_save_dir()

        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels

        # Specify number of hparams
        self.default_hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}
        # Initialize hparams
        self.hparams = self.default_hparams

    def train(self, run_id=0):
        """Train the model"""
        print("\nНачало обучения...")
        
        # Фиксируем случайность
        fix_randomness(run_id)
        
        # Инициализируем backbone
        backbone_class = get_backbone_class(self.backbone)
        
        # Настраиваем логирование
        run_name = f"{self.run_description}"
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, run_name)
        os.makedirs(self.exp_log_dir, exist_ok=True)
        copy_Files(self.exp_log_dir)
        
        # Создаем DataFrame для результатов
        df_a = pd.DataFrame(columns=['scenario', 'run_id', 'accuracy', 'f1', 'H-score'])
        
        # Получаем сценарии из конфигурации
        scenarios = self.dataset_configs.scenarios
        
        # Инициализируем лучшие метрики
        self.best_acc = 0
        self.best_f1 = 0
        self.best_h_score = 0
        
        # Проходим по всем сценариям
        for scenario in scenarios:
            src_id, trg_id = scenario
            try:
                print(f"\nRun ID: {run_id}")
                print(f"{'='*45}")
                
                # Настраиваем логирование для сценария
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, 
                                                              self.exp_log_dir, src_id, trg_id, run_id)
                self.fpath = os.path.join(self.home_path, self.scenario_log_dir, 'backbone.pth')
                self.cpath = os.path.join(self.home_path, self.scenario_log_dir, 'classifier.pth')
                
                # Загружаем данные
                self.load_data(src_id, trg_id)
                
                # Получаем класс алгоритма через фабрику
                algorithm_class = get_algorithm_class(self.da_method)
                
                # Инициализируем алгоритм с правильными параметрами
                base_params = {
                    'backbone_class': backbone_class,
                    'configs': self.dataset_configs,
                    'hparams': self.hparams,
                    'device': self.device
                }
                
                if self.da_method == 'DANCE':
                    base_params['trg_train_size'] = len(self.trg_train_dl.dataset)
                
                self.algorithm = algorithm_class(**base_params)
                
                # Перемещаем алгоритм на нужное устройство
                self.algorithm = self.algorithm.to(self.device)
                
                print(f"Инициализирован алгоритм {self.da_method}")
                
                # Проверяем метки
                print("\n=== Label Check ===")
                print("Source train labels:", np.unique(self.src_train_dl.dataset.y_data))
                print("Source test labels:", np.unique(self.src_test_dl.dataset.y_data))
                print("Target train labels:", np.unique(self.trg_train_dl.dataset.y_data))
                print("Target test labels:", np.unique(self.trg_test_dl.dataset.y_data))
                
                # Загружаем веса feature_extractor
                if hasattr(self.algorithm.feature_extractor, 'load_state_dict'):
                    try:
                        feature_extractor_state_dict = torch.load(self.fpath)
                        self.algorithm.feature_extractor.load_state_dict(feature_extractor_state_dict)
                        print("Веса feature_extractor успешно загружены")
                    except Exception as e:
                        print(f"Ошибка при загрузке весов feature_extractor: {str(e)}")
                
                # Загружаем веса classifier
                if hasattr(self.algorithm.classifier, 'load_state_dict'):
                    try:
                        classifier_state_dict = torch.load(self.cpath)
                        # Преобразуем ключи state_dict для соответствия структуре модели
                        new_state_dict = {}
                        for k, v in classifier_state_dict.items():
                            if k.startswith('0.'):
                                new_state_dict[k[2:]] = v
                            else:
                                new_state_dict[k] = v
                        self.algorithm.classifier.load_state_dict(new_state_dict)
                        print("Веса classifier успешно загружены")
                    except Exception as e:
                        print(f"Ошибка при загрузке весов классификатора: {str(e)}")
                
                # Обучение
                for epoch in range(1, self.hparams["num_epochs"] + 1):
                    print(f"\nEpoch {epoch}/{self.hparams['num_epochs']}")
                    
                    # Обучаем модель
                    self.algorithm.train()
                    for batch_idx, ((src_x, src_y, src_idx), (trg_x, _, trg_idx)) in enumerate(zip(self.src_train_dl, self.trg_train_dl)):
                        src_x = src_x.float().to(self.device)
                        src_y = src_y.long().to(self.device)
                        trg_x = trg_x.float().to(self.device)
                        
                        # Унифицированный вызов update для всех алгоритмов
                        if self.da_method == 'DANCE':
                            trg_idx = trg_idx.clone().detach().to(self.device)
                            losses = self.algorithm.update(
                                src_x, src_y, trg_x, trg_idx,
                                step=batch_idx + (epoch-1) * len(self.src_train_dl),
                                epoch=epoch,
                                total_steps=len(self.src_train_dl)
                            )
                        else:
                            losses = self.algorithm.update(src_x, src_y, trg_x)
                        
                        if batch_idx % 10 == 0:
                            loss_str = " ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
                            print(f"Batch {batch_idx}, {loss_str}")
                    
                    # Оцениваем модель
                    print("\nEvaluating model...")
                    if self.da_method == 'RAINCOAT':
                        # Для RAINCOAT используем специальный метод оценки
                        acc, f1, h_score = self.eval(final=True)
                    elif self.da_method == 'DANCE':
                        # Для DANCE используем evaluate_dance
                        acc, f1, h_score = self.evaluate_dance(self.trg_test_dl, self.src_test_dl)
                    elif self.da_method == 'UAN':
                        # Для UAN используем evaluate_uan
                        acc, f1, h_score = self.evaluate_uan(self.trg_test_dl, self.src_test_dl)
                    elif self.da_method == 'OVANet':
                        # Для OVANet используем evaluate_ovanet
                        acc, f1, h_score = self.evaluate_ovanet(self.trg_test_dl, self.src_test_dl)
                    elif self.da_method == 'UniOT':
                        # Для UniOT используем evaluate_uniot
                        acc, f1, h_score = self.evaluate_uniot(self.trg_test_dl, self.src_test_dl)
                    else:
                        # Для остальных методов используем evaluate_tfac
                        acc, f1, h_score = self.evaluate_tfac(self.algorithm.feature_extractor, 
                                                            self.algorithm.classifier,
                                                            self.trg_test_dl, 
                                                            self.src_test_dl)
                    
                    print(f"Epoch {epoch} Results:")
                    print(f"Accuracy: {acc:.4f}")
                    print(f"F1: {f1:.4f}")
                    print(f"H-score: {h_score:.4f}")
                    
                    # Сохраняем лучшую модель
                    if f1 > self.best_f1:
                        self.best_f1 = f1
                        self.best_acc = acc
                        self.best_h_score = h_score
                        torch.save(self.algorithm.feature_extractor.state_dict(), self.fpath)
                        torch.save(self.algorithm.classifier.state_dict(), self.cpath)
                
                # Фаза коррекции для RAINCOAT
                if self.da_method == 'RAINCOAT':
                    print("\n=== Correction Phase ===")
                    # Загружаем веса feature_extractor
                    feature_extractor_state_dict = torch.load(self.fpath)
                    self.algorithm.feature_extractor.load_state_dict(feature_extractor_state_dict)
                    # Загружаем веса classifier
                    classifier_state_dict = torch.load(self.cpath)
                    self.algorithm.classifier.load_state_dict(classifier_state_dict)
                    for epoch in range(1, 3):  # 2 эпохи коррекции
                        for batch_idx, ((src_x, src_y, src_idx), (trg_x, _, trg_idx)) in enumerate(zip(self.src_train_dl, self.trg_train_dl)):
                            src_x = src_x.float().to(self.device)
                            src_y = src_y.long().to(self.device)
                            trg_x = trg_x.float().to(self.device)
                            self.algorithm.correct(src_x, src_y, trg_x)
                
                # Финальная оценка
                if self.da_method == 'RAINCOAT':
                    acc, f1, h_score = self.eval(final=True)
                else:
                    acc, f1, h_score = self.evaluate_dance(self.trg_test_dl, self.src_test_dl, final=True)
                
                # Сохраняем результаты
                log = {'scenario': f"{src_id}→{trg_id}",
                      'run_id': run_id,
                      'accuracy': acc,
                      'f1': f1,
                      'H-score': h_score}
                df_a = df_a.append(log, ignore_index=True)
                
                print(f"\nFinal Results for Run {run_id}:")
                print(f"Accuracy: {acc:.4f}")
                print(f"F1: {f1:.4f}")
                print(f"H-score: {h_score:.4f}")
                
            except Exception as e:
                print(f"Error in scenario {src_id}→{trg_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Сохраняем результаты
        df_a.to_csv(os.path.join(self.exp_log_dir, 'results.csv'), index=False)
        
        # Создаем таблицу результатов
        self.create_results_table()
        
        # Выводим сводную таблицу
        print("\nРезультаты {}:".format(self.dataset))
        print(f"Accuracy: {df_a['accuracy'].mean():.2f}%")
        print(f"F1-score: {df_a['f1'].mean():.4f}")
        print(f"H-score: {df_a['H-score'].mean():.4f}")
        
        # Выводим таблицу результатов
        print("\n" + "="*80)
        print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("="*80)
        print(f"{'Метод':<10} | {'WISDM 4→15'}")
        print("-"*80)
        
        # Загружаем результаты всех методов
        results_file = os.path.join(self.save_dir, 'all_methods_results.csv')
        all_results = {}
        if os.path.exists(results_file):
            try:
                results_df = pd.read_csv(results_file)
                for _, row in results_df.iterrows():
                    all_results[row['method']] = {
                        'accuracy': row['accuracy'],
                        'f1': row['f1'],
                        'h_score': row['h_score']
                    }
            except Exception as e:
                print(f"Warning: Could not load previous results: {str(e)}")
        
        # Обновляем результаты текущего метода
        all_results[self.da_method] = {
            'accuracy': df_a['accuracy'].mean(),
            'f1': df_a['f1'].mean(),
            'h_score': df_a['H-score'].mean()
        }
        
        # Сохраняем обновленные результаты
        results_to_save = []
        methods = ['UAN', 'DANCE', 'OVANet', 'UniOT', 'RAINCOAT']
        for method in methods:
            if method in all_results:
                results = all_results[method]
                results_to_save.append({
                    'method': method,
                    'accuracy': results['accuracy'],
                    'f1': results['f1'],
                    'h_score': results['h_score']
                })
                print(f"{method:<10} | A:{results['accuracy']:.1f} F1:{results['f1']:.3f} H:{results['h_score']:.3f}")
            else:
                print(f"{method:<10} | A:-- F1:-- H:--")
                
        # Сохраняем результаты в файл
        pd.DataFrame(results_to_save).to_csv(results_file, index=False)
        print("="*80)

    def detect_private(self, d1, d2, tar_uni_label, c_list):
        print("\n=== Подробная отладка detect_private ===")
        print(f"Initial labels shape: {tar_uni_label.shape}")
        print(f"Unique labels: {np.unique(tar_uni_label)}")
        
        try:
            # Проверяем входные данные
            if np.isnan(d1).any() or np.isnan(d2).any():
                print("Warning: NaN detected in input distances")
                print(f"NaN в d1: {np.isnan(d1).sum()} значений")
                print(f"NaN в d2: {np.isnan(d2).sum()} значений")
                d1 = np.nan_to_num(d1, nan=0.0)
                d2 = np.nan_to_num(d2, nan=0.0)
                
            if not hasattr(self, 'src_label') or self.src_label is None:
                print("Warning: src_label not initialized")
                return 0.0, 0.0, 0.0
                
            # Вычисляем разницу между расстояниями
            diff = np.abs(d2 - d1)
            
            # Копируем метки для предсказаний
            if self.trg_pred_labels is None:
                self.trg_pred_labels = np.copy(tar_uni_label)
            
            # Сохраняем истинные метки
            if self.trg_true_labels is None:
                self.trg_true_labels = np.copy(tar_uni_label)
            
            print(f"Diff shape: {diff.shape}")
            print(f"Diff statistics: min={diff.min():.4f}, max={diff.max():.4f}, mean={diff.mean():.4f}")
            print(f"NaN в diff: {np.isnan(diff).sum()} значений")
            
            # Проверяем приватные классы в исходных метках
            private_classes = np.setdiff1d(np.unique(tar_uni_label), np.unique(self.src_label))
            if len(private_classes) > 0:
                print(f"Found private classes in original labels: {private_classes}")
                for pc in private_classes:
                    mask = tar_uni_label == pc
                    self.trg_pred_labels[mask] = -1
                    print(f"Marked {np.sum(mask)} samples of class {pc} as private")
            
            # Используем тест Дипа для определения приватных классов
            for i in range(6):  # Предполагаем, что классов 6, как в вашем примере
                cat = np.where(self.trg_pred_labels == i)
                cc = diff[cat]
                
                if cc.shape[0] > 3:  # Проверяем, достаточно ли образцов для теста
                    dip, pval = diptest.diptest(diff[cat])
                    print(f"\nClass {i}:")
                    print(f"Number of samples: {cc.shape[0]}")
                    print(f"Dip test value: {dip:.4f}, p-value: {pval:.4f}")
                    
                    if pval < 0.05:  # Если p-value < 0.05, класс содержит приватные образцы
                        print(f"Class {i} contains private samples (p-value < 0.05)")
                        # Используем KMeans для разделения на кластеры
                        kmeans = KMeans(n_clusters=2, random_state=0).fit(cc.reshape(-1, 1))
                        # Определяем приватные образцы как те, что дальше от центра
                        private_samples = kmeans.labels_ == np.argmax(kmeans.cluster_centers_)
                        # Обновляем метки
                        self.trg_pred_labels[cat[0][private_samples]] = -1
                        print(f"Marked {np.sum(private_samples)} samples as private")
            
            # Вычисляем метрики
            accuracy = accuracy_score(self.trg_true_labels, self.trg_pred_labels)
            f1 = f1_score(self.trg_true_labels, self.trg_pred_labels, average='weighted')
            
            # Вычисляем H-score
            h_score = 0.0
            if hasattr(self, 'trg_true_labels') and hasattr(self, 'trg_pred_labels'):
                private_true = self.trg_true_labels == -1
                private_pred = self.trg_pred_labels == -1
                if np.any(private_true) or np.any(private_pred):
                    h_score = f1_score(private_true, private_pred, average='binary')
            
            return accuracy, f1, h_score
            
        except Exception as e:
            print(f"Error in detect_private: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0, 0.0

    def preprocess_labels(self, source_loader, target_loader):
        trg_y = copy.deepcopy(target_loader.dataset.y_data)
        src_y = source_loader.dataset.y_data
        
        # Сохраняем метки исходного домена
        self.src_label = src_y
        
        print("\n=== Debug Information for preprocess_labels ===")
        print("Source labels:", np.unique(src_y))
        print("Target labels before:", np.unique(trg_y))
        
        # Находим только реально существующие классы
        existing_classes = np.unique(trg_y)
        source_classes = np.unique(src_y)
        
        print("Existing classes in target:", existing_classes)
        print("Classes in source:", source_classes)
        
        # Помечаем как приватные только те классы, которые есть в target, но отсутствуют в source
        private_classes = np.setdiff1d(existing_classes, source_classes)
        print("Private classes:", private_classes)
        
        # Маркируем приватные классы как -1
        for pc in private_classes:
            mask = trg_y == pc
            trg_y[mask] = -1
            print(f"Marked {np.sum(mask)} samples of class {pc} as private")
        
        print("Target labels after:", np.unique(trg_y))
        print("Label distribution after:", np.bincount(trg_y[trg_y != -1]))
        
        return trg_y, private_classes


    def learn_t(self,d1,d2):
        print("\n=== Debug Information for learn_t ===")
        try:
            # Проверяем входные данные
            if np.isnan(d1).any() or np.isnan(d2).any():
                print("Warning: NaN detected in input distances")
                d1 = np.nan_to_num(d1, nan=0.0)
                d2 = np.nan_to_num(d2, nan=0.0)
                
            diff = np.abs(d2-d1)
            c_list = []
            
            # Проверяем метки
            if not hasattr(self, 'trg_train_dl') or not hasattr(self.trg_train_dl, 'dataset'):
                print("Error: trg_train_dl or dataset not initialized")
                return [1e10] * 6
                
            y_data = self.trg_train_dl.dataset.y_data
            if isinstance(y_data, torch.Tensor):
                y_data = y_data.cpu().numpy()
                
            print(f"Target labels shape: {y_data.shape}")
            print(f"Unique labels: {np.unique(y_data)}")
            
            for i in range(6):
                cat = np.where(y_data==i)
                cc = diff[cat]
                
                if cc.shape[0] > 3:
                    dip, pval = diptest.diptest(diff[cat])
                    print(f"\nClass {i}:")
                    print(f"Number of samples: {cc.shape[0]}")
                    print(f"Dip test value: {dip:.4f}, p-value: {pval:.4f}")
                    
                    if dip < 0.05:
                        kmeans = KMeans(n_clusters=2, random_state=0, max_iter=5000, n_init=50, init="random").fit(diff[cat].reshape(-1, 1))
                        c = max(kmeans.cluster_centers_)
                    else:
                        c = 1e10
                else:
                    print(f"Warning: Not enough samples for class {i}")
                    c = 1e10
                    
                c_list.append(c)
                
            print(f"Final thresholds: {c_list}")
            return c_list
            
        except Exception as e:
            print(f"Error in learn_t: {str(e)}")
            import traceback
            traceback.print_exc()
            return [1e10] * 6

    def calc_distance(self, len_y, dataloader):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        feature_extractor.eval()
        classifier.eval()
        
        print("\n=== Debug Information for calc_distance ===")
        
        # Получаем веса классификатора для прототипов
        proto = classifier[0].weight.data  # Берем веса первого линейного слоя
        print(f"Prototype shape: {proto.shape}")
        
        trg_drift = np.zeros(len_y)
        batch_count = 0
        error_count = 0
        
        with torch.no_grad():
            for batch_idx, (data, labels, trg_index) in enumerate(dataloader):
                batch_count += 1
                try:
                    data = data.float().to(self.device)
                    labels = labels.view((-1)).long().to(self.device)
                    
                    # Проверяем входные данные
                    if torch.isnan(data).any():
                        print(f"Warning: NaN detected in batch {batch_idx}")
                        error_count += 1
                        continue
                    
                    # Получаем features
                    features, _ = feature_extractor(data)
                    if torch.isnan(features).any():
                        print(f"Warning: NaN detected in features for batch {batch_idx}")
                        error_count += 1
                        continue
                    
                    # Получаем предсказания
                    predictions = classifier(features)
                    pred_label = torch.argmax(predictions, dim=1)
                    
                    # Нормализуем features
                    features_norm = F.normalize(features, p=2, dim=1)
                    
                    # Нормализуем прототипы
                    proto_norm = F.normalize(proto, p=2, dim=1)
                    
                    # Выбираем прототипы для предсказанных классов
                    selected_protos = proto_norm[pred_label]
                    
                    # Вычисляем косинусное сходство
                    similarity = torch.sum(features_norm * selected_protos, dim=1)
                    
                    # Проверяем на NaN
                    if torch.isnan(similarity).any():
                        print(f"Warning: NaN detected in similarity for batch {batch_idx}")
                        error_count += 1
                        continue
                        
                    trg_drift[trg_index] = similarity.cpu().numpy()
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {str(e)}")
                    error_count += 1
                    continue
        
        # Проверяем финальный результат
        if np.isnan(trg_drift).any():
            print("Warning: NaN detected in final trg_drift")
            # Заменяем NaN на 0
            trg_drift = np.nan_to_num(trg_drift, nan=0.0)
            
        print(f"\nProcessing Summary:")
        print(f"Total batches processed: {batch_count}")
        print(f"Error batches: {error_count}")
        print(f"Success rate: {(batch_count - error_count) / batch_count * 100:.2f}%")
        print(f"Final trg_drift shape: {trg_drift.shape}")
        print(f"Unique values in trg_drift: {np.unique(trg_drift)}")
        
        return trg_drift

    def evaluate_dance(self, target_loader, source_loader=None, final=False):
        print("\n=== Подробная отладка evaluate_dance ===")
        try:
            # Инициализируем списки для хранения предсказаний и истинных меток
            total_pred = []
            total_true = []
            src_pred = []
            src_true = []

            # Оцениваем на целевом домене
            # Переключаем в режим оценки без вызова eval()
            self.algorithm.feature_extractor.eval()
            self.algorithm.classifier.eval()
            
            with torch.no_grad():
                for batch_idx, (x, y, _) in enumerate(target_loader):
                    x = x.float().to(self.device)
                    y = y.long().to(self.device)
                    
                    # Получаем признаки и предсказания
                    features = self.algorithm.feature_extractor(x)
                    if isinstance(features, tuple):
                        features = features[0]
                    predictions = self.algorithm.classifier(features)
                    
                    # Вычисляем энтропию для определения неизвестных классов
                    probs = F.softmax(predictions, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                    
                    # Помечаем как -1 образцы с высокой энтропией
                    threshold = 0.5  # Порог энтропии
                    unknown_mask = entropy > threshold
                    pred = torch.argmax(predictions, dim=1)
                    pred[unknown_mask] = -1
                    
                    total_pred.extend(pred.cpu().numpy())
                    total_true.extend(y.cpu().numpy())

            # Оцениваем на исходном домене, если он предоставлен
            if source_loader is not None:
                with torch.no_grad():
                    for batch_idx, (x, y, _) in enumerate(source_loader):
                        x = x.float().to(self.device)
                        y = y.long().to(self.device)
                        
                        features = self.algorithm.feature_extractor(x)
                        if isinstance(features, tuple):
                            features = features[0]
                        predictions = self.algorithm.classifier(features)
                        pred = torch.argmax(predictions, dim=1)
                        
                        src_pred.extend(pred.cpu().numpy())
                        src_true.extend(y.cpu().numpy())

            # Преобразуем в numpy массивы
            total_pred = np.array(total_pred)
            total_true = np.array(total_true)
            src_pred = np.array(src_pred) if src_pred else None
            src_true = np.array(src_true) if src_true else None

            # Сохраняем метки для последующего использования
            self.trg_pred_labels = total_pred
            self.trg_true_labels = total_true
            if src_pred is not None and src_true is not None:
                self.src_pred_labels = src_pred
                self.src_true_labels = src_true

            print("\n=== Подробная отладка метрик ===")
            print(f"Target predictions shape: {total_pred.shape}")
            print(f"Target true labels shape: {total_true.shape}")
            if src_pred is not None:
                print(f"Source predictions shape: {src_pred.shape}")
                print(f"Source true labels shape: {src_true.shape}")

            # Вычисляем метрики для целевого домена
            # Исключаем приватные классы (-1) из вычисления accuracy
            mask = total_true != -1
            target_acc = accuracy_score(total_true[mask], total_pred[mask])
            target_f1 = f1_score(total_true[mask], total_pred[mask], average='weighted')

            # Вычисляем H-score
            h_score = 0.0
            if src_pred is not None and src_true is not None:
                # Вычисляем accuracy для исходного домена
                src_acc = accuracy_score(src_true, src_pred)
                # H-score как гармоническое среднее
                if src_acc > 0 and target_acc > 0:
                    h_score = 2 * (src_acc * target_acc) / (src_acc + target_acc)

            print(f"Target Accuracy: {target_acc:.4f}")
            print(f"Target F1-score: {target_f1:.4f}")
            if src_pred is not None:
                print(f"Source Accuracy: {src_acc:.4f}")
            print(f"H-score: {h_score:.4f}")

            return target_acc * 100, target_f1, h_score

        except Exception as e:
            print(f"Error in evaluate_dance: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0, 0.0

    def evaluate_tfac(self, feature_extractor, classifier, target_loader, source_loader):
        print("\n=== Evaluating TFAC ===")
        feature_extractor.eval()
        classifier.eval()
        
        trg_pred_labels = []
        trg_true_labels = []
        src_pred_labels = []
        src_true_labels = []
        
        with torch.no_grad():
            # Оценка на целевом домене
            for batch_idx, (data, labels, _) in enumerate(target_loader):
                data = data.float().to(self.device)
                labels = labels.long().to(self.device)
                
                # Получаем признаки и проецируем их
                features = feature_extractor(data)
                if isinstance(features, tuple):
                    features = features[0]
                
                # Преобразуем размерность если нужно
                if len(features.shape) == 3:  # [batch, channels, length]
                    features = features.reshape(features.size(0), -1)  # [batch, channels * length]
                
                # Проецируем признаки в нужную размерность
                features = self.algorithm.projection(features)  # [batch, projection_dim]
                
                logits = classifier(features)
                pred_labels = torch.argmax(logits, dim=1)
                
                trg_pred_labels.extend(pred_labels.cpu().numpy())
                trg_true_labels.extend(labels.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"Обработано {batch_idx} батчей целевого домена")
            
            # Оценка на исходном домене
            for batch_idx, (data, labels, _) in enumerate(source_loader):
                data = data.float().to(self.device)
                labels = labels.long().to(self.device)
                
                # Получаем признаки и проецируем их
                features = feature_extractor(data)
                if isinstance(features, tuple):
                    features = features[0]
                
                # Преобразуем размерность если нужно
                if len(features.shape) == 3:  # [batch, channels, length]
                    features = features.reshape(features.size(0), -1)  # [batch, channels * length]
                
                # Проецируем признаки в нужную размерность
                features = self.algorithm.projection(features)  # [batch, projection_dim]
                
                logits = classifier(features)
                pred_labels = torch.argmax(logits, dim=1)
                
                src_pred_labels.extend(pred_labels.cpu().numpy())
                src_true_labels.extend(labels.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"Обработано {batch_idx} батчей исходного домена")
        
        # Сохраняем предсказания и истинные метки
        self.trg_pred_labels = np.array(trg_pred_labels)
        self.trg_true_labels = np.array(trg_true_labels)
        
        # Вычисляем метрики
        trg_acc = accuracy_score(self.trg_true_labels, self.trg_pred_labels)
        trg_f1 = f1_score(self.trg_true_labels, self.trg_pred_labels, average='weighted')
        
        src_acc = accuracy_score(src_true_labels, src_pred_labels)
        src_f1 = f1_score(src_true_labels, src_pred_labels, average='weighted')
        
        # Вычисляем H-score
        h_score = 2 * (trg_acc * src_acc) / (trg_acc + src_acc)
        
        print(f"\nРезультаты оценки:")
        print(f"Точность целевого домена: {trg_acc:.4f}")
        print(f"F1-score целевого домена: {trg_f1:.4f}")
        print(f"H-score: {h_score:.4f}")
        
        return trg_acc * 100, trg_f1, h_score

    def H_score(self):
        print("\n=== Подробная отладка H_score ===")
        
        try:
            # Проверяем инициализацию меток
            if not hasattr(self, 'trg_true_labels') or not hasattr(self, 'trg_pred_labels'):
                print("Инициализируем метки целевого домена...")
                # Получаем метки целевого домена
                self.trg_true_labels = self.trg_test_dl.dataset.y_data.numpy()
                self.trg_pred_labels = np.zeros_like(self.trg_true_labels)
                
            if not hasattr(self, 'src_true_labels') or not hasattr(self, 'src_pred_labels'):
                print("Инициализируем метки исходного домена...")
                # Получаем метки исходного домена
                self.src_true_labels = self.src_test_dl.dataset.y_data.numpy()
                self.src_pred_labels = np.zeros_like(self.src_true_labels)
            
            print(f"Форма истинных меток целевого домена: {self.trg_true_labels.shape}")
            print(f"Форма предсказанных меток целевого домена: {self.trg_pred_labels.shape}")
            print(f"Форма истинных меток исходного домена: {self.src_true_labels.shape}")
            print(f"Форма предсказанных меток исходного домена: {self.src_pred_labels.shape}")
            
            # Вычисляем точность для исходного домена
            src_acc = accuracy_score(self.src_true_labels, self.src_pred_labels)
            print(f"\nТочность исходного домена: {src_acc:.4f}")
            
            # Вычисляем точность для целевого домена
            # Исключаем приватные классы (метка -1)
            trg_mask = self.trg_true_labels != -1
            if np.sum(trg_mask) > 0:
                trg_acc = accuracy_score(
                    self.trg_true_labels[trg_mask],
                    self.trg_pred_labels[trg_mask]
                )
            else:
                trg_acc = 0.0
            print(f"Точность целевого домена: {trg_acc:.4f}")
            
            # Вычисляем H-score как гармоническое среднее
            if src_acc + trg_acc > 0:
                h_score = 2 * (src_acc * trg_acc) / (src_acc + trg_acc)
            else:
                h_score = 0.0
            
            print(f"Итоговый H-score: {h_score:.4f}")
            return h_score
            
        except Exception as e:
            print(f"ОШИБКА в H_score: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0

    def save_result(self, df_a, df_c=None):
        print("\n" + "="*80)
        print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
        print("="*80)
        
        try:
            # Проверяем наличие данных
            if df_a.empty:
                print("ОШИБКА: Нет данных для сохранения")
                return
                
            print("\nСырые данные:")
            print(df_a)
            
            # Группируем по сценарию и вычисляем средние значения
            result = df_a.groupby('scenario', as_index=False).agg({
                'accuracy': 'mean',
                'f1': 'mean',
                'H-score': 'mean'
            })
            
            # Форматируем вывод
            print("\nРезультаты по сценариям:")
            print("-"*80)
            for _, row in result.iterrows():
                print(f"Сценарий {row['scenario']}:")
                print(f"Accuracy: {row['accuracy']:.2f}%")
                print(f"F1-score: {row['f1']:.4f}")
                print(f"H-score: {row['H-score']:.4f}")
                print("-"*40)
            
            # Вычисляем общие средние значения
            print("\nОбщие средние значения:")
            print(f"Средняя точность: {result['accuracy'].mean():.2f}%")
            print(f"Средний F1-score: {result['f1'].mean():.4f}")
            print(f"Средний H-score: {result['H-score'].mean():.4f}")
            
            # Сохраняем результаты
            save_path = os.path.join(self.exp_log_dir, 'results.csv')
            result.to_csv(save_path, index=False)
            print(f"\nРезультаты сохранены в: {save_path}")
            
            # Создаем сводную таблицу
            print("\n" + "="*80)
            print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
            print("="*80)
            print(f"{'Метод':<10} | {'Сценарий':<15} | {'Accuracy':<8} | {'F1':<8} | {'H-score':<8}")
            print("-"*80)
            
            for _, row in result.iterrows():
                print(f"{self.da_method:<10} | {row['scenario']:<15} | "
                      f"{row['accuracy']:>8.2f} | {row['f1']:>8.4f} | {row['H-score']:>8.4f}")
            
            print("="*80)
            
        except Exception as e:
            print(f"Ошибка при сохранении результатов: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id):
        self.src_train_dl, self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs,
                                                             self.hparams)
        self.trg_train_dl, self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs,
                                                             self.hparams)


    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir) 

    def eval(self, final=False):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        feature_extractor.eval()
        classifier.eval()
        
        data = copy.deepcopy(self.trg_test_dl.dataset.x_data).float().to(self.device)
        labels = self.trg_test_dl.dataset.y_data.view((-1)).long().to(self.device)
        
        # Получаем признаки
        features = feature_extractor(data)
        if isinstance(features, tuple):
            features = features[0]
            
        # Преобразуем размерность если нужно
        if len(features.shape) == 3:  # [batch, channels, length]
            features = features.reshape(features.size(0), -1)  # [batch, channels * length]
            
        # Проецируем признаки в нужную размерность
        features = self.algorithm.projection(features)  # [batch, projection_dim]
        
        predictions = classifier(features)
        pred_labels = torch.argmax(predictions, dim=1)
        
        acc = accuracy_score(labels.cpu().numpy(), pred_labels.cpu().numpy()) * 100
        f1 = f1_score(labels.cpu().numpy(), pred_labels.cpu().numpy(), average='weighted')
        h_score = self.H_score() if final else 0.0
        
        return acc, f1, h_score 

    def evaluate_uan(self, trg_test_dl, src_test_dl):
        """Оценка для метода UAN"""
        try:
            self.algorithm.eval()
            with torch.no_grad():
                total_acc = 0
                total_samples = 0
                all_preds = []
                all_labels = []
                
                # Оцениваем на целевом домене
                for x, y, _ in trg_test_dl:
                    x = x.float().to(self.device)
                    y = y.long().to(self.device)
                    
                    features = self.algorithm.feature_extractor(x)
                    if isinstance(features, tuple):
                        features = features[0]
                    if len(features.shape) == 3:
                        features = features.reshape(features.size(0), -1)
                    
                    # Нормализуем признаки
                    features = F.normalize(features, p=2, dim=1)
                    
                    # Проверяем на NaN/Inf с более мягким порогом
                    if torch.isnan(features).sum() / features.numel() > 0.5 or \
                       torch.isinf(features).sum() / features.numel() > 0.5:
                        print("Предупреждение: Слишком много NaN/Inf значений в признаках целевого домена")
                        continue
                    
                    # Заменяем оставшиеся NaN/Inf значения
                    features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # Получаем предсказания
                    logits = self.algorithm.classifier(features)
                    
                    # Проверяем логиты на NaN/Inf
                    if torch.isnan(logits).sum() / logits.numel() > 0.5 or \
                       torch.isinf(logits).sum() / logits.numel() > 0.5:
                        print("Предупреждение: Слишком много NaN/Inf значений в логитах целевого домена")
                        continue
                    
                    # Заменяем оставшиеся NaN/Inf значения
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # Получаем предсказания и обновляем метрики
                    preds = logits.argmax(dim=1)
                    correct = (preds == y).sum().item()
                    total_acc += correct
                    total_samples += y.size(0)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
                
                if total_samples == 0:
                    print("Предупреждение: Нет валидных предсказаний для целевого домена")
                    return 0.0, 0.0, 0.0
                
                # Вычисляем метрики с балансировкой классов
                trg_acc = total_acc / total_samples
                f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
                
                # Оцениваем на исходном домене
                src_correct = 0
                src_total = 0
                
                for x, y, _ in src_test_dl:
                    x = x.float().to(self.device)
                    y = y.long().to(self.device)
                    
                    # Извлекаем и нормализуем признаки
                    features = self.algorithm.feature_extractor(x)
                    if isinstance(features, tuple):
                        features = features[0]
                    if len(features.shape) == 3:
                        features = features.reshape(features.size(0), -1)
                    
                    # Нормализуем признаки
                    features = F.normalize(features, p=2, dim=1)
                    
                    # Проверяем на NaN/Inf с более мягким порогом
                    if torch.isnan(features).sum() / features.numel() > 0.5 or \
                       torch.isinf(features).sum() / features.numel() > 0.5:
                        print("Предупреждение: Слишком много NaN/Inf значений в признаках исходного домена")
                        continue
                    
                    # Заменяем оставшиеся NaN/Inf значения
                    features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # Получаем предсказания
                    logits = self.algorithm.classifier(features)
                    
                    # Проверяем логиты на NaN/Inf
                    if torch.isnan(logits).sum() / logits.numel() > 0.5 or \
                       torch.isinf(logits).sum() / logits.numel() > 0.5:
                        print("Предупреждение: Слишком много NaN/Inf значений в логитах исходного домена")
                        continue
                    
                    # Заменяем оставшиеся NaN/Inf значения
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    preds = torch.argmax(logits, dim=1)
                    src_correct += (preds == y).sum().item()
                    src_total += y.size(0)
                
                if src_total == 0:
                    print("Предупреждение: Нет валидных предсказаний для исходного домена")
                    return 0.0, 0.0, 0.0
                
                src_acc = src_correct / src_total
                
                # Вычисляем H-score с защитой от деления на ноль
                if src_acc + trg_acc == 0:
                    print("Предупреждение: Нулевая точность в обоих доменах")
                    h_score = 0.0
                else:
                    h_score = 2 * (src_acc * trg_acc) / (src_acc + trg_acc + 1e-10)
                
                # Проверяем финальные результаты на NaN/Inf
                if np.isnan([trg_acc, f1, h_score]).any() or np.isinf([trg_acc, f1, h_score]).any():
                    print("Предупреждение: NaN/Inf значения в финальных результатах")
                    return 0.0, 0.0, 0.0
                
                print("\n=== Подробная отладка H_score ===")
                print(f"Форма истинных меток целевого домена: {np.array(all_labels).shape}")
                print(f"Форма предсказанных меток целевого домена: {np.array(all_preds).shape}")
                print(f"Форма истинных меток исходного домена: {y.shape}")
                print(f"Форма предсказанных меток исходного домена: {preds.shape}")
                print(f"\nТочность исходного домена: {src_acc:.4f}")
                print(f"Точность целевого домена: {trg_acc:.4f}")
                print(f"Итоговый H-score: {h_score:.4f}\n")
                
                return trg_acc * 100, f1, h_score
                
        except Exception as e:
            print(f"Ошибка в evaluate_uniot: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0, 0.0

    def evaluate_ovanet(self, trg_test_dl, src_test_dl):
        """Оценка для метода OVANet"""
        try:
            self.algorithm.eval()
            with torch.no_grad():
                total_acc = 0
                total_samples = 0
                all_preds = []
                all_labels = []
                
                # Оцениваем на целевом домене
                for x, y, _ in trg_test_dl:
                    x = x.float().to(self.device)
                    y = y.long().to(self.device)
                    
                    features = self.algorithm.feature_extractor(x)
                    if isinstance(features, tuple):
                        features = features[0]
                    if len(features.shape) == 3:
                        features = features.reshape(features.size(0), -1)
                    
                    # Проверяем на NaN
                    if torch.isnan(features).any():
                        continue
                    
                    logits = self.algorithm.classifier(features)
                    if torch.isnan(logits).any():
                        continue
                        
                    preds = logits.argmax(dim=1)
                    correct = (preds == y).sum().item()
                    total_acc += correct
                    total_samples += y.size(0)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
                
                if total_samples == 0:
                    print("Предупреждение: Нет валидных предсказаний")
                    return 0.0, 0.0, 0.0
                
                trg_acc = total_acc / total_samples
                f1 = f1_score(all_labels, all_preds, average='weighted')
                
                # Оцениваем на исходном домене
                src_correct = 0
                src_total = 0
                for x, y, _ in src_test_dl:
                    x = x.float().to(self.device)
                    y = y.long().to(self.device)
                    
                    features = self.algorithm.feature_extractor(x)
                    if isinstance(features, tuple):
                        features = features[0]
                    if len(features.shape) == 3:
                        features = features.reshape(features.size(0), -1)
                    
                    # Проверяем на NaN
                    if torch.isnan(features).any():
                        continue
                    
                    logits = self.algorithm.classifier(features)
                    if torch.isnan(logits).any():
                        continue
                        
                    preds = logits.argmax(dim=1)
                    src_correct += (preds == y).sum().item()
                    src_total += y.size(0)
                
                if src_total == 0:
                    print("Предупреждение: Нет валидных предсказаний для исходного домена")
                    return 0.0, 0.0, 0.0
                
                src_acc = src_correct / src_total
                h_score = 2 * (src_acc * trg_acc) / (src_acc + trg_acc)
                
                # Проверяем финальные результаты на NaN
                if np.isnan(trg_acc) or np.isnan(f1) or np.isnan(h_score):
                    print("Предупреждение: NaN в финальных результатах")
                    return 0.0, 0.0, 0.0
                
                return trg_acc * 100, f1, h_score
                
        except Exception as e:
            print(f"Ошибка в evaluate_ovanet: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0, 0.0

    def create_results_table(self):
        """Создание и сохранение таблицы результатов"""
        print("\n" + "="*80)
        print("СОЗДАНИЕ ТАБЛИЦЫ РЕЗУЛЬТАТОВ")
        print("="*80)
        
        try:
            # Создаем директорию для сохранения, если её нет
            os.makedirs(self.save_dir, exist_ok=True)
            
            # Путь к файлу с результатами
            results_file = os.path.join(self.save_dir, 'all_methods_results.csv')
            
            # Загружаем существующие результаты, если они есть
            all_results = {}
            if os.path.exists(results_file):
                try:
                    results_df = pd.read_csv(results_file)
                    for _, row in results_df.iterrows():
                        # Проверяем на NaN значения и заменяем их на 0
                        if pd.isna(row['accuracy']) or pd.isna(row['f1']) or pd.isna(row['h_score']):
                            print(f"Предупреждение: NaN значения найдены в результатах метода {row['method']}, заменяем на 0")
                            all_results[row['method']] = {
                                'accuracy': 0.0,
                                'f1': 0.0,
                                'h_score': 0.0
                            }
                        else:
                            all_results[row['method']] = {
                                'accuracy': float(row['accuracy']),
                                'f1': float(row['f1']),
                                'h_score': float(row['h_score'])
                            }
                except Exception as e:
                    print(f"Предупреждение: Не удалось загрузить предыдущие результаты: {str(e)}")
            
            # Получаем результаты текущего метода
            current_results = {
                'accuracy': float(self.best_acc * 100),  # Преобразуем в проценты
                'f1': float(self.best_f1),
                'h_score': float(self.best_h_score)
            }
            
            # Проверяем на NaN значения и заменяем их на 0
            if np.isnan(current_results['accuracy']) or np.isnan(current_results['f1']) or np.isnan(current_results['h_score']):
                print(f"Предупреждение: NaN значения найдены в результатах текущего метода {self.da_method}, заменяем на 0")
                current_results = {
                    'accuracy': 0.0,
                    'f1': 0.0,
                    'h_score': 0.0
                }
            
            # Обновляем результаты текущего метода
            all_results[self.da_method] = current_results
            
            # Создаем DataFrame с результатами
            results_data = []
            methods = ['UAN', 'DANCE', 'OVANet', 'UniOT', 'RAINCOAT']
            
            for method in methods:
                if method in all_results:
                    results = all_results[method]
                    results_data.append({
                        'method': method,
                        'accuracy': results['accuracy'],
                        'f1': results['f1'],
                        'h_score': results['h_score']
                    })
                else:
                    print(f"Предупреждение: Метод {method} не найден в результатах, добавляем с нулевыми значениями")
                    results_data.append({
                        'method': method,
                        'accuracy': 0.0,
                        'f1': 0.0,
                        'h_score': 0.0
                    })
            
            # Создаем DataFrame и сохраняем его
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(results_file, index=False)
            
            # Выводим таблицу результатов
            print("\nТАБЛИЦА РЕЗУЛЬТАТОВ")
            print("-"*80)
            print(f"{'Метод':<10} | {'Accuracy':<8} | {'F1':<8} | {'H-score':<8}")
            print("-"*80)
            
            for _, row in results_df.iterrows():
                print(f"{row['method']:<10} | {row['accuracy']:>8.2f} | {row['f1']:>8.4f} | {row['h_score']:>8.4f}")
            
            print("="*80)
            print(f"Результаты сохранены в: {results_file}")
            
        except Exception as e:
            print(f"Ошибка при создании таблицы результатов: {str(e)}")
            import traceback
            traceback.print_exc() 
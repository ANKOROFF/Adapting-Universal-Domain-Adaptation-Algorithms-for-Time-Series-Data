import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn

import random
import os
import sys
import logging
import numpy as np
import pandas as pd

from shutil import copy
from datetime import datetime

from skorch import NeuralNetClassifier  # for DIV Risk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,f1_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def starting_logs(data_type, da_method, exp_log_dir, src_id, tgt_id, run_id):
    log_dir = os.path.join(exp_log_dir, src_id + "_to_" + tgt_id + "_run_" + str(run_id))
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug(f'Method:  {da_method}')
    logger.debug("=" * 45)
    logger.debug(f'Source: {src_id} ---> Target: {tgt_id}')
    logger.debug(f'Run ID: {run_id}')
    logger.debug("=" * 45)
    return logger, log_dir


def save_checkpoint(home_path, algorithm, selected_scenarios, dataset_configs, log_dir, hparams):
    save_dict = {
        "x-domains": selected_scenarios,
        "configs": dataset_configs.__dict__,
        "hparams": dict(hparams),
        "model_dict": algorithm.state_dict()
    }
    # save classification report
    save_path = os.path.join(home_path, log_dir, "checkpoint.pt")
    torch.save(save_dict, save_path)


def load_checkpoint(checkpoint_path, algorithm):
    checkpoint = torch.load(checkpoint_path)
    algorithm.load_state_dict(checkpoint["model_dict"])
    return algorithm


def weights_init(m):
    """
    Initialize network weights using Xavier initialization
    """
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def _calc_metrics(pred_labels, true_labels, log_dir, home_path, target_names):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)
    target_names = [f"class{i}" for i in range(max(len(set(pred_labels)), len(set(true_labels))))]
    # r = classification_report(true_labels, pred_labels, target_names=target_names, digits=6, output_dict=True)

    # df = pd.DataFrame(r)
    accuracy = accuracy_score(true_labels, pred_labels)
    # df["accuracy"] = accuracy
    # df = df * 100

    # # save classification report
    # file_name = "classification_report.xlsx"
    # report_Save_path = os.path.join(home_path, log_dir, file_name)
    # df.to_excel(report_Save_path)
    f1 = f1_score(true_labels, pred_labels,average='macro')
    return accuracy * 100, f1


def copy_Files(destination):
    destination_dir = os.path.join(destination, "MODEL_BACKUP_FILES")
    os.makedirs(destination_dir, exist_ok=True)
    
    # Получаем абсолютный путь к корневой директории проекта
    home_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Копируем файлы с абсолютными путями
    copy(os.path.join(home_path, "main.py"), os.path.join(destination_dir, "main.py"))
    copy(os.path.join(home_path, "algorithms", "utils.py"), os.path.join(destination_dir, "utils.py"))
    copy(os.path.join(home_path, "trainers", "trainer.py"), os.path.join(destination_dir, "trainer.py"))
    copy(os.path.join(home_path, "trainers", "same_domain_trainer.py"), os.path.join(destination_dir, "same_domain_trainer.py"))
    copy(os.path.join(home_path, "dataloader", "dataloader.py"), os.path.join(destination_dir, "dataloader.py"))
    copy(os.path.join(home_path, "models", "models.py"), os.path.join(destination_dir, "models.py"))
    copy(os.path.join(home_path, "models", "loss.py"), os.path.join(destination_dir, "loss.py"))
    copy(os.path.join(home_path, "algorithms", "algorithms.py"), os.path.join(destination_dir, "algorithms.py"))
    copy(os.path.join(home_path, "configs", "data_model_configs.py"), os.path.join(destination_dir, "data_model_configs.py"))
    copy(os.path.join(home_path, "configs", "hparams.py"), os.path.join(destination_dir, "hparams.py"))




def get_iwcv_value(weight, error):
    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, 'dimension mismatch!'
    weighted_error = weight * error
    return np.mean(weighted_error)


def get_dev_value(weight, error):
    """
    :param weight: shape [N, 1], the importance weight for N source samples in the validation set
    :param error: shape [N, 1], the error value for each source sample in the validation set
    (typically 0 for correct classification and 1 for wrong classification)
    """
    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, 'dimension mismatch!'
    weighted_error = weight * error
    cov = np.cov(np.concatenate((weighted_error, weight), axis=1), rowvar=False)[0][1]
    var_w = np.var(weight, ddof=1)
    eta = - cov / var_w
    return np.mean(weighted_error) + eta * np.mean(weight) - eta


class simple_MLP(nn.Module):
    def __init__(self, inp_units, out_units=2):
        super(simple_MLP, self).__init__()

        self.dense0 = nn.Linear(inp_units, inp_units // 2)
        self.nonlin = nn.ReLU()
        self.output = nn.Linear(inp_units // 2, out_units)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, **kwargs):
        x = self.nonlin(self.dense0(x))
        x = self.softmax(self.output(x))
        return x


def get_weight_gpu(source_feature, target_feature, validation_feature, configs, device):
    """
    :param source_feature: shape [N_tr, d], features from training set
    :param target_feature: shape [N_te, d], features from test set
    :param validation_feature: shape [N_v, d], features from validation set
    :return:
    """
    import copy
    N_s, d = source_feature.shape
    N_t, _d = target_feature.shape
    source_feature = copy.deepcopy(source_feature.detach().cpu())  # source_feature.clone()
    target_feature = copy.deepcopy(target_feature.detach().cpu())  # target_feature.clone()
    source_feature = source_feature.to(device)
    target_feature = target_feature.to(device)
    all_feature = torch.cat((source_feature, target_feature), dim=0)
    all_label = torch.from_numpy(np.asarray([1] * N_s + [0] * N_t, dtype=np.int32)).long()

    feature_for_train, feature_for_test, label_for_train, label_for_test = train_test_split(all_feature, all_label,
                                                                                            train_size=0.8)
    learning_rates = [1e-1, 5e-2, 1e-2]
    val_acc = []
    domain_classifiers = []

    for lr in learning_rates:
        domain_classifier = NeuralNetClassifier(
            simple_MLP,
            module__inp_units=configs.final_out_channels * configs.features_len,
            max_epochs=30,
            lr=lr,
            device=device,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            callbacks="disable"
        )
        domain_classifier.fit(feature_for_train.float(), label_for_train.long())
        output = domain_classifier.predict(feature_for_test)
        acc = np.mean((label_for_test.numpy() == output).astype(np.float32))
        val_acc.append(acc)
        domain_classifiers.append(domain_classifier)

    index = val_acc.index(max(val_acc))
    domain_classifier = domain_classifiers[index]

    domain_out = domain_classifier.predict_proba(validation_feature.to(device).float())
    return domain_out[:, :1] / domain_out[:, 1:] * N_s * 1.0 / N_t


def calc_dev_risk(target_model, src_train_dl, tgt_train_dl, src_valid_dl, configs, device):
    src_train_feats = target_model.feature_extractor(src_train_dl.dataset.x_data.float().to(device))
    tgt_train_feats = target_model.feature_extractor(tgt_train_dl.dataset.x_data.float().to(device))
    src_valid_feats = target_model.feature_extractor(src_valid_dl.dataset.x_data.float().to(device))
    src_valid_pred = target_model.classifier(src_valid_feats)

    dev_weights = get_weight_gpu(src_train_feats.to(device), tgt_train_feats.to(device),
                                 src_valid_feats.to(device), configs, device)
    dev_error = F.cross_entropy(src_valid_pred, src_valid_dl.dataset.y_data.long().to(device), reduction='none')
    dev_risk = get_dev_value(dev_weights, dev_error.unsqueeze(1).detach().cpu().numpy())
    # iwcv_risk = get_iwcv_value(dev_weights, dev_error.unsqueeze(1).detach().cpu().numpy())
    return dev_risk


def calculate_risk_ours(target_model, risk_dataloader, device):
    if type(risk_dataloader) == tuple:
        x_data = torch.cat((risk_dataloader[0].dataset.x_data, risk_dataloader[1].dataset.x_data), axis=0)
        y_data = torch.cat((risk_dataloader[0].dataset.y_data, risk_dataloader[1].dataset.y_data), axis=0)
    else:
        x_data = risk_dataloader.dataset.x_data
        y_data = risk_dataloader.dataset.y_data

    feat,_ = target_model.encoder(x_data.float().to(device))
    pred = target_model.classifier(feat)
    cls_loss = F.cross_entropy(pred, y_data.long().to(device))
    return cls_loss.item()

def calculate_risk(target_model, risk_dataloader, device):
    if type(risk_dataloader) == tuple:
        x_data = torch.cat((risk_dataloader[0].dataset.x_data, risk_dataloader[1].dataset.x_data), axis=0)
        y_data = torch.cat((risk_dataloader[0].dataset.y_data, risk_dataloader[1].dataset.y_data), axis=0)
    else:
        x_data = risk_dataloader.dataset.x_data
        y_data = risk_dataloader.dataset.y_data

    feat = target_model.feature_extractor(x_data.float().to(device))
    pred = target_model.classifier(feat)
    cls_loss = F.cross_entropy(pred, y_data.long().to(device))
    return cls_loss.item()


# For DIRT-T
class EMA:
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.params = self.shadow.keys()

    def __call__(self, model):
        if self.decay > 0:
            for name, param in model.named_parameters():
                if name in self.params and param.requires_grad:
                    self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)
                    param.data = self.shadow[name]

# For DANN
class EMA2():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def entropy(p):
    """
    Compute entropy of a probability distribution for each example
    """
    p = F.softmax(p, dim=1)
    return -torch.sum(p * torch.log(p + 1e-5), dim=1)

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    """
    Calculate coefficient for the gradual change
    """
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    """
    Gradient reversal hook
    """
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1